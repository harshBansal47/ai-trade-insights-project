from typing import AsyncGenerator, Literal
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from pydantic import BaseModel, EmailStr, Field, field_validator

from src.core.database import get_db
from src.models.user import User, UserRead
from src.middlewares.auth import get_current_user
from src.utils.security import hash_password, verify_password, create_access_token
from src.utils.otp import create_otp, verify_otp as check_otp
from src.utils.email import send_otp_email, send_welcome_email
from src.core.config import settings

router = APIRouter(prefix="/auth", tags=["auth"])




# ═══════════════════════════════════════════════════════
# SHARED RESPONSE HELPER
# ═══════════════════════════════════════════════════════

class AuthResponse(BaseModel):
    user: UserRead
    token: str
    points: int
    message: str = "Success"

    model_config = {"from_attributes": True}


def _make_auth_response(user: User, message: str = "Success") -> AuthResponse:
    return AuthResponse(
        user=UserRead.model_validate(user),
        token=create_access_token({"sub": user.id, "email": user.email}),
        points=user.points,
        message=message,
    )


# ═══════════════════════════════════════════════════════
# REQUEST BODIES
# ═══════════════════════════════════════════════════════

class SignupRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    otp: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")

    @field_validator("password")
    @classmethod
    def strong_password(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    method: Literal["password", "otp"] = "password"
    password: str | None = None
    otp: str | None = None


class SendOtpRequest(BaseModel):
    email: EmailStr
    purpose: Literal["signup", "login", "forgot_password"]


class VerifyOtpRequest(BaseModel):
    email: EmailStr
    otp: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")
    purpose: Literal["signup", "login", "forgot_password"]


class ForgotPasswordRequest(BaseModel):
    email: EmailStr
    otp: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")
    new_password: str = Field(min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def strong_password(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class GoogleRequest(BaseModel):
    id_token: str


class OtpResponse(BaseModel):
    message: str
    expires_in: int


class VerifyOtpResponse(BaseModel):
    verified: bool
    message: str


# ═══════════════════════════════════════════════════════
# ENDPOINTS
# Every handler receives `db: AsyncSession = Depends(get_db)`.
# The singleton is accessed once per request via get_db().
# ═══════════════════════════════════════════════════════

# ── POST /auth/send-otp ───────────────────────────────────────────────────────

@router.post("/send-otp", response_model=OtpResponse)
async def send_otp(
    body: SendOtpRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate and email an OTP.
    - signup:         reject if email already exists.
    - login/forgot:   reject if email NOT found.
    """
    result = await db.execute(select(User).where(User.email == body.email.lower()))
    user = result.first()

    if body.purpose == "signup" and user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )
    if body.purpose in ("login", "forgot_password") and not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account found with this email",
        )

    otp_code = await create_otp(body.email, body.purpose)
    name = user.name if user else "there"
    await send_otp_email(body.email, name, otp_code, body.purpose)

    return OtpResponse(
        message="OTP sent to your email",
        expires_in=settings.otp_expire_minutes * 60,
    )


# ── POST /auth/verify-otp ─────────────────────────────────────────────────────

@router.post("/verify-otp", response_model=VerifyOtpResponse)
async def verify_otp_endpoint(body: VerifyOtpRequest):
    """
    Validate an OTP and store a short-lived 'verified' flag in Redis.

    Why a flag instead of re-using the raw OTP?
      check_otp() deletes the Redis key on success (single-use).
      The subsequent /signup or /forgot-password call needs proof the
      email was verified, so we store a separate flag for 5 minutes.
      The final endpoint checks this flag AND/OR the raw OTP directly.
    """
    verified = await check_otp(body.email, body.otp, body.purpose)

    if not verified:
        return VerifyOtpResponse(verified=False, message="Invalid or expired code")

    # Store a short-lived verified flag (5 minutes) for the next step
    from src.core.database import get_redis
    redis = await get_redis()
    await redis.setex(
        f"otp_verified:{body.purpose}:{body.email.lower()}",
        300,   # 5 minutes
        "1",
    )

    return VerifyOtpResponse(verified=True, message="Email verified")


# ── POST /auth/signup ─────────────────────────────────────────────────────────

@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    body: SignupRequest,
    db: AsyncSession = Depends(get_db),
):
    email = body.email.lower()

    # Race-condition safety: re-check uniqueness inside the transaction
    existing = await db.execute(select(User).where(User.email == email))
    if existing.first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # Accept EITHER the raw OTP (client sends it again) OR the verified flag
    otp_valid = await check_otp(email, body.otp, "signup")
    if not otp_valid:
        from src.core.database import get_redis
        redis = await get_redis()
        flag = await redis.get(f"otp_verified:signup:{email}")
        if not flag:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired verification code",
            )
        await redis.delete(f"otp_verified:signup:{email}")

    user = User(
        name=body.name.strip(),
        email=email,
        hashed_password=hash_password(body.password),
        is_verified=True,
        points=settings.free_signup_points,
    )
    db.add(user)
    await db.flush()  # populate user.id before building the response

    try:
        await send_welcome_email(user.email, user.name, settings.free_signup_points)
    except Exception:
        pass  # non-critical

    return _make_auth_response(
        user,
        f"Account created! {settings.free_signup_points} points added.",
    )


# ── POST /auth/login ──────────────────────────────────────────────────────────

@router.post("/login", response_model=AuthResponse)
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    email = body.email.lower()

    result = await db.execute(select(User).where(User.email == email))
    user = result.first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account deactivated",
        )

    if body.method == "password":
        if not body.password or not user.hashed_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password login not available for this account. Use Google or OTP.",
            )
        if not verify_password(body.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

    elif body.method == "otp":
        if not body.otp:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OTP is required",
            )
        valid = await check_otp(email, body.otp, "login")
        if not valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired OTP",
            )

    return _make_auth_response(user, f"Welcome back, {user.name}!")


# ── POST /auth/google ─────────────────────────────────────────────────────────

@router.post("/google", response_model=AuthResponse)
async def google_login(
    body: GoogleRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Called by NextAuth's signIn() callback after Google OAuth completes.
    Verifies the Google id_token and finds-or-creates the local user,
    all within the single `db` session provided by Depends(get_db).
    """
    # 1. Verify the Google id_token
    try:
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests

        idinfo = google_id_token.verify_oauth2_token(
            body.id_token,
            google_requests.Request(),
            settings.google_client_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid Google token: {exc}",
        ) from exc

    email    = idinfo["email"].lower()
    name     = idinfo.get("name") or email.split("@")[0].capitalize()
    verified = idinfo.get("email_verified", False)

    if not verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google account email is not verified",
        )

    # 2. Find or create user — single session, no nested context managers
    result = await db.execute(select(User).where(User.email == email))
    user = result.first()

    if not user:
        user = User(
            name=name,
            email=email,
            hashed_password=None,   # Google users authenticate via token, not password
            is_verified=True,
            points=settings.free_signup_points,
        )
        db.add(user)
        await db.flush()            

        try:
            await send_welcome_email(user.email, user.name, settings.free_signup_points)
        except Exception:
            pass

    else:
        if user.name != name:
            user.name = name
            db.add(user)

    return _make_auth_response(user, f"Welcome, {user.name}!")


# ── POST /auth/forgot-password ────────────────────────────────────────────────

@router.post("/forgot-password")
async def forgot_password(
    body: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    email = body.email.lower()

    # Accept raw OTP or the verified flag
    otp_valid = await check_otp(email, body.otp, "forgot_password")
    if not otp_valid:
        from src.core.database import get_redis
        redis = await get_redis()
        flag = await redis.get(f"otp_verified:forgot_password:{email}")
        if not flag:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired code",
            )
        await redis.delete(f"otp_verified:forgot_password:{email}")

    result = await db.execute(select(User).where(User.email == email))
    user = result.first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    user.hashed_password = hash_password(body.new_password)
    db.add(user)

    return {"message": "Password reset successfully"}


# ── GET /auth/me ──────────────────────────────────────────────────────────────

@router.get("/me")
async def get_profile(current_user: User = Depends(get_current_user)):
    """
    Returns the authenticated user's profile and current points.
    The frontend calls this to seed the NextAuth session after login.
    """
    return {
        "user":   UserRead.model_validate(current_user),
        "points": current_user.points,
    }


# ── POST /auth/logout ─────────────────────────────────────────────────────────

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """
    JWT is stateless — the token is valid until it expires.
    NextAuth clears the session cookie on the client side.
    To enable instant revocation, add a Redis denylist check in
    get_current_user() and insert the token jti here.
    """
    return {"message": "Logged out"}