import stripe
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.database import get_session
from app.middleware.auth import get_current_user
from app.models.user import User
from app.models.payment import (
    Payment, PaymentStatus, POINTS_BUNDLES,
    CreateSessionRequest, CreateSessionResponse,
    VerifySessionResponse, PaymentHistoryResponse, PaymentHistoryItem,
)
from app.config import settings

stripe.api_key = settings.stripe_secret_key
router = APIRouter(prefix="/payment", tags=["payment"])

# Bundle lookup
_BUNDLE_MAP = {b["id"]: b for b in POINTS_BUNDLES}


# ── POST /payment/create-stripe-session ───────────────────────────────────────

@router.post("/create-stripe-session", response_model=CreateSessionResponse)
async def create_stripe_session(
    body: CreateSessionRequest,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    bundle = _BUNDLE_MAP.get(body.bundle_id)
    if not bundle:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid bundle ID")

    success_url = body.success_url or f"{settings.frontend_url}/payment-success?session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url  = body.cancel_url  or f"{settings.frontend_url}/payment-failed"

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "unit_amount": bundle["price_cents"],
                    "product_data": {
                        "name": f"CryptoAI — {bundle['name']} ({bundle['points']} points)",
                        "description": f"{bundle['points']} AI analysis points",
                    },
                },
                "quantity": 1,
            }],
            mode="payment",
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={
                "user_id":   current_user.id,
                "bundle_id": bundle["id"],
                "points":    str(bundle["points"]),
            },
            customer_email=current_user.email,
        )
    except stripe.StripeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    # Create a pending payment record
    payment = Payment(
        user_id=current_user.id,
        stripe_session_id=session.id,
        bundle_id=bundle["id"],
        points=bundle["points"],
        amount_cents=bundle["price_cents"],
        status=PaymentStatus.PENDING,
    )
    db.add(payment)

    return CreateSessionResponse(session_id=session.id, checkout_url=session.url)


# ── GET /payment/verify/{session_id} ─────────────────────────────────────────

@router.get("/verify/{session_id}", response_model=VerifySessionResponse)
async def verify_payment(
    session_id: str,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Frontend calls this after Stripe redirects to /payment-success."""
    result = await db.exec(
        select(Payment).where(
            Payment.stripe_session_id == session_id,
            Payment.user_id == current_user.id,
        )
    )
    payment = result.first()

    if not payment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Payment not found")

    # Already processed
    if payment.status == PaymentStatus.COMPLETED:
        return VerifySessionResponse(
            success=True,
            points_added=payment.points,
            new_balance=current_user.points,
        )

    # Verify with Stripe
    try:
        session = stripe.checkout.Session.retrieve(session_id)
    except stripe.StripeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if session.payment_status != "paid":
        payment.status = PaymentStatus.FAILED
        db.add(payment)
        return VerifySessionResponse(success=False, points_added=0, new_balance=current_user.points)

    # Credit points
    current_user.points += payment.points
    payment.status = PaymentStatus.COMPLETED
    payment.completed_at = datetime.now(timezone.utc)
    db.add(current_user)
    db.add(payment)

    return VerifySessionResponse(
        success=True,
        points_added=payment.points,
        new_balance=current_user.points,
    )


# ── POST /payment/webhook ─────────────────────────────────────────────────────

@router.post("/webhook", include_in_schema=False)
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_session)):
    """
    Stripe sends checkout.session.completed here.
    This is the authoritative credit path — verify/ is just for UX.
    """
    payload = await request.body()
    sig     = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig, settings.stripe_webhook_secret)
    except stripe.SignatureVerificationError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid webhook signature")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if event["type"] != "checkout.session.completed":
        return {"received": True}

    session  = event["data"]["object"]
    meta     = session.get("metadata", {})
    user_id  = meta.get("user_id")
    bundle_id = meta.get("bundle_id")
    points   = int(meta.get("points", 0))

    if not user_id or not points:
        return {"received": True}

    # Idempotency: skip if already processed
    existing = await db.exec(
        select(Payment).where(
            Payment.stripe_session_id == session["id"],
            Payment.status == PaymentStatus.COMPLETED,
        )
    )
    if existing.first():
        return {"received": True}

    # Find or create payment record
    pay_result = await db.exec(
        select(Payment).where(Payment.stripe_session_id == session["id"])
    )
    payment = pay_result.first()

    bundle = _BUNDLE_MAP.get(bundle_id, {})
    if not payment:
        payment = Payment(
            user_id=user_id,
            stripe_session_id=session["id"],
            bundle_id=bundle_id or "unknown",
            points=points,
            amount_cents=bundle.get("price_cents", 0),
            status=PaymentStatus.PENDING,
        )
        db.add(payment)

    payment.status = PaymentStatus.COMPLETED
    payment.completed_at = datetime.now(timezone.utc)
    payment.stripe_payment_intent_id = session.get("payment_intent")
    db.add(payment)

    # Credit user
    user_result = await db.exec(select(User).where(User.id == user_id))
    user = user_result.first()
    if user:
        user.points += points
        db.add(user)

    return {"received": True}


# ── GET /payment/history ──────────────────────────────────────────────────────

@router.get("/history", response_model=PaymentHistoryResponse)
async def get_payment_history(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    result = await db.exec(
        select(Payment)
        .where(Payment.user_id == current_user.id)
        .order_by(Payment.created_at.desc())  # type: ignore[attr-defined]
        .limit(50)
    )
    payments = result.all()
    return PaymentHistoryResponse(
        items=[PaymentHistoryItem.model_validate(p) for p in payments]
    )