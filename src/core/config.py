from functools import lru_cache
import os
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

def is_local_environment() -> bool:
    return os.getenv("ENVIRONMENT", "production").lower() in ["local", "development", "dev"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "CryptoAI Insights"
    app_env: Literal["development", "production", "test"] = "development"
    debug: bool = True
    frontend_url: str = "http://localhost:3000"

    # Database
    database_url: str =  Field("", alias="SUPABASE_DATABASE_CONNECTION_STRING")
    database_url_sync: str = ""

    # Redis / Celery
    redis_url: str = "redis://localhost:6379"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # JWT
    secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 10080  # 7 days

    # Google OAuth
    google_client_id: str = Field("", alias="GOOGLE_AUTH_CLIENT_ID")
    google_client_secret: str = Field("", alias="GOOGLE_AUTH_CLIENT_SECRET")

    # Anthropic
    anthropic_api_key: str = ""
    ai_model: str = "claude-3-5-sonnet-20241022"
    ai_max_tokens: int = 2048
    ai_temperature: float = 0.3

    # Stripe
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""

    # Email
    resend_api_key: str = Field("", alias="RESEND_API_KEY")
    email_from: str = "noreply@ai_crypto_insights.com"
    email_from_name: str = "CryptoAI Insights"

    gmail_user: str = Field("", alias="SMTP_USER")
    gmail_app_password: str = Field("", alias="SMTP_PASSWORD")

    # OTP
    otp_expire_minutes: int = 10
    otp_length: int = 6

    # Points
    free_signup_points: int = 50
    points_per_query: int = 10

    # Stripe bundle prices (in cents)
    price_50_points: int = 500
    price_150_points: int = 1200
    price_500_points: int = 3500

    


    # Binance
    binance_base_url: str = "https://api.binance.com"

    # Tasks
    task_soft_time_limit: int = 120
    task_hard_time_limit: int = 180

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def cors_origins(self) -> list[str]:
        origins = [self.frontend_url]
        if not self.is_production:
            origins += ["http://localhost:3000", "http://127.0.0.1:3000"]
        return origins


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()