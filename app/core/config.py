# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database (no hard-coded password here; use .env)
    DATABASE_URL: str

    # API
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Quant Trading Assistant"

    # Trading Parameters
    DEFAULT_SPARSITY: int = 15  # Number of assets in portfolio
    DEVIATION_THRESHOLD: float = 0.02  # 2% threshold for trade signals
    MAX_POSITION_SIZE: float = 0.20  # 20% max per asset

    # Data Sources
    DATA_START_DATE: str = "2020-01-01"
    DATA_END_DATE: str = "2024-10-01"

    class Config:
        env_file = ".env"
        validate_assignment = True

settings = Settings()
