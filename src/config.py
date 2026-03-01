"""Configuration settings loaded from environment variables."""

import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: str
    target_language: str = "pt-BR"
    log_level: str = "INFO"
    max_concurrent_translations: int = 5
    gemini_model: str = "gemini-2.0-flash"

    @property
    def data_dir(self) -> Path:
        """Directory for input PDFs."""
        return Path(__file__).parent.parent / "data"

    @property
    def output_dir(self) -> Path:
        """Directory for output PDFs."""
        return Path(__file__).parent.parent / "output"


def get_settings() -> Settings:
    """Load and return application settings."""
    return Settings()


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
