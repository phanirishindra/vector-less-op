"""Configuration management using Pydantic Settings.

Loads configuration from environment variables and .env files.
Designed for low-RAM environments with sensible defaults.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Configuration (llama.cpp server)
    llm_base_url: str = Field(
        default="http://127.0.0.1:8000/v1",
        description="Base URL for the local LLM server (llama.cpp)",
    )
    llm_api_key: str = Field(
        default="sk-local",
        description="API key for LLM server (can be any string for local)",
    )
    llm_model: str = Field(
        default="qwen2.5-3b-instruct",
        description="Model identifier for the LLM",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM generation",
    )
    llm_max_tokens: int = Field(
        default=2048,
        ge=1,
        description="Maximum tokens for LLM response",
    )
    llm_timeout: float = Field(
        default=120.0,
        ge=1.0,
        description="Timeout in seconds for LLM requests",
    )

    # Token limits for processing
    max_tokens_per_chunk: int = Field(
        default=6000,
        ge=100,
        description="Maximum tokens before HTML splitting",
    )
    max_signpost_tokens: int = Field(
        default=30,
        ge=10,
        description="Maximum tokens for dense signpost",
    )
    max_context_tokens: int = Field(
        default=4096,
        ge=512,
        description="Maximum context window for retrieval",
    )

    # Crawler settings
    crawl_delay_ms: int = Field(
        default=500,
        ge=0,
        description="Delay between requests in milliseconds",
    )
    max_concurrent_requests: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent HTTP requests",
    )
    max_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum crawl depth from seed URL",
    )
    render_js: bool = Field(
        default=True,
        description="Use Playwright for JS-heavy sites",
    )
    user_agent: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        description="User agent string for requests",
    )

    # Bloom filter settings
    bloom_filter_size: int = Field(
        default=1_000_000,
        ge=1000,
        description="Expected number of URLs for Bloom filter",
    )
    bloom_filter_fp_rate: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="False positive rate for Bloom filter",
    )

    # Data storage
    data_dir: Path = Field(
        default=Path("./data"),
        description="Base directory for data storage",
    )

    # API Server
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host",
    )
    api_port: int = Field(
        default=8080,
        ge=1,
        le=65535,
        description="API server port",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    @field_validator("data_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        """Convert string to Path and ensure it exists."""
        path = Path(v) if isinstance(v, str) else v
        return path

    @property
    def raw_dir(self) -> Path:
        """Directory for raw crawled HTML."""
        path = self.data_dir / "raw"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def markdown_dir(self) -> Path:
        """Directory for converted Markdown."""
        path = self.data_dir / "markdown"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def index_dir(self) -> Path:
        """Directory for ToC JSON index files."""
        path = self.data_dir / "index"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_directories(self) -> None:
        """Create all required data directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        _ = self.raw_dir
        _ = self.markdown_dir
        _ = self.index_dir


# Global settings instance
settings = Settings()
