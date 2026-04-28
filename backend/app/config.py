from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://invplatform:invplatform@localhost:5433/invplatform"
    database_url_sync: str = "postgresql+psycopg://invplatform:invplatform@localhost:5433/invplatform"

    # Redis / Celery
    redis_url: str = "redis://localhost:6380/0"
    celery_broker_url: str = "redis://localhost:6380/0"
    celery_result_backend: str = "redis://localhost:6380/1"

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"

    # Storage (relative to backend/ cwd by default)
    data_dir: Path = Path("../data")
    videos_dir: Path = Path("../data/videos")
    frames_dir: Path = Path("../data/frames")
    thumbnails_dir: Path = Path("../data/thumbnails")

    # Frame sampling defaults
    default_min_frames_per_shot: int = 1
    default_max_inter_frame_seconds: float = 3.0

    # Subject reference set cap (farthest-point sampling)
    subject_reference_max: int = 20

    # Top-k cap for tool results
    tool_result_top_k_default: int = 50

    # ML mode — set false on the GPU box once real models are installed.
    # When true, ML tasks produce deterministic fake outputs so the DAG can
    # run end-to-end without GPU / heavy model downloads.
    use_fake_ml: bool = True

    # Server-side directory scan: comma-separated absolute paths the API is
    # allowed to scan for videos (e.g. /data/footage,/mnt/cameras).
    # The /api/ingest/scan endpoint refuses paths outside these roots.
    ingest_scan_roots: str = ""

    # HuggingFace token for gated models (SAM 3.1, Parakeet). Forwarded to
    # os.environ so transformers/huggingface_hub auto-pick it up.
    hf_token: str = ""

    @property
    def scan_roots(self) -> list[str]:
        return [p.strip() for p in self.ingest_scan_roots.split(",") if p.strip()]


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    # Push gated-model auth into os.environ so transformers / huggingface_hub
    # see it (they read the env directly, not pydantic-settings).
    import os
    if s.hf_token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = s.hf_token
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", s.hf_token)
    return s