"""Environment-backed API configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


@dataclass(frozen=True)
class Settings:
    project_root: Path
    assets_dir: Path
    storage_dir: Path
    api_host: str
    api_port: int
    log_level: str
    cors_origins: tuple[str, ...]
    max_workers: int
    max_upload_mb: int
    max_result_states: int

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def jobs_dir(self) -> Path:
        return self.storage_dir / "jobs"


def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parent.parent
    origins = tuple(
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "").split(",")
        if origin.strip()
    )
    settings = Settings(
        project_root=project_root,
        assets_dir=Path(os.getenv("ASSEMBLY_ASSETS_DIR", str(project_root / "assets"))).resolve(),
        storage_dir=Path(os.getenv("API_STORAGE_DIR", str(project_root / ".api-data"))).resolve(),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=_positive_int("API_PORT", 8000),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        cors_origins=origins,
        max_workers=_positive_int("API_MAX_WORKERS", 1),
        max_upload_mb=_positive_int("API_MAX_UPLOAD_MB", 250),
        max_result_states=_positive_int("API_MAX_RESULT_STATES", 10000),
    )
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    return settings


settings = get_settings()
