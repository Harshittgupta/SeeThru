"""Backend configuration (BUILD_PLAN T54).

Env-var driven via pydantic-settings, all under the ``SEETHRU_`` prefix. Fails
fast at startup on an invalid value rather than at the first request that trips
over it.

The two settings that are safety controls, not tuning:

* ``allow_untrained`` -- default **False**. ``pretrained=True`` only loads
  ImageNet into the spatial backbone; fusion/classifier/temporal are random, so
  serving that produces confident-looking verdicts drawn from noise. The API
  refuses to come ready without real weights unless this is explicitly flipped
  for local dev.
* ``cors_origins`` -- default **empty**, never ``*``. A wildcard with credentials
  is rejected by browsers anyway, and a wildcard without them still invites any
  site to drive this API on a user's behalf.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SEETHRU_", env_file=".env", extra="ignore")

    # --- model ---
    image_weights: str = Field("", description="Path to the image checkpoint (best.pt).")
    video_weights: str = Field("", description="Path to the video checkpoint.")
    device: str = Field("auto", description="auto | cpu | cuda")
    # See the module docstring. False is the only safe default.
    allow_untrained: bool = False

    # --- upload limits (T55) ---
    max_image_bytes: int = 15 * 1024 * 1024      # 15 MB
    max_video_bytes: int = 200 * 1024 * 1024     # 200 MB
    max_video_seconds: float = 60.0
    max_image_pixels: int = 50_000_000           # ~50 MP; a decompression-bomb guard
    max_faces: int = 10

    # --- jobs (T57) ---
    job_ttl_hours: int = 24
    max_queue_size: int = 32

    # --- serving ---
    artifact_dir: str = "backend/_artifacts"
    upload_tmp_dir: str = "backend/_tmp"
    cors_origins: list[str] = Field(default_factory=list)
    rate_limit_image: str = "10/minute"
    rate_limit_video: str = "3/minute"

    @field_validator("device")
    @classmethod
    def _valid_device(cls, v: str) -> str:
        if v not in ("auto", "cpu", "cuda"):
            raise ValueError(f"device must be auto|cpu|cuda, got {v!r}")
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v):
        # Accept a comma-separated env string OR a real list.
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @field_validator("cors_origins")
    @classmethod
    def _no_wildcard(cls, v: list[str]) -> list[str]:
        if "*" in v:
            raise ValueError(
                "cors_origins must not be '*': a wildcard lets any site drive this "
                "API on a user's behalf. List the exact origins (T55)."
            )
        return v


@lru_cache
def get_settings() -> Settings:
    """Cached accessor. One Settings instance for the process."""
    return Settings()
