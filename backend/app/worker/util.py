"""Worker-side helpers: status updates, fake-ML synthesis, ffprobe."""
import hashlib
import json
import logging
import shutil
import subprocess
from uuid import UUID

import numpy as np
from sqlalchemy.orm import Session

from app.models import Video

logger = logging.getLogger(__name__)

# Defensive cap for any free-text field stored in the DB.  Even though the
# columns are now TEXT (no schema bound), runaway model output should not be
# allowed to push multi-MB blobs into rows that the agent later loads en masse.
# 32k chars ≈ ~10 minutes of dense speech or a very thorough caption.
MAX_TEXT_CHARS = 32_768


def clamp_text(text: str | None, *, source: str = "text") -> str:
    """Clip free-text ML output to MAX_TEXT_CHARS, with an ellipsis suffix.
    Logs a warning so silent truncation never goes unnoticed."""
    if text is None:
        return ""
    if len(text) <= MAX_TEXT_CHARS:
        return text
    logger.warning("clamp_text: %s exceeded %d chars (was %d) — truncating",
                   source, MAX_TEXT_CHARS, len(text))
    return text[:MAX_TEXT_CHARS] + "…"


def update_status(db: Session, video_id: UUID, *, stage: str | None = None,
                  status: str | None = None, progress_pct: int | None = None,
                  error: str | None = None) -> None:
    v: Video | None = db.get(Video, video_id)
    if v is None:
        return
    if stage is not None:
        v.stage = stage
    if status is not None:
        v.status = status
    if progress_pct is not None:
        v.progress_pct = progress_pct
    if error is not None:
        v.error = error
    db.commit()


def ffprobe(path: str) -> dict:
    """Return basic video metadata: duration, fps, width, height. Empty on failure."""
    if shutil.which("ffprobe") is None:
        return {}
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-print_format", "json",
                "-show_streams", "-show_format", path,
            ],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout
        data = json.loads(out)
        v_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
        if not v_streams:
            return {}
        s = v_streams[0]
        # avg_frame_rate is "num/den"
        num, den = (s.get("avg_frame_rate") or "0/1").split("/")
        fps = float(num) / float(den) if float(den) else 0.0
        duration = float(data.get("format", {}).get("duration") or s.get("duration") or 0)
        return {
            "duration_seconds": duration,
            "fps": fps,
            "width": int(s.get("width") or 0),
            "height": int(s.get("height") or 0),
        }
    except Exception:
        return {}


def deterministic_unit_vector(seed_bytes: bytes, dim: int = 512) -> list[float]:
    """Reproducible unit vector from a seed (for fake embeddings).
    Produces stable vectors so the same input always yields the same vector,
    useful for verifying retrieval logic end-to-end without real models."""
    seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype("float32")
    v /= np.linalg.norm(v) + 1e-9
    return v.tolist()
