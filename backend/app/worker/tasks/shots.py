"""Shot boundary detection (TransNetV2). Fake mode: uniform 5s shots."""
import logging
from uuid import UUID, uuid4

from sqlalchemy import delete

from app.config import get_settings
from app.models import Shot, Video
from app.worker.celery_app import celery_app
from app.worker.db import session_scope
from app.worker.util import update_status

logger = logging.getLogger(__name__)
settings = get_settings()


@celery_app.task(name="app.worker.tasks.shots.run", bind=True, acks_late=True)
def run(self, video_id: str) -> int:
    vid = UUID(video_id)
    with session_scope() as db:
        update_status(db, vid, stage="shots", status="processing")
        v: Video | None = db.get(Video, vid)
        if v is None:
            return 0
        duration = v.duration_seconds or 0.0
        if duration <= 0:
            return 0

        # Clear any prior shots
        db.execute(delete(Shot).where(Shot.video_id == vid))

        if settings.use_fake_ml:
            shots = _fake_shots(duration)
        else:
            update_status(db, vid, progress_pct=4)  # scanning started
            shots = _real_transnet_v2(v.filepath)
            update_status(db, vid, progress_pct=8)  # scanning done, writing to DB

        if not shots:
            raise RuntimeError(
                f"shots: produced 0 shots for {duration:.1f}s video — "
                "TransNetV2 and OpenCV fallback both failed"
            )

        for idx, (start, end) in enumerate(shots):
            db.add(Shot(id=uuid4(), video_id=vid, start_seconds=start,
                        end_seconds=end, shot_index=idx))
        db.commit()
        update_status(db, vid, progress_pct=10)
        logger.info("[shots] produced %d shots for %.1fs video", len(shots), duration)
        return len(shots)


def _fake_shots(duration: float, shot_len: float = 5.0) -> list[tuple[float, float]]:
    out = []
    t = 0.0
    while t < duration:
        out.append((t, min(t + shot_len, duration)))
        t += shot_len
    return out


def _real_transnet_v2(filepath: str) -> list[tuple[float, float]]:
    """Run TransNetV2 (or the OpenCV fallback) over the full video."""
    from app.worker.ml import transnetv2_shots
    return transnetv2_shots(filepath)