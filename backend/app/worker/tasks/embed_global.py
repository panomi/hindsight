"""Frame-level SigLIP embeddings.

Fake mode uses deterministic seeded vectors per frame_id so identical inputs
always produce identical embeddings — which lets retrieval logic be tested.
"""
import logging
from uuid import UUID

from sqlalchemy import select

from app.config import get_settings
from app.models import Frame
from app.worker.celery_app import celery_app
from app.worker.db import session_scope
from app.worker.util import deterministic_unit_vector, update_status

logger = logging.getLogger(__name__)
settings = get_settings()


@celery_app.task(name="app.worker.tasks.embed_global.run", bind=True, acks_late=True)
def run(self, video_id: str) -> int:
    vid = UUID(video_id)
    with session_scope() as db:
        update_status(db, vid, stage="embed_global", status="processing")
        frames: list[Frame] = list(db.execute(
            select(Frame).where(Frame.video_id == vid)
        ).scalars().all())
        if not frames:
            return 0

        if settings.use_fake_ml:
            for f in frames:
                f.siglip_embedding = deterministic_unit_vector(f.id.bytes + b":frame")
        else:
            _real_siglip_global(frames)

        db.commit()
        update_status(db, vid, progress_pct=50)
        return len(frames)


def _real_siglip_global(frames):
    """SigLIP image embeddings on each frame, projected to 512-dim and stored."""
    from PIL import Image

    from app.worker.ml import siglip_encode_images

    images, idx_map, load_failures = [], [], 0
    for i, f in enumerate(frames):
        try:
            images.append(Image.open(f.filepath).convert("RGB"))
            idx_map.append(i)
        except Exception as e:
            load_failures += 1
            logger.warning("embed_global: failed to open frame %s (%s)", f.filepath, e)

    if not images:
        raise RuntimeError(
            f"embed_global: 0/{len(frames)} frames could load — visual search will be empty"
        )
    if load_failures:
        logger.warning("embed_global: %d/%d frames failed to load",
                       load_failures, len(frames))

    embeddings = siglip_encode_images(images, batch_size=32)
    for k, idx in enumerate(idx_map):
        frames[idx].siglip_embedding = [float(x) for x in embeddings[k]]
    logger.info("[embed_global] stored %d/%d frame embeddings",
                len(idx_map), len(frames))