"""Audio transcription (Parakeet) + segment embeddings.

Fake mode produces one short transcript segment per shot so the audio_search
tool path is exercisable without an ASR model.
"""
import logging
from uuid import UUID, uuid4

from sqlalchemy import delete, select

from app.config import get_settings
from app.models import Shot, TranscriptSegment, Video
from app.worker.celery_app import celery_app
from app.worker.db import session_scope
from app.worker.util import clamp_text, deterministic_unit_vector, update_status

logger = logging.getLogger(__name__)
settings = get_settings()


@celery_app.task(name="app.worker.tasks.transcribe.run", bind=True, acks_late=True)
def run(self, video_id: str) -> int:
    vid = UUID(video_id)
    with session_scope() as db:
        update_status(db, vid, stage="transcribe", status="processing")
        db.execute(delete(TranscriptSegment).where(TranscriptSegment.video_id == vid))
        v: Video | None = db.get(Video, vid)
        if v is None:
            return 0

        if settings.use_fake_ml:
            shots: list[Shot] = list(db.execute(
                select(Shot).where(Shot.video_id == vid).order_by(Shot.shot_index)
            ).scalars().all())
            for s in shots:
                text = f"[fake transcript shot {s.shot_index}]"
                seg = TranscriptSegment(
                    id=uuid4(), video_id=vid, text=text,
                    start_seconds=s.start_seconds, end_seconds=s.end_seconds,
                    # 384-dim to match the new BGE-sized column.
                    embedding=deterministic_unit_vector(text.encode() + s.id.bytes, dim=384),
                )
                db.add(seg)
            count = len(shots)
        else:
            count = _real_parakeet(vid, v.filepath)

        db.commit()
        update_status(db, vid, progress_pct=75)
        return count


def _real_parakeet(video_id: UUID, filepath: str) -> int:
    """Extract audio → 16 kHz mono WAV → Parakeet → segment rows with embeddings."""
    import tempfile

    from app.worker.db import session_scope
    from app.worker.ml import (
        bge_encode_text,
        extract_audio_16k_mono,
        parakeet_transcribe_segments,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    def _progress_cb(current_sec: float, total_sec: float) -> None:
        """Update DB progress: interpolates 66%→74% across audio duration."""
        if total_sec > 0:
            frac = min(current_sec / total_sec, 1.0)
            pct = int(66 + frac * 8)
            with session_scope() as db:
                update_status(db, video_id, progress_pct=pct)

    try:
        with session_scope() as db:
            update_status(db, video_id, progress_pct=66)  # audio extraction started
        try:
            extract_audio_16k_mono(filepath, wav_path)
        except Exception as e:
            raise RuntimeError(f"transcribe: audio extraction failed for {filepath}: {e}") from e

        import os
        wav_size = os.path.getsize(wav_path) if os.path.exists(wav_path) else 0
        if wav_size < 1024:
            logger.warning(
                "transcribe: extracted WAV is suspiciously small (%d bytes) — "
                "video may have no audio track; producing 0 segments", wav_size
            )

        with session_scope() as db:
            update_status(db, video_id, progress_pct=68)  # extraction done, transcribing
        segments = parakeet_transcribe_segments(wav_path, progress_callback=_progress_cb)
        if not segments:
            logger.warning("[transcribe] parakeet produced 0 segments (likely silent or speechless audio)")
            return 0

        texts = [clamp_text(s["text"], source="transcribe segment") for s in segments]
        # BGE-small is the right tool for text-text retrieval; SigLIP-text is
        # tuned for image-text alignment and was effectively useless here.
        embeddings = bge_encode_text(texts) if texts else []

        with session_scope() as db:
            for seg, text, emb in zip(segments, texts, embeddings, strict=True):
                row = TranscriptSegment(
                    id=uuid4(), video_id=video_id, text=text,
                    start_seconds=float(seg["start_seconds"]),
                    end_seconds=float(seg["end_seconds"]),
                    embedding=[float(x) for x in emb],
                )
                db.add(row)
        logger.info("[transcribe] produced %d segments", len(segments))
        return len(segments)
    finally:
        try:
            import os
            os.unlink(wav_path)
        except OSError:
            pass