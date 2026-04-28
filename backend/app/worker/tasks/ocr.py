"""OCR — gated by caption keywords (sign, screen, paper, plate, label).

Backed by RapidOCR (PP-OCRv4 weights via ONNX Runtime).  Only runs on
frames whose caption flagged a text carrier — full-frame OCR on every
frame would be ~50× more expensive with little extra recall.

Fake mode produces no OCR results (most footage doesn't have visible text);
the OCR-search tool path is still exercisable via the agent calling
`search_ocr` and getting an empty result, which is correct behaviour.
"""
import logging
from uuid import UUID

from sqlalchemy import select

from app.config import get_settings
from app.models import Caption, Frame, OcrText
from app.worker.celery_app import celery_app
from app.worker.db import session_scope
from app.worker.util import clamp_text, update_status

logger = logging.getLogger(__name__)
settings = get_settings()

OCR_TRIGGER_TERMS = ("sign", "screen", "paper", "plate", "label", "text", "newspaper", "menu")


@celery_app.task(name="app.worker.tasks.ocr.run", bind=True, acks_late=True)
def run(self, video_id: str) -> int:
    vid = UUID(video_id)
    with session_scope() as db:
        update_status(db, vid, stage="ocr", status="processing")

        # Find frames whose caption mentions any trigger term
        captions = list(db.execute(
            select(Caption).join(Frame, Frame.id == Caption.frame_id).where(Frame.video_id == vid)
        ).scalars().all())
        candidate_frame_ids = [
            c.frame_id for c in captions
            if any(term in (c.text or "").lower() for term in OCR_TRIGGER_TERMS)
        ]

        if not candidate_frame_ids:
            logger.info(
                "[ocr] gated 0/%d captions matched trigger terms %s — skipping OCR",
                len(captions), OCR_TRIGGER_TERMS,
            )
        else:
            logger.info(
                "[ocr] gated %d/%d captions matched trigger terms — running OCR",
                len(candidate_frame_ids), len(captions),
            )

        count = 0
        if not settings.use_fake_ml:  # pragma: no cover
            count = _real_ocr(candidate_frame_ids)
        # Fake: skip OCR; the gating logic itself is what matters here.

        db.commit()
        update_status(db, vid, progress_pct=98)
        return count


def _real_ocr(frame_ids: list[UUID]) -> int:
    """Run the OCR engine (RapidOCR) on each candidate frame.

    Embeds each OCR string with SigLIP text so search_ocr's embedding
    fallback path works.  TODO: swap this to BGE for better text-text
    retrieval, mirroring the transcript / caption migrations (chunk 4 +
    chunk 5).  Tracked in docs/code-audit-2026-04-28.md.
    """
    from uuid import uuid4

    from app.models import Frame as FrameModel
    from app.worker.ml import paddleocr_run, siglip_encode_text

    if not frame_ids:
        return 0

    written = 0
    ocr_errors = 0
    with session_scope() as db:
        frames = list(db.execute(
            select(FrameModel).where(FrameModel.id.in_(frame_ids))
        ).scalars().all())

        per_frame: list[tuple[UUID, list[dict]]] = []
        all_texts: list[str] = []
        text_owners: list[tuple[UUID, int]] = []  # (frame_id, idx_in_per_frame)
        for f in frames:
            try:
                hits = paddleocr_run(f.filepath)
            except Exception as e:
                ocr_errors += 1
                logger.warning("ocr: engine failed on %s (%s)", f.filepath, e)
                hits = []
            if not hits:
                continue
            per_frame.append((f.id, hits))
            for i, h in enumerate(hits):
                all_texts.append(h["text"])
                text_owners.append((f.id, len(per_frame) - 1))

        # Batch-embed all OCR strings via SigLIP text encoder.
        embeddings = siglip_encode_text(all_texts) if all_texts else []
        emb_iter = iter(embeddings)

        for frame_id, hits in per_frame:
            for h in hits:
                emb = next(emb_iter, None)
                db.add(OcrText(
                    id=uuid4(), frame_id=frame_id,
                    text=clamp_text(h["text"], source="ocr string"),
                    bbox=h["bbox"],
                    confidence=float(h["confidence"]),
                    embedding=[float(x) for x in emb] if emb is not None else None,
                ))
                written += 1

    if ocr_errors and written == 0:
        raise RuntimeError(
            f"ocr: engine failed on all {ocr_errors}/{len(frame_ids)} candidate frames "
            f"— wrote 0 strings (most recent error logged above)"
        )
    if ocr_errors:
        logger.warning("[ocr] %d/%d frames errored, wrote %d strings",
                       ocr_errors, len(frame_ids), written)
    else:
        logger.info("[ocr] %d frames scanned, wrote %d strings", len(frame_ids), written)
    return written