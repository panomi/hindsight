"""Re-embed every existing TranscriptSegment with BGE-small-en-v1.5.

Run after applying migration 0004 (which nulls transcript_segments.embedding
because the old SigLIP-text vectors were the wrong tool for text-text
retrieval and would be meaningless against new BGE queries).

This script does NOT re-run ASR — it just encodes the existing segment
text with the new model.  Safe to re-run; it only touches rows where
embedding IS NULL.

Usage:
    cd backend
    PYTHONPATH=. uv run python scripts/backfill_transcript_embeddings.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(batch_size: int = 256) -> None:
    from sqlalchemy import select, update

    from app.models import TranscriptSegment
    from app.worker.db import session_scope
    from app.worker.ml import bge_encode_text

    with session_scope() as db:
        total = db.execute(
            select(TranscriptSegment.id).where(TranscriptSegment.embedding.is_(None))
        ).scalars().all()
        total_ct = len(total)

    if total_ct == 0:
        print("Nothing to backfill — every segment already has an embedding.")
        return

    print(f"Backfilling {total_ct} transcript segment(s) with BGE-small-en-v1.5 …")

    done = 0
    t0 = time.monotonic()
    while True:
        with session_scope() as db:
            rows = db.execute(
                select(TranscriptSegment.id, TranscriptSegment.text)
                .where(TranscriptSegment.embedding.is_(None))
                .limit(batch_size)
            ).all()
            if not rows:
                break

            texts = [r.text or "" for r in rows]
            embeddings = bge_encode_text(texts)
            for r, emb in zip(rows, embeddings):
                db.execute(
                    update(TranscriptSegment)
                    .where(TranscriptSegment.id == r.id)
                    .values(embedding=[float(x) for x in emb])
                )
            db.commit()

        done += len(rows)
        rate = done / max(time.monotonic() - t0, 1e-6)
        print(f"  • {done}/{total_ct} ({rate:.1f}/s)")

    print(f"Done in {time.monotonic() - t0:.1f}s.")


if __name__ == "__main__":
    main()
