"""Re-embed every existing Caption with BGE-small-en-v1.5.

Run after applying migration 0005 (which nulls captions.embedding because
the old SigLIP-text vectors were the wrong tool for text-text retrieval
over 50-word action paragraphs and would be meaningless against new BGE
queries).

Does NOT re-run Qwen — just encodes the existing caption text with the
new model.  Safe to re-run; only touches rows where embedding IS NULL.

Usage:
    cd backend
    PYTHONPATH=. uv run python scripts/backfill_caption_embeddings.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(batch_size: int = 256) -> None:
    from sqlalchemy import select, update

    from app.models import Caption
    from app.worker.db import session_scope
    from app.worker.ml import bge_encode_text

    with session_scope() as db:
        total = db.execute(
            select(Caption.id).where(Caption.embedding.is_(None))
        ).scalars().all()
        total_ct = len(total)

    if total_ct == 0:
        print("Nothing to backfill — every caption already has an embedding.")
        return

    print(f"Backfilling {total_ct} caption(s) with BGE-small-en-v1.5 …")

    done = 0
    t0 = time.monotonic()
    while True:
        with session_scope() as db:
            rows = db.execute(
                select(Caption.id, Caption.text)
                .where(Caption.embedding.is_(None))
                .limit(batch_size)
            ).all()
            if not rows:
                break

            texts = [r.text or "" for r in rows]
            embeddings = bge_encode_text(texts)
            for r, emb in zip(rows, embeddings, strict=True):
                db.execute(
                    update(Caption)
                    .where(Caption.id == r.id)
                    .values(embedding=[float(x) for x in emb])
                )
            db.commit()

        done += len(rows)
        rate = done / max(time.monotonic() - t0, 1e-6)
        print(f"  • {done}/{total_ct} ({rate:.1f}/s)")

    print(f"Done in {time.monotonic() - t0:.1f}s.")


if __name__ == "__main__":
    main()
