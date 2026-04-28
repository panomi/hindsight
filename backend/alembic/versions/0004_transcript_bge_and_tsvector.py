"""transcript_segments: BGE-384 embeddings + tsvector GIN

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-29 00:00:00

Two changes, both targeting transcript search quality:

1. **Embedding dim 512 → 384.**  We swap SigLIP (image-text alignment, 512-dim,
   poor at text-text retrieval) for BAAI/bge-small-en-v1.5 (384-dim, top-tier
   sentence encoder).  Existing embeddings are nulled — they were computed
   with the wrong model and would be meaningless against new BGE queries.
   A backfill script (`backend/scripts/backfill_transcript_embeddings.py`)
   re-encodes existing transcripts without re-running ASR.

2. **Add `text_tsv` (tsvector) generated column + GIN index.**  Enables
   millisecond keyword retrieval via Postgres FTS — `text_tsv @@
   websearch_to_tsquery('english', :q)` with `ts_rank()` scoring.  This is
   the BM25-equivalent path used by the new hybrid search_transcript
   (substring → BM25 → semantic).
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop old embedding column and recreate at 384 dims.  We don't try to
    # ALTER it in place — pgvector ALTER would attempt a cast/copy of values
    # that are about to be discarded anyway.
    op.execute("ALTER TABLE transcript_segments DROP COLUMN IF EXISTS embedding")
    op.execute("ALTER TABLE transcript_segments ADD COLUMN embedding vector(384)")

    # Generated tsvector + GIN index for keyword/BM25-style retrieval.
    op.execute(
        "ALTER TABLE transcript_segments "
        "ADD COLUMN text_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', text)) STORED"
    )
    op.execute(
        "CREATE INDEX ix_transcript_segments_text_tsv "
        "ON transcript_segments USING GIN (text_tsv)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_transcript_segments_text_tsv")
    op.execute("ALTER TABLE transcript_segments DROP COLUMN IF EXISTS text_tsv")
    op.execute("ALTER TABLE transcript_segments DROP COLUMN IF EXISTS embedding")
    op.execute("ALTER TABLE transcript_segments ADD COLUMN embedding vector(512)")
