"""captions: BGE-384 embeddings + tsvector GIN

Revision ID: 0005
Revises: 0004
Create Date: 2026-04-29 01:15:00

Mirrors migration 0004 (which did this for transcript_segments) — same
rationale, applied to captions:

1. **Embedding dim 512 → 384.**  Captions were embedded with SigLIP's
   text tower, which is trained for image-text alignment (max 64 tokens,
   weak text-text similarity).  New Qwen2.5-VL captions are ~50-word
   action paragraphs that get truncated and embed poorly under SigLIP.
   Switch to BAAI/bge-small-en-v1.5 (384-dim) and null out existing
   embeddings — the backfill script
   `backend/scripts/backfill_caption_embeddings.py` re-encodes them
   without re-running Qwen.

2. **Add `text_tsv` (tsvector) generated column + GIN index.**  Enables
   keyword retrieval via Postgres FTS — `text_tsv @@ websearch_to_tsquery
   ('english', :q)` with `ts_rank()` scoring.  Backs the new hybrid
   `search_captions` (substring → BM25 → semantic).
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE captions DROP COLUMN IF EXISTS embedding")
    op.execute("ALTER TABLE captions ADD COLUMN embedding vector(384)")

    op.execute(
        "ALTER TABLE captions "
        "ADD COLUMN text_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', text)) STORED"
    )
    op.execute(
        "CREATE INDEX ix_captions_text_tsv "
        "ON captions USING GIN (text_tsv)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_captions_text_tsv")
    op.execute("ALTER TABLE captions DROP COLUMN IF EXISTS text_tsv")
    op.execute("ALTER TABLE captions DROP COLUMN IF EXISTS embedding")
    op.execute("ALTER TABLE captions ADD COLUMN embedding vector(512)")
