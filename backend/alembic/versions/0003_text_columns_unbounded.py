"""widen text columns to TEXT

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-28 00:00:00

Free-text outputs from ML models (Parakeet transcripts, Florence-2 captions,
PaddleOCR strings) routinely exceed the original VARCHAR(N) caps.  TEXT has
no length limit and zero performance cost on PostgreSQL — these columns
should never have had a bound.

The ALTER COLUMN TYPE TEXT operation does not rewrite the table since
VARCHAR(N) and TEXT share an on-disk representation, so the migration is
near-instant even on populated databases.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE transcript_segments ALTER COLUMN text TYPE TEXT")
    op.execute("ALTER TABLE captions             ALTER COLUMN text TYPE TEXT")
    op.execute("ALTER TABLE ocr_texts            ALTER COLUMN text TYPE TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE transcript_segments ALTER COLUMN text TYPE VARCHAR(4000)")
    op.execute("ALTER TABLE captions             ALTER COLUMN text TYPE VARCHAR(4000)")
    op.execute("ALTER TABLE ocr_texts            ALTER COLUMN text TYPE VARCHAR(2000)")
