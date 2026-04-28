"""subject reference set as a separate table

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-25 00:00:01

The plan calls for a SET of confirmed embeddings per Subject (max-similarity
matching, capped at N=20 via farthest-point sampling). Initially we tried
ARRAY(Vector(512)) but pgvector's SQLAlchemy adapter doesn't cleanly handle
arrays of vectors. A dedicated subject_references table is cleaner and
allows server-side ANN against the set if needed later.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE subjects DROP COLUMN IF EXISTS reference_embeddings")

    op.create_table(
        "subject_references",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("subject_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("subjects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("embedding", Vector(512), nullable=False),
    )
    op.create_index("ix_subject_references_subject_id", "subject_references", ["subject_id"])


def downgrade() -> None:
    op.drop_table("subject_references")
    op.execute("ALTER TABLE subjects ADD COLUMN reference_embeddings vector(512)[]")
