"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-25 00:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "collections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.String(2000), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "videos",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("filepath", sa.String(1024), nullable=False),
        sa.Column("duration_seconds", sa.Float()),
        sa.Column("fps", sa.Float()),
        sa.Column("resolution", sa.String(32)),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("stage", sa.String(32)),
        sa.Column("progress_pct", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.String(2000)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_videos_collection_id", "videos", ["collection_id"])
    op.create_index("ix_videos_status", "videos", ["status"])

    op.create_table(
        "shots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
        sa.Column("start_seconds", sa.Float(), nullable=False),
        sa.Column("end_seconds", sa.Float(), nullable=False),
        sa.Column("shot_index", sa.Integer(), nullable=False),
    )
    op.create_index("ix_shots_video_id", "shots", ["video_id"])

    op.create_table(
        "frames",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
        sa.Column("shot_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("shots.id", ondelete="SET NULL")),
        sa.Column("timestamp_seconds", sa.Float(), nullable=False),
        sa.Column("frame_number", sa.Integer(), nullable=False),
        sa.Column("filepath", sa.String(1024), nullable=False),
        sa.Column("siglip_embedding", Vector(512)),
    )
    op.create_index("ix_frames_video_id", "frames", ["video_id"])
    op.execute(
        "CREATE INDEX ix_frames_siglip_hnsw ON frames "
        "USING hnsw (siglip_embedding vector_cosine_ops)"
    )

    op.create_table(
        "detections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("frame_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("frames.id", ondelete="CASCADE"), nullable=False),
        sa.Column("video_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
        sa.Column("class_name", sa.String(128), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("bbox", postgresql.JSONB, nullable=False),
        sa.Column("instance_id", sa.Integer()),
        sa.Column("box_embedding", Vector(512)),
    )
    op.create_index("ix_detections_frame_id", "detections", ["frame_id"])
    op.create_index("ix_detections_video_id", "detections", ["video_id"])
    op.create_index("ix_detections_video_class", "detections", ["video_id", "class_name"])
    op.create_index("ix_detections_video_instance", "detections", ["video_id", "instance_id"])
    op.execute(
        "CREATE INDEX ix_detections_box_hnsw ON detections "
        "USING hnsw (box_embedding vector_cosine_ops)"
    )

    op.create_table(
        "transcript_segments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
        sa.Column("text", sa.String(4000), nullable=False),
        sa.Column("start_seconds", sa.Float(), nullable=False),
        sa.Column("end_seconds", sa.Float(), nullable=False),
        sa.Column("embedding", Vector(512)),
    )
    op.create_index("ix_transcript_video_id", "transcript_segments", ["video_id"])
    op.execute(
        "CREATE INDEX ix_transcript_embedding_hnsw ON transcript_segments "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "captions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("shot_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("shots.id", ondelete="CASCADE")),
        sa.Column("frame_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("frames.id", ondelete="CASCADE"), nullable=False),
        sa.Column("text", sa.String(4000), nullable=False),
        sa.Column("source", sa.String(64), nullable=False),
        sa.Column("embedding", Vector(512)),
    )
    op.create_index("ix_captions_shot_id", "captions", ["shot_id"])
    op.create_index("ix_captions_frame_id", "captions", ["frame_id"])
    op.execute(
        "CREATE INDEX ix_captions_embedding_hnsw ON captions "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "ocr_texts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("frame_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("frames.id", ondelete="CASCADE"), nullable=False),
        sa.Column("text", sa.String(2000), nullable=False),
        sa.Column("bbox", postgresql.JSONB, nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("embedding", Vector(512)),
    )
    op.create_index("ix_ocr_frame_id", "ocr_texts", ["frame_id"])

    op.create_table(
        "investigations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_investigations_collection_id", "investigations", ["collection_id"])

    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("investigation_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", postgresql.JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_messages_investigation_id", "messages", ["investigation_id"])

    # Subjects: reference_embeddings is an ARRAY of vector(512), capped at N=20 in app code.
    op.execute(
        """
        CREATE TABLE subjects (
          id UUID PRIMARY KEY,
          investigation_id UUID NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
          label VARCHAR(128) NOT NULL,
          kind VARCHAR(32) NOT NULL,
          reference_embeddings vector(512)[]
        )
        """
    )
    op.create_index("ix_subjects_investigation_id", "subjects", ["investigation_id"])

    op.create_table(
        "subject_instances",
        sa.Column("subject_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("subjects.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("videos.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("instance_id", sa.Integer(), primary_key=True),
        sa.Column("match_score", sa.Float(), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
    )

    op.create_table(
        "prompt_cache",
        sa.Column("prompt_hash", sa.String(64), primary_key=True),
        sa.Column("frame_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("frames.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("tool", sa.String(64), primary_key=True),
        sa.Column("payload", postgresql.JSONB, nullable=False),
    )

    op.create_table(
        "agent_actions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("investigation_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("tool", sa.String(64), nullable=False),
        sa.Column("params_json", postgresql.JSONB, nullable=False),
        sa.Column("result_summary", sa.String(4000)),
        sa.Column("ui_payload_hash", sa.String(64)),
        sa.Column("result_count", sa.Integer()),
        sa.Column("duration_ms", sa.Integer()),
        sa.Column("confirmation_id", postgresql.UUID(as_uuid=True)),
        sa.Column("parent_action_id", postgresql.UUID(as_uuid=True)),
    )
    op.create_index("ix_agent_actions_investigation_ts", "agent_actions",
                    ["investigation_id", "ts"])

    op.create_table(
        "results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("investigation_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("frame_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("frames.id", ondelete="CASCADE"), nullable=False),
        sa.Column("detection_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("detections.id", ondelete="SET NULL")),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("confirmed", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("rejected", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("rank", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("subject_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("subjects.id", ondelete="SET NULL")),
    )
    op.create_index("ix_results_investigation_id", "results", ["investigation_id"])


def downgrade() -> None:
    for t in [
        "results", "agent_actions", "prompt_cache", "subject_instances", "subjects",
        "messages", "investigations", "ocr_texts", "captions", "transcript_segments",
        "detections", "frames", "shots", "videos", "collections",
    ]:
        op.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
    op.execute("DROP EXTENSION IF EXISTS vector")
