from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Shot(Base):
    __tablename__ = "shots"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    video_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), index=True
    )
    start_seconds: Mapped[float] = mapped_column(Float)
    end_seconds: Mapped[float] = mapped_column(Float)
    shot_index: Mapped[int] = mapped_column(Integer)


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    frame_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("frames.id", ondelete="CASCADE"), index=True
    )
    video_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), index=True
    )
    class_name: Mapped[str] = mapped_column(String(128))
    confidence: Mapped[float] = mapped_column(Float)
    bbox: Mapped[dict] = mapped_column(JSONB)  # {x1,y1,x2,y2} normalized
    instance_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    box_embedding: Mapped[list[float] | None] = mapped_column(Vector(512), nullable=True)

    __table_args__ = (
        Index("ix_detections_video_class", "video_id", "class_name"),
        Index("ix_detections_video_instance", "video_id", "instance_id"),
    )


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    video_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), index=True
    )
    text: Mapped[str] = mapped_column(Text)
    start_seconds: Mapped[float] = mapped_column(Float)
    end_seconds: Mapped[float] = mapped_column(Float)
    # BGE-small-en-v1.5 = 384 dims (was SigLIP 512, ill-suited for text-text retrieval).
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)
    # Generated tsvector for BM25-style keyword search via Postgres FTS.
    # `english` config = stemming + stopword removal + lowercasing.
    # GIN index defined in __table_args__ — together they give millisecond
    # `text @@ websearch_to_tsquery(...)` queries with ts_rank scoring.
    text_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', text)", persisted=True),
        nullable=True,
    )

    __table_args__ = (
        Index("ix_transcript_segments_text_tsv", "text_tsv", postgresql_using="gin"),
    )


class Caption(Base):
    __tablename__ = "captions"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    shot_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("shots.id", ondelete="CASCADE"), nullable=True, index=True
    )
    frame_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("frames.id", ondelete="CASCADE"), index=True
    )
    text: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(64))
    # BGE-small-en-v1.5 = 384 dims (was SigLIP 512, ill-suited for text-text
    # retrieval over 50-word action paragraphs).  Mirrors transcripts (chunk 4).
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)
    # Generated tsvector + GIN index for keyword/BM25-style retrieval
    # (websearch_to_tsquery + ts_rank).  Same pattern as transcript_segments.
    text_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', text)", persisted=True),
        nullable=True,
    )

    __table_args__ = (
        Index("ix_captions_text_tsv", "text_tsv", postgresql_using="gin"),
    )


class OcrText(Base):
    __tablename__ = "ocr_texts"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    frame_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("frames.id", ondelete="CASCADE"), index=True
    )
    text: Mapped[str] = mapped_column(Text)
    bbox: Mapped[dict] = mapped_column(JSONB)
    confidence: Mapped[float] = mapped_column(Float)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(512), nullable=True)
