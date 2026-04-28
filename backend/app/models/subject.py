from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Subject(Base):
    """A session-scoped subject identity. Reference embeddings live in
    `subject_references` (one row per embedding) — gives us standard
    pgvector typing and server-side ANN if needed. Matching is
    max-similarity over the set; capped at settings.subject_reference_max
    via farthest-point sampling in the registration / feedback tools.
    """

    __tablename__ = "subjects"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    investigation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("investigations.id", ondelete="CASCADE"),
        index=True,
    )
    label: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(32))  # 'person' | 'vehicle' | 'object'

    references: Mapped[list["SubjectReference"]] = relationship(
        back_populates="subject", cascade="all, delete-orphan",
    )


class SubjectReference(Base):
    """One row per confirmed reference embedding. Set is bounded by
    settings.subject_reference_max via farthest-point sampling."""

    __tablename__ = "subject_references"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    subject_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("subjects.id", ondelete="CASCADE"),
        index=True,
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(512))

    subject: Mapped["Subject"] = relationship(back_populates="references")


class SubjectInstance(Base):
    """Cross-video propagation. A subject accumulates (video_id, instance_id)
    pairs whose detections matched. There is NO global instance id."""

    __tablename__ = "subject_instances"

    subject_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("subjects.id", ondelete="CASCADE"),
        primary_key=True,
    )
    video_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="CASCADE"),
        primary_key=True,
    )
    instance_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_score: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32))  # 'user_confirmed' | 'embedding_match'
