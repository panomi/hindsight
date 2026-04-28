from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Investigation(Base):
    __tablename__ = "investigations"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    collection_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("collections.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    investigation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # user | assistant | tool
    content: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PromptCache(Base):
    """Cache for SAM 3.1 / open_vocab_detect by (prompt_hash, frame_id, tool)."""

    __tablename__ = "prompt_cache"

    prompt_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    frame_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("frames.id", ondelete="CASCADE"),
        primary_key=True,
    )
    tool: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSONB)


class AgentAction(Base):
    """Structured, queryable audit trail for evidentiary use.
    Complements Message (which stores conversation transcript)."""

    __tablename__ = "agent_actions"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    investigation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="CASCADE")
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    turn_index: Mapped[int] = mapped_column(Integer)
    tool: Mapped[str] = mapped_column(String(64))
    params_json: Mapped[dict] = mapped_column(JSONB)
    result_summary: Mapped[str | None] = mapped_column(String(4000), nullable=True)
    ui_payload_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    result_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confirmation_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    parent_action_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)

    __table_args__ = (Index("ix_agent_actions_investigation_ts", "investigation_id", "ts"),)


class Result(Base):
    __tablename__ = "results"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    investigation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="CASCADE"), index=True
    )
    frame_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("frames.id", ondelete="CASCADE")
    )
    detection_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("detections.id", ondelete="SET NULL"),
        nullable=True,
    )
    score: Mapped[float] = mapped_column(Float)
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    rejected: Mapped[bool] = mapped_column(Boolean, default=False)
    rank: Mapped[int] = mapped_column(Integer, default=0)
    subject_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("subjects.id", ondelete="SET NULL"),
        nullable=True,
    )
