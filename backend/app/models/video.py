from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    videos: Mapped[list["Video"]] = relationship(back_populates="collection", cascade="all, delete-orphan")


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    collection_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("collections.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(512))
    filepath: Mapped[str] = mapped_column(String(1024))
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    resolution: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    progress_pct: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    collection: Mapped[Collection] = relationship(back_populates="videos")


class Frame(Base):
    __tablename__ = "frames"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    video_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), index=True
    )
    shot_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("shots.id", ondelete="SET NULL"), nullable=True
    )
    timestamp_seconds: Mapped[float] = mapped_column(Float)
    frame_number: Mapped[int] = mapped_column(Integer)
    filepath: Mapped[str] = mapped_column(String(1024))
    siglip_embedding: Mapped[list[float] | None] = mapped_column(Vector(512), nullable=True)
