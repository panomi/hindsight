from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


# ── Collections ────────────────────────────────────────────────────────────────

class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None


class CollectionOut(BaseModel):
    id: UUID
    name: str
    description: str | None
    created_at: datetime
    video_count: int = 0

    class Config:
        from_attributes = True


# ── Videos ─────────────────────────────────────────────────────────────────────

class VideoOut(BaseModel):
    id: UUID
    collection_id: UUID
    filename: str
    duration_seconds: float | None
    fps: float | None
    resolution: str | None
    status: str
    stage: str | None
    progress_pct: int
    error: str | None
    created_at: datetime

    class Config:
        from_attributes = True


class VideoIngestResponse(BaseModel):
    video_id: UUID
    status: str


class ScanRequest(BaseModel):
    collection_id: UUID
    server_path: str = Field(min_length=1, description="Absolute path on the server")
    recursive: bool = True


class ScanResponse(BaseModel):
    queued: list[VideoIngestResponse]
    skipped: list[str] = Field(default_factory=list)
    error: str | None = None


class ScanRootsResponse(BaseModel):
    roots: list[str]


# ── Investigations ─────────────────────────────────────────────────────────────

class InvestigationCreate(BaseModel):
    collection_id: UUID
    title: str = Field(min_length=1, max_length=512)


class InvestigationOut(BaseModel):
    id: UUID
    collection_id: UUID
    title: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class InvestigationMessageIn(BaseModel):
    content: str = Field(min_length=1)


# ── Confirmations ──────────────────────────────────────────────────────────────

class ConfirmationIn(BaseModel):
    """User's response to a `request_user_confirmation` prompt.

    The IDs are opaque tokens the agent itself put into the items list — they
    may be UUIDs (detection_id / frame_id) but can also be track keys or
    composite strings.  The agent matches them back by string equality, so we
    accept any string here rather than constraining to UUID.

    `skipped=True` means the user moved on (typed a new chat message before
    answering) — the agent unblocks with 0 confirmed / 0 rejected and the UI
    clears the popup.
    """
    confirmation_id: UUID
    confirmed_ids: list[str] = []
    rejected_ids: list[str] = []
    skipped: bool = False