"""Video API — list, fetch metadata, stream file.

The `/file` endpoint is a hot path called for every video thumbnail in the
results UI.  We deliberately do NOT use the FastAPI `Depends(get_session)`
dependency for it because that holds the DB connection open for the entire
lifetime of the FileResponse stream — which can be many seconds — and
quickly exhausts the pool when several thumbnails load in parallel.

Instead we open a short-lived session, fetch the filepath (cached after the
first hit), close the session, and only then return the FileResponse.
"""
from collections import OrderedDict
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import VideoOut
from app.database import SessionLocal, get_session
from app.models import Video

router = APIRouter(prefix="/api/videos", tags=["videos"])


# Bounded LRU cache: video filepaths never change after ingest, so we can
# avoid the DB round-trip entirely for repeat requests.
_FILEPATH_CACHE_MAX = 1024
_filepath_cache: "OrderedDict[UUID, str]" = OrderedDict()


def _cache_get(video_id: UUID) -> str | None:
    path = _filepath_cache.get(video_id)
    if path is not None:
        _filepath_cache.move_to_end(video_id)
    return path


def _cache_put(video_id: UUID, filepath: str) -> None:
    _filepath_cache[video_id] = filepath
    _filepath_cache.move_to_end(video_id)
    if len(_filepath_cache) > _FILEPATH_CACHE_MAX:
        _filepath_cache.popitem(last=False)


async def _fetch_video_filepath(video_id: UUID) -> str | None:
    """Look up a video's filepath, caching results.  Opens its own session
    so the caller can release the connection before streaming."""
    cached = _cache_get(video_id)
    if cached is not None:
        return cached
    async with SessionLocal() as session:
        v = await session.get(Video, video_id)
        if v is None:
            return None
        _cache_put(video_id, v.filepath)
        return v.filepath


@router.get("", response_model=list[VideoOut])
async def list_videos(
    collection_id: UUID | None = None, session: AsyncSession = Depends(get_session)
) -> list[VideoOut]:
    stmt = select(Video).order_by(Video.created_at.desc())
    if collection_id is not None:
        stmt = stmt.where(Video.collection_id == collection_id)
    rows = (await session.execute(stmt)).scalars().all()
    return [VideoOut.model_validate(v) for v in rows]


@router.get("/{video_id}", response_model=VideoOut)
async def get_video(video_id: UUID, session: AsyncSession = Depends(get_session)) -> VideoOut:
    v = await session.get(Video, video_id)
    if v is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "video not found")
    return VideoOut.model_validate(v)


@router.get("/{video_id}/file")
async def get_video_file(video_id: UUID):
    """Stream the source video file.

    Note: no DB-session dependency.  The session is opened, the filepath is
    fetched (or read from cache), and the session is closed BEFORE the
    FileResponse begins streaming.  This prevents the pool from being held
    for the entire stream duration.
    """
    filepath = await _fetch_video_filepath(video_id)
    if filepath is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "video not found")
    return FileResponse(filepath, media_type="video/mp4")
