"""Frame image serving endpoint.

Same hot-path treatment as `videos.py /file`: open a short-lived session,
fetch the cached filepath, close it, then stream the JPEG.  Holding the DB
connection for the entire FileResponse stream exhausted the pool when many
thumbnails loaded in parallel.
"""
from collections import OrderedDict
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from app.database import SessionLocal
from app.models import Frame

router = APIRouter(prefix="/api/frames", tags=["frames"])


# Frame filepaths never change after ingest — safe to cache aggressively.
_FILEPATH_CACHE_MAX = 8192
_filepath_cache: "OrderedDict[UUID, str]" = OrderedDict()


def _cache_get(frame_id: UUID) -> str | None:
    path = _filepath_cache.get(frame_id)
    if path is not None:
        _filepath_cache.move_to_end(frame_id)
    return path


def _cache_put(frame_id: UUID, filepath: str) -> None:
    _filepath_cache[frame_id] = filepath
    _filepath_cache.move_to_end(frame_id)
    if len(_filepath_cache) > _FILEPATH_CACHE_MAX:
        _filepath_cache.popitem(last=False)


async def _fetch_frame_filepath(frame_id: UUID) -> str | None:
    cached = _cache_get(frame_id)
    if cached is not None:
        return cached
    async with SessionLocal() as session:
        f = await session.get(Frame, frame_id)
        if f is None:
            return None
        _cache_put(frame_id, f.filepath)
        return f.filepath


@router.get("/{frame_id}/image")
async def get_frame_image(frame_id: UUID):
    """Serve the JPEG for a single extracted frame."""
    filepath = await _fetch_frame_filepath(frame_id)
    if filepath is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "frame not found")
    return FileResponse(filepath, media_type="image/jpeg")
