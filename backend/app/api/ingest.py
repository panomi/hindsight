"""Video upload + ingestion trigger.

Wired to Celery in step 3; this module handles the HTTP-side concerns:
streaming the upload to disk, creating the Video row, and enqueueing the
ingest DAG. Celery import is lazy so the API process can start without
worker dependencies installed.
"""
from pathlib import Path
from uuid import UUID, uuid4

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (ScanRequest, ScanResponse, ScanRootsResponse,
                             VideoIngestResponse)
from app.config import get_settings
from app.database import get_session
from app.models import Collection, Video

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"}

router = APIRouter(prefix="/api/ingest", tags=["ingest"])
settings = get_settings()


CHUNK_SIZE = 1024 * 1024  # 1 MB


@router.post(
    "",
    response_model=VideoIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    description="Upload a video file and queue preprocessing.",
)
async def upload_video(
    collection_id: UUID = Form(...),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> VideoIngestResponse:
    coll = await session.get(Collection, collection_id)
    if coll is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "collection not found")

    # Save to disk under a uuid prefix to avoid name collisions
    settings.videos_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    video_id = uuid4()
    target = settings.videos_dir / f"{video_id}{suffix}"

    async with aiofiles.open(target, "wb") as out:
        while chunk := await file.read(CHUNK_SIZE):
            await out.write(chunk)

    v = Video(
        id=video_id,
        collection_id=collection_id,
        filename=file.filename or target.name,
        filepath=str(target.resolve()),
        status="pending",
        stage=None,
        progress_pct=0,
    )
    session.add(v)
    await session.commit()

    # Enqueue preprocessing DAG (lazy import — keeps API import light)
    try:
        from app.worker.tasks.ingest import run_ingest  # noqa: WPS433

        run_ingest.delay(str(video_id))
    except Exception as exc:  # pragma: no cover — broker may be unavailable in tests
        # We still return 202; status will remain "pending" until broker recovers.
        # The error is recorded so the UI can surface it.
        v.error = f"queue_failed: {exc!r}"
        await session.commit()

    return VideoIngestResponse(video_id=video_id, status=v.status)


@router.get("/scan-roots", response_model=ScanRootsResponse)
async def get_scan_roots() -> ScanRootsResponse:
    """The configured allowlist of absolute paths the server may scan."""
    return ScanRootsResponse(roots=settings.scan_roots)


@router.post(
    "/scan", response_model=ScanResponse, status_code=status.HTTP_202_ACCEPTED,
    description="Scan a server-side directory for videos and queue ingest "
                "for each. Path must be inside INGEST_SCAN_ROOTS. No upload — "
                "files are referenced in place."
)
async def scan_directory(
    payload: ScanRequest,
    session: AsyncSession = Depends(get_session),
) -> ScanResponse:
    coll = await session.get(Collection, payload.collection_id)
    if coll is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "collection not found")

    if not settings.scan_roots:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Server-side scan is disabled. Set INGEST_SCAN_ROOTS in the API .env "
            "to a comma-separated list of allowed root directories.",
        )

    target = Path(payload.server_path).expanduser()
    try:
        target_resolved = target.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"path not found: {exc}")

    # Path must be inside one of the allowed roots
    allowed = False
    for root in settings.scan_roots:
        try:
            root_resolved = Path(root).expanduser().resolve(strict=True)
        except OSError:
            continue
        try:
            target_resolved.relative_to(root_resolved)
            allowed = True
            break
        except ValueError:
            continue
    if not allowed:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"path is not inside any configured scan root: {settings.scan_roots}",
        )

    # Collect video files
    if target_resolved.is_file():
        files = [target_resolved] if target_resolved.suffix.lower() in VIDEO_EXTS else []
    elif target_resolved.is_dir():
        it = target_resolved.rglob("*") if payload.recursive else target_resolved.iterdir()
        files = sorted(p for p in it if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "path is neither a file nor a directory")

    queued: list[VideoIngestResponse] = []
    skipped: list[str] = []

    # Lazy-import the celery task so the API doesn't crash if broker is down.
    try:
        from app.worker.tasks.ingest import run_ingest
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"queue unavailable: {exc!r}")

    for f in files:
        try:
            video_id = uuid4()
            v = Video(
                id=video_id,
                collection_id=payload.collection_id,
                filename=f.name,
                filepath=str(f),
                status="pending",
                stage=None,
                progress_pct=0,
            )
            session.add(v)
            await session.commit()
            run_ingest.delay(str(video_id))
            queued.append(VideoIngestResponse(video_id=video_id, status="pending"))
        except Exception as exc:
            skipped.append(f"{f.name}: {exc!r}")

    return ScanResponse(queued=queued, skipped=skipped)


@router.post("/recover", status_code=status.HTTP_200_OK)
async def recover_stuck(session: AsyncSession = Depends(get_session)) -> dict:
    """Re-queue any video that should be ingesting but isn't: stuck-mid-flight
    'processing' rows AND 'pending' rows with no inflight Celery task.

    Call this if a worker died mid-ingest, the queue was purged, or a row was
    manually reset and never re-queued."""
    from sqlalchemy import or_

    from app.worker.tasks.ingest import run_ingest

    result = await session.execute(
        select(Video).where(or_(Video.status == "processing",
                                Video.status == "pending"))
    )
    orphans = result.scalars().all()
    for v in orphans:
        if v.status == "processing":
            await session.execute(
                sa_update(Video)
                .where(Video.id == v.id)
                .values(status="pending", stage=None, progress_pct=0, error=None)
            )
    await session.commit()
    for v in orphans:
        run_ingest.apply_async(args=[str(v.id)], queue="cpu")
    return {"recovered": len(orphans), "video_ids": [str(v.id) for v in orphans]}