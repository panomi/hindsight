from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import collections as collections_api
from app.api import frames as frames_api
from app.api import ingest as ingest_api
from app.api import investigations as investigations_api
from app.api import stream as stream_api
from app.api import videos as videos_api
from app.config import get_settings

settings = get_settings()


async def _recover_stuck_videos() -> int:
    """On startup, re-queue any video that should be ingesting but isn't:

    - 'processing' videos: a worker died mid-stage. Reset to pending and re-queue.
    - 'pending' videos: either a worker died before picking up the task, or the
      queue was purged, or someone reset the row manually.  At API-startup time
      no fresh upload can be inflight (the upload endpoint enqueues synchronously
      before returning) so any pending row is safe to re-queue.

    Idempotent — re-queueing a video that's already in the queue just produces a
    no-op duplicate; the global ingest lock + ingest_active marker handles that
    inside run_ingest itself.
    """
    import logging
    from sqlalchemy import or_, select, update
    from app.database import SessionLocal
    from app.models import Video
    from app.worker.tasks.ingest import run_ingest

    logger = logging.getLogger(__name__)
    try:
        async with SessionLocal() as session:
            result = await session.execute(
                select(Video).where(or_(Video.status == "processing",
                                        Video.status == "pending"))
            )
            orphans = result.scalars().all()
            if not orphans:
                return 0
            for v in orphans:
                if v.status == "processing":
                    await session.execute(
                        update(Video)
                        .where(Video.id == v.id)
                        .values(status="pending", stage=None, progress_pct=0, error=None)
                    )
                    logger.warning("watchdog: reset stuck video %s (%s) → pending",
                                   v.filename, v.id)
                else:
                    logger.warning("watchdog: pending video with no inflight task — %s (%s)",
                                   v.filename, v.id)
            await session.commit()
        for v in orphans:
            run_ingest.apply_async(args=[str(v.id)], queue="cpu")
            logger.warning("watchdog: re-queued %s", v.filename)
        return len(orphans)
    except Exception as exc:
        logging.getLogger(__name__).error("watchdog error: %s", exc)
        return 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.videos_dir.mkdir(parents=True, exist_ok=True)
    settings.frames_dir.mkdir(parents=True, exist_ok=True)
    settings.thumbnails_dir.mkdir(parents=True, exist_ok=True)
    recovered = await _recover_stuck_videos()
    if recovered:
        import logging
        logging.getLogger(__name__).warning("watchdog: recovered %d stuck video(s)", recovered)
    yield


app = FastAPI(
    title="Investigation Platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(collections_api.router)
app.include_router(frames_api.router)
app.include_router(videos_api.router)
app.include_router(ingest_api.router)
app.include_router(investigations_api.router)
app.include_router(stream_api.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
