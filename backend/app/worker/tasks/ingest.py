"""Ingest DAG orchestrator.

Probes the video, then runs the 8 preprocessing stages sequentially.
Stages are individual Celery tasks routed to stage-specific queues so
they can run on different workers in production, but they're chained
through this single orchestrator task for clarity and idempotency.

Memory strategy
---------------
After each stage we flush PyTorch's CUDA caching allocator so VRAM
fragmentation doesn't slowly grow over many videos.  All models are kept
resident on GPU between stages — the full model set fits in ~11 GiB on a
24 GiB card, so there is no pressure to evict between stages.

The main OOM risk is stale worker processes from a previous ``./start.sh``
run holding GPU memory.  ``start.sh`` kills those before launching new
workers, which is the correct fix.

Timing
------
Wall-clock seconds per stage are collected and logged at the end together
with the video's duration.  Use the ``x_realtime`` column to estimate
throughput: e.g. ``video_duration / stage_wall_s`` gives you how many
seconds of footage are processed per real second for that stage.
"""
import logging
import time
from uuid import UUID

from app.worker.celery_app import celery_app
from app.worker.db import session_scope
from app.worker.tasks import caption as caption_task
from app.worker.tasks import detect_track as detect_task
from app.worker.tasks import embed_box as embed_box_task
from app.worker.tasks import embed_global as embed_global_task
from app.worker.tasks import frames as frames_task
from app.worker.tasks import ocr as ocr_task
from app.worker.tasks import shots as shots_task
from app.worker.tasks import transcribe as transcribe_task
from app.worker.util import ffprobe, update_status

logger = logging.getLogger(__name__)

# Order matches the plan's preprocessing pipeline.
STAGE_TASKS = [
    ("shots",        shots_task.run),
    ("frames",       frames_task.run),
    ("detect_track", detect_task.run),
    ("embed_global", embed_global_task.run),
    ("embed_box",    embed_box_task.run),
    ("transcribe",   transcribe_task.run),
    ("caption",      caption_task.run),
    ("ocr",          ocr_task.run),
]


def _run_stage(task, video_id: str, stage_name: str, max_retries: int = 4) -> None:
    """Call a stage task directly, retrying on PostgreSQL deadlock up to max_retries times.

    Two parallel ingest workers can deadlock on shared index pages when both
    execute DELETE FROM <table> WHERE video_id=? at the same moment.  PostgreSQL
    resolves it by aborting one transaction; we just need to retry that side.
    """
    from sqlalchemy.exc import OperationalError

    for attempt in range(max_retries):
        try:
            task.run(video_id)
            return
        except OperationalError as exc:
            if "DeadlockDetected" in str(exc) and attempt < max_retries - 1:
                wait_s = 0.5 * (2 ** attempt)  # 0.5 s, 1 s, 2 s, …
                logger.warning(
                    "pg deadlock in stage %s (attempt %d/%d) — retry in %.1fs",
                    stage_name, attempt + 1, max_retries, wait_s,
                )
                time.sleep(wait_s)
                continue
            raise  # non-deadlock OperationalError or retries exhausted


def _gpu_cleanup(stage: str) -> None:
    """Flush CUDA caching allocator and log a one-line VRAM snapshot.

    All models stay resident on GPU between stages — this is just a
    diagnostic + a hint to the allocator to defragment if it can.
    """
    try:
        import gc
        import torch
        if not torch.cuda.is_available():
            return
        gc.collect()
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        free = total - reserved
        logger.warning(
            "[gpu] after %-14s  alloc=%.2f GiB  reserved=%.2f GiB  free=%.2f GiB / %.2f GiB total",
            stage, alloc, reserved, free, total,
        )
    except Exception:
        pass



@celery_app.task(name="app.worker.tasks.ingest.run_ingest", bind=True, acks_late=True)
def run_ingest(self, video_id: str, force: bool = False) -> dict:
    """Probe + run all 8 stages. Each stage updates progress in DB.

    Args:
        video_id: UUID of the video to ingest.
        force: If True, re-ingest even if the video is already ``ready``
               (e.g. after a model upgrade).  If False (default), a task
               that arrives for an already-ready video is treated as an
               accidental duplicate and skipped silently.
    """
    vid = UUID(video_id)

    # ── Global serial ingest lock ──────────────────────────────────────────────
    # With a single GPU, running two ingests simultaneously causes VRAM
    # contention, OOMs, and DB deadlocks.  We use one global Redis lock so only
    # one video ingests at a time.  The second task blocks here (holding its
    # worker slot) until the first finishes, then proceeds automatically.
    # Per-video key is kept as a suffix so logs are readable.
    from redis import Redis
    from app.config import get_settings as _gs
    _redis = Redis.from_url(_gs().redis_url)
    _lock_key = "ingest_lock:global"
    _lock = _redis.lock(_lock_key, timeout=7200)  # 2h safety TTL

    # Quick duplicate check — bail if THIS video is already being processed.
    _dup_key = f"ingest_active:{video_id}"
    if _redis.get(_dup_key):
        logger.warning("run_ingest %s: duplicate task — another worker is already processing this video", video_id)
        return {"ok": False, "error": "duplicate: this video is already being ingested"}

    # Show the user we're in the queue, waiting for the GPU slot.
    if not _lock.acquire(blocking=False):
        with session_scope() as db:
            update_status(db, vid, status="processing", stage="queued", progress_pct=0)
        logger.info("run_ingest %s: global lock busy — waiting for current ingest to finish", video_id)
        acquired = _lock.acquire(blocking=True, blocking_timeout=7200)
        if not acquired:
            with session_scope() as db:
                update_status(db, vid, status="error", error="ingest lock timeout after 2h")
            return {"ok": False, "error": "ingest lock timeout"}

    # Everything below holds the global lock — must release in finally even on early return.
    try:
        # Skip if a previous task already completed this video while we were queued.
        # Bypass with force=True for intentional re-ingestion (e.g. model upgrades).
        if not force:
            with session_scope() as _db:
                from app.models import Video as _Video
                _v = _db.get(_Video, vid)
                if _v is not None and _v.status == "ready":
                    logger.info("run_ingest %s: already ready — skipping duplicate task (pass force=True to re-ingest)", video_id)
                    return {"ok": True, "skipped": "already ready"}

        # Mark this video as actively ingesting (TTL matches lock TTL).
        _redis.set(_dup_key, "1", ex=7200)
        try:
            return _run_ingest_inner(self, video_id, vid)
        finally:
            _redis.delete(_dup_key)
    finally:
        try:
            _lock.release()
        except Exception:
            pass


def _run_ingest_inner(self, video_id: str, vid: UUID) -> dict:

    # ── Probe ──────────────────────────────────────────────────────────────────
    duration_seconds: float | None = None
    with session_scope() as db:
        from app.models import Video
        v = db.get(Video, vid)
        if v is None:
            return {"ok": False, "error": "video not found"}
        meta = ffprobe(v.filepath)
        if meta:
            v.duration_seconds = meta.get("duration_seconds")
            v.fps = meta.get("fps")
            w, h = meta.get("width"), meta.get("height")
            if w and h:
                v.resolution = f"{w}x{h}"
            db.commit()
            duration_seconds = v.duration_seconds
        update_status(db, vid, status="processing", progress_pct=2, error=None)

    # ── Stages ─────────────────────────────────────────────────────────────────
    timings: dict[str, float] = {}
    ingest_start = time.perf_counter()

    for name, task in STAGE_TASKS:
        t0 = time.perf_counter()
        try:
            _run_stage(task, video_id, name)
        except Exception as e:
            with session_scope() as db:
                update_status(db, vid, status="error", error=f"{name}: {e!r}")
            return {"ok": False, "stage": name, "error": repr(e)}
        finally:
            elapsed = time.perf_counter() - t0
            timings[name] = round(elapsed, 2)
            _gpu_cleanup(stage=name)

    total_wall = round(time.perf_counter() - ingest_start, 2)

    # ── Timing report ──────────────────────────────────────────────────────────
    _log_timing_summary(video_id, duration_seconds, timings, total_wall)

    # ── Output sanity report — surfaces any stage that produced 0 rows ─────────
    counts = _stage_output_counts(vid)
    _log_output_summary(video_id, counts)

    with session_scope() as db:
        update_status(db, vid, status="ready", progress_pct=100, stage="done", error=None)
    return {"ok": True, "timings": timings, "total_wall_s": total_wall, "counts": counts}


def _stage_output_counts(vid: UUID) -> dict[str, int]:
    """Read row counts produced by each stage so silent zeros are obvious."""
    from sqlalchemy import func, select as _select

    from app.models import (Caption, Detection, Frame, OcrText, Shot,
                            TranscriptSegment)

    counts: dict[str, int] = {}
    with session_scope() as db:
        counts["shots"] = db.execute(
            _select(func.count(Shot.id)).where(Shot.video_id == vid)
        ).scalar_one() or 0
        counts["frames"] = db.execute(
            _select(func.count(Frame.id)).where(Frame.video_id == vid)
        ).scalar_one() or 0
        counts["detections"] = db.execute(
            _select(func.count(Detection.id)).where(Detection.video_id == vid)
        ).scalar_one() or 0
        counts["frames_with_emb"] = db.execute(
            _select(func.count(Frame.id)).where(
                Frame.video_id == vid, Frame.siglip_embedding.isnot(None)
            )
        ).scalar_one() or 0
        counts["dets_with_emb"] = db.execute(
            _select(func.count(Detection.id)).where(
                Detection.video_id == vid, Detection.box_embedding.isnot(None)
            )
        ).scalar_one() or 0
        counts["transcript_segments"] = db.execute(
            _select(func.count(TranscriptSegment.id)).where(TranscriptSegment.video_id == vid)
        ).scalar_one() or 0
        counts["captions"] = db.execute(
            _select(func.count(Caption.id)).join(Frame, Frame.id == Caption.frame_id)
            .where(Frame.video_id == vid)
        ).scalar_one() or 0
        counts["ocr"] = db.execute(
            _select(func.count(OcrText.id)).join(Frame, Frame.id == OcrText.frame_id)
            .where(Frame.video_id == vid)
        ).scalar_one() or 0
    return counts


# Stages that should ALWAYS produce >0 rows for any non-silent video.  OCR and
# transcript can legitimately be 0 (no text in scene; silent video).
_REQUIRED_NONZERO = ("shots", "frames", "detections",
                     "frames_with_emb", "dets_with_emb", "captions")


def _log_output_summary(video_id: str, counts: dict[str, int]) -> None:
    """Print per-stage DB row counts; flag any required stage with 0 rows."""
    lines = [f"[ingest counts] video={video_id}"]
    flagged: list[str] = []
    for k, v in counts.items():
        marker = ""
        if k in _REQUIRED_NONZERO and v == 0:
            marker = "  ← UNEXPECTED ZERO"
            flagged.append(k)
        lines.append(f"  {k:<22}: {v}{marker}")
    if flagged:
        lines.append(f"  ⚠ Stages flagged: {', '.join(flagged)} — silent failure likely")
    logger.warning("\n".join(lines))


def _log_timing_summary(
    video_id: str,
    duration_s: float | None,
    timings: dict[str, float],
    total_wall: float,
) -> None:
    lines = [
        f"[ingest timing] video={video_id}",
        f"  video_duration : {duration_s:.1f}s" if duration_s else "  video_duration : unknown",
        f"  total_wall     : {total_wall:.1f}s",
    ]
    for stage, wall in timings.items():
        if duration_s and wall > 0:
            x_rt = duration_s / wall  # e.g. 3.2 → processes 3.2s of footage per real second
            lines.append(f"  {stage:<14}: {wall:7.1f}s  ({x_rt:.2f}× realtime)")
        else:
            lines.append(f"  {stage:<14}: {wall:7.1f}s")
    if duration_s and total_wall > 0:
        lines.append(f"  TOTAL          : {total_wall:7.1f}s  ({duration_s / total_wall:.2f}× realtime end-to-end)")
        mins_per_hour = (total_wall / duration_s) * 3600 / 60
        lines.append(f"  → ~{mins_per_hour:.0f} min of compute per hour of footage")
    logger.warning("\n".join(lines))
