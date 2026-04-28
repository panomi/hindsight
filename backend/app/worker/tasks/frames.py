"""Frame extraction via ffmpeg.

Motion-adaptive sampling: within each shot we first look for scene-change
timestamps from ffmpeg's built-in scene filter (a single fast pass over the
whole video).  Only frames where visual content changes significantly are
selected; long static stretches get at most one frame.  The uniform midpoint
fallback is preserved for shots with no detected changes.

To revert to plain 1-fps / uniform sampling, set:
    default_max_inter_frame_seconds = 1.0  (in config / .env)
"""
import logging
import re
import shutil
import subprocess
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import delete, select

from app.config import get_settings
from app.models import Frame, Shot, Video
from app.worker.celery_app import celery_app
from app.worker.db import session_scope
from app.worker.util import update_status

logger = logging.getLogger(__name__)
settings = get_settings()

# Minimum visual change (0–1 scale) for ffmpeg scene filter to flag a frame.
# Lower → more frames; higher → only hard cuts.
_SCENE_THRESHOLD = 0.25


@celery_app.task(name="app.worker.tasks.frames.run", bind=True, acks_late=True)
def run(self, video_id: str) -> int:
    vid = UUID(video_id)
    with session_scope() as db:
        update_status(db, vid, stage="frames", status="processing")
        v: Video | None = db.get(Video, vid)
        if v is None:
            return 0

        db.execute(delete(Frame).where(Frame.video_id == vid))

        shots: list[Shot] = list(db.execute(
            select(Shot).where(Shot.video_id == vid).order_by(Shot.shot_index)
        ).scalars().all())

        if not shots:
            duration = v.duration_seconds or 0.0
            samples = _samples_uniform(duration)
        else:
            scene_changes = _detect_scene_changes(v.filepath)
            logger.info("frames: %d scene-change timestamps detected", len(scene_changes))
            samples = _samples_from_shots(shots, scene_changes)

        out_dir = settings.frames_dir / str(vid)
        out_dir.mkdir(parents=True, exist_ok=True)

        if shutil.which("ffmpeg") is None:
            raise RuntimeError("frames: ffmpeg not installed in PATH")

        fps = v.fps or 30.0
        extracted = 0
        failures: list[str] = []
        for idx, (ts, shot_id) in enumerate(samples):
            target = out_dir / f"{idx:06d}.jpg"
            try:
                _extract_frame(v.filepath, ts, target)
            except Exception as e:
                failures.append(f"#{idx}@{ts:.2f}s: {e!r}")
                logger.warning("frames: extraction failed at %.2fs: %s", ts, e)
                continue
            db.add(Frame(
                id=uuid4(), video_id=vid, shot_id=shot_id,
                timestamp_seconds=ts, frame_number=int(ts * fps),
                filepath=str(target.resolve()),
            ))
            extracted += 1

        db.commit()
        update_status(db, vid, progress_pct=20)

        if extracted == 0:
            sample_errors = "; ".join(failures[:3])
            raise RuntimeError(
                f"frames: extracted 0/{len(samples)} frames — first errors: {sample_errors}"
            )
        if failures:
            logger.warning("[frames] extracted %d/%d, %d failures",
                           extracted, len(samples), len(failures))
        else:
            logger.info("[frames] extracted %d/%d (from %d scene changes)",
                        extracted, len(samples), len(scene_changes) if shots else 0)
        return extracted


def _detect_scene_changes(filepath: str) -> list[float]:
    """Single ffmpeg pass to find timestamps where visual content changes.

    Uses the select+showinfo filter — fast because only metadata is read,
    no output frames are written.  Returns sorted list of seconds.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", filepath,
                "-vf", f"select='gt(scene,{_SCENE_THRESHOLD})',showinfo",
                "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=120,
        )
        timestamps: list[float] = []
        for line in result.stderr.splitlines():
            m = re.search(r"pts_time:([\d.]+)", line)
            if m:
                timestamps.append(float(m.group(1)))
        return sorted(set(timestamps))
    except Exception as e:
        logger.warning("frames: scene-change detection failed (%s) — using uniform sampling", e)
        return []


def _samples_from_shots(
    shots: list[Shot],
    scene_changes: list[float],
) -> list[tuple[float, UUID | None]]:
    """For each shot, pick the most visually distinct moments.

    Within a shot we prefer detected scene-change timestamps; if none fall
    inside, we fall back to the existing uniform-midpoint logic so behaviour
    is identical to before for static footage.
    """
    max_gap = settings.default_max_inter_frame_seconds
    min_frames = settings.default_min_frames_per_shot
    out: list[tuple[float, UUID | None]] = []

    for shot in shots:
        length = max(0.001, shot.end_seconds - shot.start_seconds)
        # How many frames would the uniform approach produce?
        n_uniform = max(min_frames, int(length // max_gap) + 1)

        # Scene changes that fall inside this shot (with a small margin)
        margin = 0.1
        in_shot = [
            t for t in scene_changes
            if shot.start_seconds + margin <= t <= shot.end_seconds - margin
        ]

        if in_shot:
            # Deduplicate changes that are very close together (< 0.5 s apart)
            deduped: list[float] = [in_shot[0]]
            for t in in_shot[1:]:
                if t - deduped[-1] >= 0.5:
                    deduped.append(t)
            # Cap at n_uniform so we never produce more frames than before
            selected = deduped[:n_uniform]
            # Always ensure at least one anchor at the shot midpoint if we'd
            # otherwise have only edge timestamps
            if len(selected) < min_frames:
                selected.append(shot.start_seconds + length / 2)
            for ts in sorted(selected):
                out.append((ts, shot.id))
        else:
            # No scene changes inside shot — use uniform midpoints (original logic)
            for k in range(n_uniform):
                ts = shot.start_seconds + (k + 0.5) * length / n_uniform
                out.append((ts, shot.id))

    return out


def _samples_uniform(duration: float) -> list[tuple[float, UUID | None]]:
    step = settings.default_max_inter_frame_seconds
    out: list[tuple[float, UUID | None]] = []
    t = step / 2
    while t < duration:
        out.append((t, None))
        t += step
    return out


def _extract_frame(src: str, ts: float, out: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", src,
            "-frames:v", "1", "-q:v", "3", str(out),
        ],
        check=True, capture_output=True, timeout=30,
    )
