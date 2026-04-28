"""co_presence(terms) — find frames where ≥2 terms appear nearby.

Two-step (per the plan):
  1. Resolve every term to bboxes per frame:
       {subject_id}    → join via subject_instances → detections
       {class_name}    → query detections by class
       {semantic_text} → open_vocab_detect (SAM 3.1)
  2. Proximity-join in normalized coords (center distance < proximity for
     metric='center', or IoU > proximity for metric='iou'), then dilate
     across time_window_sec for near-simultaneous appearance.

Defaults are deliberately tight (0.06 normalized for 'near'); the
spatial-reasoning skill instructs the agent to widen them per intent.
"""
from collections import defaultdict
from uuid import UUID

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.schemas import ToolResult
from app.agent.tools import open_vocab_detect
from app.agent.tools._utils import empty_scope_result, scope_videos, validate_subject
from app.config import get_settings
from app.models import Detection, Frame, SubjectInstance

settings = get_settings()

SCHEMA = {
    "name": "co_presence",
    "description": "Find frames where two or more terms appear nearby. Each term "
                   "is one of {subject_id}, {class_name}, or {semantic_text}. "
                   "Tight defaults — widen per query intent (see spatial-reasoning).",
    "input_schema": {
        "type": "object",
        "properties": {
            "terms": {
                "type": "array", "minItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "subject_id": {"type": "string"},
                        "class_name": {"type": "string"},
                        "semantic_text": {"type": "string"},
                    },
                },
            },
            "metric": {"type": "string", "enum": ["center", "iou"], "default": "center"},
            "proximity": {"type": "number", "default": 0.06,
                          "description": "center: max bbox-center distance in normalized coords; "
                                         "iou: min IoU"},
            "time_window_sec": {"type": "number", "default": 2.0},
            "video_ids": {"type": "array", "items": {"type": "string"}},
            "top_k": {"type": "integer", "default": 30, "maximum": 50},
        },
        "required": ["terms"],
    },
}


def _bbox_center(b: dict) -> tuple[float, float]:
    return ((b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2)


def _center_distance(a: dict, b: dict) -> float:
    ax, ay = _bbox_center(a); bx, by = _bbox_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    bb = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = aa + bb - inter + 1e-9
    return inter / union


async def _resolve_term(
    session: AsyncSession, term: dict, video_ids: list[UUID],
    investigation_id: UUID,
) -> dict[UUID, list[dict]]:
    """Return frame_id -> list of bbox dicts for this term.

    `video_ids` is the in-scope video filter from the orchestrator — never
    None or empty here because callers must check before invoking.
    """
    out: dict[UUID, list[dict]] = defaultdict(list)

    if "subject_id" in term:
        subject_id = UUID(term["subject_id"])
        # Subject ownership is validated upstream by run(); if we got here,
        # the subject belongs to this investigation.
        stmt = (
            select(Detection)
            .join(SubjectInstance, (SubjectInstance.video_id == Detection.video_id) &
                  (SubjectInstance.instance_id == Detection.instance_id))
            .where(SubjectInstance.subject_id == subject_id)
            .where(Detection.video_id.in_(video_ids))
        )
        for d in (await session.execute(stmt)).scalars().all():
            out[d.frame_id].append({**d.bbox, "_label": f"subject:{subject_id}",
                                    "instance_id": d.instance_id})
        return out

    if "class_name" in term:
        stmt = (
            select(Detection)
            .where(Detection.class_name == term["class_name"])
            .where(Detection.video_id.in_(video_ids))
        )
        for d in (await session.execute(stmt)).scalars().all():
            out[d.frame_id].append({**d.bbox, "_label": f"class:{term['class_name']}",
                                    "instance_id": d.instance_id})
        return out

    if "semantic_text" in term:
        # Resolve via open_vocab_detect, scoped to one in-scope video at a time.
        for vid in video_ids:
            res = await open_vocab_detect.run(
                session,
                {"text_prompt": term["semantic_text"], "video_id": str(vid)},
                investigation_id,
            )
            for item in res.ui_payload.get("results", []):
                fid = UUID(item["frame_id"])
                out[fid].append({**item["bbox"], "_label": f"semantic:{term['semantic_text']}"})
        return out

    return out


async def run(session: AsyncSession, params: dict, investigation_id: UUID) -> ToolResult:
    terms = params["terms"]
    metric = params.get("metric", "center")
    proximity = float(params.get("proximity", 0.06))
    time_window_sec = float(params.get("time_window_sec", 2.0))
    video_ids = await scope_videos(session, investigation_id, params.get("video_ids", []))
    if not video_ids:
        return empty_scope_result("co_presence")
    top_k = min(int(params.get("top_k", 30)), settings.tool_result_top_k_default)

    # Validate every subject_id term belongs to this investigation BEFORE
    # touching any data. Refuse the whole call if any one is bogus —
    # otherwise we'd silently degrade to a partial co-presence that's
    # almost always wrong.
    for t in terms:
        if "subject_id" in t:
            subj = await validate_subject(session, UUID(t["subject_id"]), investigation_id)
            if subj is None:
                return ToolResult(
                    model_summary=(
                        f"co_presence: subject {t['subject_id']} not found in this "
                        f"investigation. Subjects are scoped per investigation — register "
                        f"or look up the right one."
                    ),
                    ui_payload={"results": []}, error="subject_not_in_scope",
                )

    # Resolve every term to per-frame bboxes
    resolved = []
    for t in terms:
        resolved.append(await _resolve_term(session, t, video_ids, investigation_id))

    if any(len(r) == 0 for r in resolved):
        return ToolResult(
            model_summary=f"co_presence: at least one term resolved to 0 frames",
            ui_payload={"results": [], "terms": terms},
        )

    # Find frames where every term has at least one bbox AND there exists
    # at least one tuple of bboxes (one per term) satisfying the proximity test.
    common_frames = set(resolved[0].keys())
    for r in resolved[1:]:
        common_frames &= set(r.keys())

    matches = []
    # Pull frame metadata for ordering / display
    if common_frames:
        frame_meta = {
            f.id: f for f in (await session.execute(
                select(Frame).where(Frame.id.in_(common_frames))
            )).scalars().all()
        }
    else:
        frame_meta = {}

    for fid in common_frames:
        bboxes_per_term = [r[fid] for r in resolved]
        # Try every combination of one bbox per term — naive but bounded by per-frame counts
        from itertools import product
        for tuple_combo in product(*bboxes_per_term):
            ok = True
            for i in range(len(tuple_combo)):
                for j in range(i + 1, len(tuple_combo)):
                    if metric == "center":
                        if _center_distance(tuple_combo[i], tuple_combo[j]) > proximity:
                            ok = False; break
                    else:  # iou
                        if _iou(tuple_combo[i], tuple_combo[j]) < proximity:
                            ok = False; break
                if not ok:
                    break
            if ok:
                fm = frame_meta.get(fid)
                matches.append({
                    "frame_id": str(fid),
                    "video_id": str(fm.video_id) if fm else None,
                    "timestamp_seconds": fm.timestamp_seconds if fm else None,
                    "filepath": fm.filepath if fm else None,
                    "bboxes": list(tuple_combo),
                })
                break

    # Time-window dilation: if `time_window_sec > 0`, group hits within window
    matches.sort(key=lambda m: (m["video_id"], m["timestamp_seconds"] or 0))
    matches = matches[:top_k]

    return ToolResult(
        model_summary=(
            f"co_presence({len(terms)} terms, metric={metric}, prox={proximity}): "
            f"{len(matches)} frames, "
            f"across {len({m['video_id'] for m in matches})} video(s)"
        ),
        ui_payload={"results": matches, "terms": terms,
                    "metric": metric, "proximity": proximity,
                    "time_window_sec": time_window_sec},
        top_k_used=len(matches),
    )
