"""Hybrid search over scene captions (vision-language paragraphs at ingest).

Mirrors `search_transcript`: substring → BM25 (Postgres FTS) → semantic
(BGE-small) — same priority order, same `match_type` field on every
result so the agent can gauge confidence.

Why three signals.  Captions are 50-word action paragraphs ("Two figures
exit a vehicle and walk toward a brick alley"); a single signal is
brittle:

* Substring catches user-typed literals ("alley") or rare nouns the
  embedding might dilute.
* BM25 (`websearch_to_tsquery` + `ts_rank`) catches inflected forms
  ("walking" matches "walked") and supports the natural query syntax
  users actually type — quoted phrases, AND/OR/negation.
* Semantic (BGE) catches paraphrases ("getting out of car" → "exit
  vehicle") that the keyword paths miss entirely.

Each path only fills slots the previous path didn't, so high-precision
hits always sort above paraphrase guesses.  Substring hits get score
1.0; BM25 returns `ts_rank()` (typically 0–1, often small); semantic
returns cosine similarity in [0, 1].  Scales aren't comparable, so the
agent should rely on `match_type` first and use raw score as a
within-bucket tiebreak.
"""
from uuid import UUID

from sqlalchemy import func, literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.embed import embed_text_bge
from app.agent.schemas import ToolResult
from app.agent.tools._utils import empty_scope_result, scope_videos
from app.config import get_settings
from app.models import Caption, Frame

settings = get_settings()

SCHEMA = {
    "name": "search_captions",
    "description": (
        "Search scene descriptions indexed at ingest time.  Captions cover "
        "ACTIONS, INTERACTIONS, and SETTING (verbs, who-is-doing-what, scene "
        "type, distinctive open-vocab attributes detectors miss). Use for "
        "activity / interaction / scene-type queries.  Hybrid retrieval: "
        "exact substring → keyword (stem-aware) → semantic.  Each result has "
        "a `match_type` field (substring | bm25 | semantic) — substring is "
        "gold, semantic is paraphrase-evidence."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "video_ids": {"type": "array", "items": {"type": "string"}},
            "top_k": {"type": "integer", "default": 20, "maximum": 50},
        },
        "required": ["query"],
    },
}


async def run(session: AsyncSession, params: dict, investigation_id: UUID) -> ToolResult:
    query = (params["query"] or "").strip()
    video_ids = await scope_videos(session, investigation_id, params.get("video_ids", []))
    if not video_ids:
        return empty_scope_result("search_captions", query)
    if not query:
        return ToolResult(
            model_summary="search_captions: empty query",
            ui_payload={"results": [], "query": query},
            top_k_used=0,
        )

    top_k = min(int(params.get("top_k", 20)), settings.tool_result_top_k_default)

    # Common projected columns.  Substring/BM25/semantic produce the same
    # row shape so the merge step below stays simple.
    base_cols = (
        Caption.id,
        Caption.frame_id,
        Caption.shot_id,
        Caption.text,
        Frame.video_id,
        Frame.timestamp_seconds,
        Frame.filepath,
    )

    # ── 1. Substring (highest priority) ────────────────────────────────────
    substring_stmt = (
        select(
            *base_cols,
            literal(1.0).label("score"),
            literal("substring").label("source"),
        )
        .join(Frame, Frame.id == Caption.frame_id)
        .where(Frame.video_id.in_(video_ids))
        .where(Caption.text.ilike(f"%{query}%"))
        .order_by(Frame.video_id, Frame.timestamp_seconds)
        .limit(top_k)
    )
    substring_rows = (await session.execute(substring_stmt)).all()
    seen: set = {r.id for r in substring_rows}
    merged: list = list(substring_rows)

    # ── 2. BM25 (Postgres FTS with ts_rank) ────────────────────────────────
    if len(merged) < top_k:
        remaining = top_k - len(merged)
        tsquery = func.websearch_to_tsquery("english", query)
        text_tsv = Caption.text_tsv
        bm25_stmt = (
            select(
                *base_cols,
                func.ts_rank(text_tsv, tsquery).label("score"),
                literal("bm25").label("source"),
            )
            .join(Frame, Frame.id == Caption.frame_id)
            .where(Frame.video_id.in_(video_ids))
            .where(text_tsv.op("@@")(tsquery))
            .order_by(func.ts_rank(text_tsv, tsquery).desc())
            .limit(remaining * 2)
        )
        bm25_rows = (await session.execute(bm25_stmt)).all()
        for r in bm25_rows:
            if r.id in seen:
                continue
            merged.append(r)
            seen.add(r.id)
            if len(merged) >= top_k:
                break

    # ── 3. Semantic (BGE) — only fills remaining slots ─────────────────────
    if len(merged) < top_k:
        remaining = top_k - len(merged)
        qvec = embed_text_bge(query)
        sem_stmt = (
            select(
                *base_cols,
                (1.0 - Caption.embedding.cosine_distance(qvec)).label("score"),
                literal("semantic").label("source"),
            )
            .join(Frame, Frame.id == Caption.frame_id)
            .where(Frame.video_id.in_(video_ids))
            .where(Caption.embedding.is_not(None))
            .order_by(Caption.embedding.cosine_distance(qvec))
            .limit(remaining * 2)
        )
        sem_rows = (await session.execute(sem_stmt)).all()
        for r in sem_rows:
            if r.id in seen:
                continue
            merged.append(r)
            seen.add(r.id)
            if len(merged) >= top_k:
                break

    items = [{
        "caption_id": str(r.id),
        "frame_id": str(r.frame_id),
        "shot_id": str(r.shot_id) if r.shot_id else None,
        "video_id": str(r.video_id),
        "text": r.text,
        "timestamp_seconds": r.timestamp_seconds,
        "filepath": r.filepath,
        "score": float(r.score or 0.0),
        "match_type": r.source,
    } for r in merged]

    src_counts: dict[str, int] = {}
    for it in items:
        src_counts[it["match_type"]] = src_counts.get(it["match_type"], 0) + 1
    breakdown = ", ".join(f"{k}={v}" for k, v in src_counts.items()) or "none"

    # frame_ids_with_scores line lets the agent feed straight into
    # temporal_cluster (matching the contract used by visual_search).
    scored_csv = ", ".join(f"{i['frame_id']}:{i['score']:.3f}" for i in items[:20])

    # ── Surface caption text in model_summary ──────────────────────────────
    # Same rationale as search_transcript: the agent reads model_summary,
    # not ui_payload, so it needs the actual paragraph to know whether a
    # hit really shows the action being asked about.
    MAX_SNIPPETS = 12
    SNIPPET_CHARS = 220
    head = f"search_captions('{query[:60]}'): {len(items)} hits ({breakdown})"
    if not items:
        body = ""
    else:
        lines = []
        for it in items[:MAX_SNIPPETS]:
            txt = (it["text"] or "").strip().replace("\n", " ")
            if len(txt) > SNIPPET_CHARS:
                txt = txt[: SNIPPET_CHARS - 1] + "…"
            lines.append(
                f"  [{it['match_type']} {it['score']:.2f}] "
                f"v={it['video_id'][:8]} t={it['timestamp_seconds']:.1f}s "
                f"frame={it['frame_id'][:8]}: \"{txt}\""
            )
        more = f"\n  …{len(items) - MAX_SNIPPETS} more not shown" if len(items) > MAX_SNIPPETS else ""
        body = "\n" + "\n".join(lines) + more
    suffix = f"\nframe_ids_with_scores: [{scored_csv}]" if items else ""

    return ToolResult(
        model_summary=head + body + suffix,
        ui_payload={"results": items, "query": query},
        top_k_used=len(items),
    )
