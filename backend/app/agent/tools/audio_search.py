"""Hybrid search over speech-to-text transcript segments.

Three retrieval signals fused into a single ranked list:

1. **Substring (exact-phrase) match** — case-insensitive ILIKE. Fastest path,
   highest precision.  If the user types a quoted phrase or a distinctive
   word ("fire!", "Cosmos"), they almost certainly want the literal hit.

2. **BM25-equivalent (Postgres FTS)** — `text_tsv @@ websearch_to_tsquery`
   with `ts_rank()` scoring.  Stemming + stopword removal handle inflected
   forms ("running" matches "ran"), and the websearch parser tolerates the
   syntax users actually type ("two men flying" → AND of stemmed terms,
   `"exact phrase"` → phrase, `-word` → negation).

3. **Semantic (BGE-small)** — cosine similarity against `embedding`.
   Catches paraphrases the keyword path misses ("yelled help" → "shouted
   for assistance").  Only used to *fill* below the keyword cap, so a
   weak vector model can't drown out an exact match.

Why the layering: a pure-vector search over transcripts kept hiding obvious
literal matches behind near-paraphrases — the audit flagged the previous
implementation's substring boost as a vacuous OR (always true), so the
"prefer exact" intent was never enforced.  This rewrite makes the priority
explicit at the SQL level instead of relying on the embedding to be smart.
"""
from uuid import UUID

from sqlalchemy import func, literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.embed import embed_text_bge
from app.agent.schemas import ToolResult
from app.agent.tools._utils import empty_scope_result, scope_videos
from app.config import get_settings
from app.models import TranscriptSegment

settings = get_settings()

SCHEMA = {
    "name": "search_transcript",
    "description": (
        "Search the speech-to-text transcript for a quote, keyword, or topic. "
        "Hybrid retrieval: exact substring → keyword (stem-aware) → semantic. "
        "Use a quoted phrase for literal matches (\"fire\"), bare words for "
        "keyword search (gun shop), or a paraphrased description for semantic "
        "search (someone shouting for help). Each result has a `match_type` "
        "field (substring | bm25 | semantic) — substring is gold, semantic "
        "is paraphrase-evidence and should be verified by reading the text."
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
        return empty_scope_result("search_transcript", query)
    if not query:
        return ToolResult(
            model_summary="search_transcript: empty query",
            ui_payload={"results": [], "query": query},
            top_k_used=0,
        )

    top_k = min(int(params.get("top_k", 20)), settings.tool_result_top_k_default)

    # ── 1. Substring (highest priority) ────────────────────────────────────
    # ILIKE is fine here — transcript_segments tables are small (≤ tens of
    # thousands of rows per investigation).  Score is a flat 1.0 so these
    # always sort above BM25/semantic results in the merged list.
    substring_stmt = (
        select(
            TranscriptSegment.id,
            TranscriptSegment.video_id,
            TranscriptSegment.text,
            TranscriptSegment.start_seconds,
            TranscriptSegment.end_seconds,
            literal(1.0).label("score"),
            literal("substring").label("source"),
        )
        .where(TranscriptSegment.video_id.in_(video_ids))
        .where(TranscriptSegment.text.ilike(f"%{query}%"))
        .order_by(TranscriptSegment.video_id, TranscriptSegment.start_seconds)
        .limit(top_k)
    )
    substring_rows = (await session.execute(substring_stmt)).all()
    seen: set = {r.id for r in substring_rows}
    merged: list = list(substring_rows)

    # ── 2. BM25 (Postgres FTS with ts_rank) ────────────────────────────────
    # websearch_to_tsquery is the user-friendly parser: it handles AND/OR,
    # quoted phrases and negation without raising on malformed input.
    if len(merged) < top_k:
        remaining = top_k - len(merged)
        tsquery = func.websearch_to_tsquery("english", query)
        text_tsv = TranscriptSegment.text_tsv
        bm25_stmt = (
            select(
                TranscriptSegment.id,
                TranscriptSegment.video_id,
                TranscriptSegment.text,
                TranscriptSegment.start_seconds,
                TranscriptSegment.end_seconds,
                func.ts_rank(text_tsv, tsquery).label("score"),
                literal("bm25").label("source"),
            )
            .where(TranscriptSegment.video_id.in_(video_ids))
            .where(text_tsv.op("@@")(tsquery))
            .order_by(func.ts_rank(text_tsv, tsquery).desc())
            .limit(remaining * 2)  # over-fetch; we'll filter dups + clip below
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
                TranscriptSegment.id,
                TranscriptSegment.video_id,
                TranscriptSegment.text,
                TranscriptSegment.start_seconds,
                TranscriptSegment.end_seconds,
                # 1 - cosine_distance == cosine_similarity for normalized vectors.
                (1.0 - TranscriptSegment.embedding.cosine_distance(qvec)).label("score"),
                literal("semantic").label("source"),
            )
            .where(TranscriptSegment.video_id.in_(video_ids))
            .where(TranscriptSegment.embedding.is_not(None))
            .order_by(TranscriptSegment.embedding.cosine_distance(qvec))
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
        "segment_id": str(r.id),
        "video_id": str(r.video_id),
        "text": r.text,
        "start_seconds": float(r.start_seconds),
        "end_seconds": float(r.end_seconds),
        # Alias for FrameGrid: when no frame_id is present it falls back to
        # `<video #t=...>` thumbnails using video_id + timestamp_seconds.
        "timestamp_seconds": float(r.start_seconds),
        "score": float(r.score or 0.0),
        "match_type": r.source,
    } for r in merged]

    src_counts: dict[str, int] = {}
    for it in items:
        src_counts[it["match_type"]] = src_counts.get(it["match_type"], 0) + 1
    breakdown = ", ".join(f"{k}={v}" for k, v in src_counts.items()) or "none"

    # ── Surface actual text in model_summary ───────────────────────────────
    # The agent reads `model_summary` (the `ui_payload` is for the UI panel
    # only).  Give it the literal spoken words so it can quote, paraphrase,
    # or decide whether a hit is on-topic — counts alone made the agent say
    # things like "I can see hits but not the text".  Cap at 12 lines and
    # truncate each to keep prompt size bounded.
    MAX_SNIPPETS = 12
    SNIPPET_CHARS = 220
    head = f"search_transcript('{query[:60]}'): {len(items)} segments ({breakdown})"
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
                f"v={it['video_id'][:8]} t={it['start_seconds']:.1f}-{it['end_seconds']:.1f}s: "
                f"\"{txt}\""
            )
        more = f"\n  …{len(items) - MAX_SNIPPETS} more not shown" if len(items) > MAX_SNIPPETS else ""
        body = "\n" + "\n".join(lines) + more

    return ToolResult(
        model_summary=head + body,
        ui_payload={"results": items, "query": query},
        top_k_used=len(items),
    )
