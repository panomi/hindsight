# Code Audit — Apr 28, 2026

Comprehensive review of the ingest pipeline and agent system.
Findings prioritised by impact × likelihood of biting in production.

---

## P0 — Fix before next user demo

### Cross-collection data leakage on empty/missing investigation scope

**Where:** `backend/app/agent/tools/_utils.py:36` + 7 call sites (`visual_search.py:45`, `caption_search.py:48`, `audio_search.py:43`, `ocr_search.py:43,56`, `detection_query.py:44`, `image_search.py`, `instance_search.py`)

When `scope_videos()` returns `None` (investigation doesn't exist OR has zero ready videos in its collection), every search tool currently does `if video_ids: stmt.where(...)` — so `None` becomes "no filter", returning frames/captions/transcripts from ALL collections globally.

**Repro:** Create a brand new investigation, ask a question before any video is `ready`. Search results will include frames from other investigations.

**Fix:**
```python
if video_ids is None:
    return ToolResult(model_summary="...0 results", ui_payload={"results": []}, top_k_used=0)
stmt = stmt.where(Frame.video_id.in_(video_ids))
```

### `open_vocab_detect`, `search_by_image`, `search_instances_by_subject`, `co_presence` ignore `investigation_id`

Same family — they only filter when `video_id`/`video_ids` is explicitly passed. Default behaviour is global DB search.

### Subject ownership not validated

**Where:** `scene_assembly.py:35`, `co_presence.py`

`Subject` rows have an `investigation_id` field but tools don't check it. Anyone with a valid subject UUID can resolve timelines/co-presence for someone else's investigation. Mitigated by UUID unguessability but still wrong.

---

## P1 — Real bugs / silent failures

### Tool history not persisted across turns

**Where:** `orchestrator.py:32-38`, `147-151`

On reload, only plain user/assistant text is restored. The agent loses all prior tool IDs, scores, and intermediate reasoning. Follow-up questions like "show me the second-best one" can't work.

### ByteTrack desync on frame load failure

**Where:** `detect_track.py:122-124`

If a frame fails to load, the loop `continue`s without calling `bytetrack_step()`. The tracker uses frame index internally — skipping a step misaligns track IDs for the rest of the video. Rare but corrupts data when it happens.

### `temporal_cluster` silently disables relative filter

**Where:** `temporal_cluster.py:53-57`

If the agent passes `frame_ids` but a `scores` list of different length, every score becomes 1.0 → relative threshold becomes a no-op → all results pass through. Should fail loudly (raise ValueError or return error result).

### `investigation-strategy.md` references wrong tool name and broken chain

**Where:** `skills/investigation-strategy.md:13`

- Says `search_audio_transcripts` but registered name is `search_transcript`
- Says "parse `frame_ids_with_scores` from `search_transcript` and pass to `temporal_cluster`" — but transcript segments have no `frame_id` column at all, only timestamps. The documented pipeline cannot work.

### Score-plumbing inconsistency across search tools

Visual + caption_search have relative pruning + `frame_ids_with_scores` line.
Transcript, OCR, image_search, open_vocab_detect, detection_query do **not**.
The agent can't apply a uniform strategy.

### Concurrent turns on same investigation

**Where:** `investigations.py:100`, `sse_bus.py:11`

Two messages in flight for the same investigation interleave on a single SSE queue and race on DB writes. Easy to hit by double-clicking send.

### Caption real-mode: SigLIP per-row encode

**Where:** `caption.py:88-90`

After Qwen batching, we're still calling `siglip_encode_text([text])[0]` once per shot in a Python loop. Should batch all 436+ captions in one call. Likely saves 30-60s on a long video.

---

## P2 — Should-fix when convenient

- **`co_presence.time_window_sec` unused** — feature documented in schema but not implemented (only sorts/truncates)
- **`instance_search` Pass 2 has `LIMIT` without `ORDER BY`** — non-deterministic which detections expand
- **`caption_search`'s "substring boost"** — `or_(text.ilike(%q%), id.is_not(None))` is a vacuous filter; ordering is purely embedding distance
- **OCR stage doesn't DELETE on re-run** — if invoked alone, would create duplicate rows (no unique constraint)
- **PaddleOCR opens image twice per frame** (`ml.py:673,677`) — wasted I/O
- **Stage idempotency varies** — `embed_global`/`embed_box` don't clear before re-run, so partial re-runs can leave stale embeddings mixed with new
- **TransNet timeout doesn't actually stop the thread** (`ml.py:295-298`) — daemon thread keeps using the GPU until it finishes naturally
- **`update_status` always commits** — splits transactions in fragile ways. Brittle pattern.
- **Tool descriptions leak model names** to the LLM (SigLIP, Parakeet, RT-DETR, PaddleOCR) — these can echo in user-facing answers, contradicting `communication-style.md`
- **`open_vocab_detect` doc still says SAM 3.1 but uses Florence-2** (`open_vocab_detect.py:46-47` vs `129-136`)
- **Transcribe pattern is fragile** — outer DELETE in one transaction, inner INSERTs in separate transactions. Works correctly in PostgreSQL because DELETE evaluates `WHERE` at execution time, but the pattern is confusing. Refactor to keep DELETE + INSERT in same transaction.

---

## Done since audit

- **P0 scoping** — `_utils.scope_videos()` now returns `[]` instead of `None`; all 13 search/grounding tools use `empty_scope_result()` to abort cleanly.
- **P0 subject ownership** — `_utils.validate_subject()` added; called from `co_presence`, `instance_search`, `scene_assembly`, `apply_user_feedback`, `transcript_for_subject`.
- **P1 tool history** — `orchestrator._persist_blocks()` writes raw Anthropic tool_use / tool_result blocks; `_fetch_history()` rebuilds full context on reload.
- **P1 transcript embedding model + substring OR bug** — Replaced SigLIP-text (512-dim, image-text) with BGE-small-en-v1.5 (384-dim, text-text). `search_transcript` rewritten as hybrid substring → BM25 (ts_rank) → semantic (BGE). Migration `0004` adds tsvector + GIN, nulls old embeddings; `scripts/backfill_transcript_embeddings.py` re-encodes existing rows without re-running ASR.
- **P1 caption embedding model + substring OR bug + tool description leakage** — Same treatment for captions: migration `0005` (`Caption.embedding` 512→384, `text_tsv` GIN), `caption.py` + `caption_frames.py` re-wired to BGE, `search_captions` rewritten as hybrid (substring → BM25 → semantic) exposing `match_type`. Backfill script `scripts/backfill_caption_embeddings.py`. Tool SCHEMAs across `audio_search`, `caption_search`, `ocr_search`, `detection_query`, `open_vocab_detect`, `caption_frames` no longer leak model names to the LLM (Parakeet, PaddleOCR, RT-DETR, Florence-2, SigLIP, SAM 3.1, qwen2.5-vl). `investigation-strategy.md` rewritten — fixed wrong tool name (`search_audio_transcripts` → `search_transcript`), fixed broken audio→cluster pipeline, added composer tools to per-query patterns.
- **DB pool exhaustion** — pool 5/10 → 10/20, `pool_timeout=15s`; `/videos/{id}/file` and `/frames/{id}/image` release the AsyncSession before returning `FileResponse`, with an LRU filepath cache to avoid repeat DB hits.

## Open follow-ups

Tracked in `code-audit-2026-04-29.md` (auth-tight REST middleware, OCR-to-BGE,
OVD replacement, `temporal_cluster` fail-loud, etc).

---

## P3 — Code health, no behaviour change

- Several tools have nearly-identical pgvector boilerplate (cosine_distance + order + limit + format CSV) — extract a helper
- Dead code: `embed_box._key_hash`, `ocr.text_owners`
- `confirmation.PENDING` global never cleared on timeout (memory leak in long-running workers)
- Several places catching bare `Exception: pass` swallow real errors

---

## Recommended order

1. **P0 scoping bugs** — small, well-defined patch (~30 min). Highest priority, real data leakage.
2. **Strategy + score-plumbing alignment** — fix `investigation-strategy.md` to match reality, add `frame_ids_with_scores` + relative pruning to transcript/OCR/image/OVD, make `temporal_cluster` fail loudly on length mismatch. ~1-2 hours.
3. **Tool history persistence** — meaningful UX improvement for multi-turn conversations. ~1-2 hours.
4. **ByteTrack frame-skip + `co_presence.time_window_sec` + caption SigLIP batching** — three independent small fixes.
5. P2 cleanups as opportunity arises.
