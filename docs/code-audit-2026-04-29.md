# Code Audit — Apr 29, 2026

Follow-ups discovered during the chunk 4 (transcripts) and chunk 5
(captions) work.  See `code-audit-2026-04-28.md` for the previous round
and what's already been resolved.

---

## P1 — Real isolation gap

### REST byte endpoints have no collection / investigation auth

**Where:** `backend/app/api/videos.py:72-92`, `backend/app/api/frames.py:51-57`, `backend/app/api/investigations.py:41-97`

**Status today:**

* **Agent layer is airtight.**  Every search tool routes through
  `scope_videos(session, investigation_id, …)`, which only returns
  `ready` videos in that investigation's collection.  Subjects are
  validated via `validate_subject()`.  All P0 scoping bugs from the
  Apr 28 audit are closed.
* **REST byte / metadata endpoints take a UUID and serve the bytes
  with no caller-side check.**

```python
# videos.py
@router.get("/{video_id}/file")
async def get_video_file(video_id: UUID):
    filepath = await _fetch_video_filepath(video_id)
    if filepath is None:
        raise HTTPException(404, "video not found")
    return FileResponse(filepath, media_type="video/mp4")
```

Same shape for `GET /api/videos/{video_id}`, `GET /api/frames/{frame_id}/image`, `GET /api/investigations/{investigation_id}` and `GET /api/investigations/{investigation_id}/history`.

**Why it doesn't surface in the UI:** the frontend is well-behaved — it
only fetches `/api/videos?collection_id=…` (which is filtered correctly)
and only requests frame URLs for `frame_id`s returned by an
investigation-scoped agent tool.  So clicking around can never show a
foreign-collection video.

**The risk:** anyone (or any process) that knows / guesses a UUID can
pull the bytes / metadata directly.  A pasted video URL crosses
collection boundaries.  Single-user dev: harmless.  Multi-tenant or
multi-user: unacceptable.

### Proposed fix (≈50 LOC)

Make collection membership the unit of authorisation, with the
investigation acting as the "session token" the frontend already has:

1. **Frontend always sends `X-Investigation-Id`** on every request that
   targets a UUID-keyed resource (`/api/videos/{id}`, `/api/videos/{id}/file`, `/api/frames/{id}/image`, `/api/investigations/{id}`, `/api/investigations/{id}/history`).  Adds one line to the axios / fetch wrapper.

2. **Middleware-style FastAPI dependency** `require_resource_in_investigation(resource_kind: str, resource_id: UUID, investigation_id: UUID)`:
   * Loads the resource's `collection_id` (cached — already cache `frame_id → filepath`, same shape).
   * Loads the investigation's `collection_id`.
   * Returns `403` if they don't match, `404` if either doesn't exist.

3. **Apply the dependency** to the four byte / metadata endpoints listed
   above.  Pure additive change, no schema migration.

4. **Open question (not blocking):** do we also want a single
   `request_id → user_id` notion later for true multi-user, or is
   collection-as-tenant enough?  For the current product surface
   (single-user investigator console) collection-level auth is
   sufficient and proportionate.

**Effort:** ~½ hour code, ~½ hour testing (verify cross-investigation
URL paste returns 403; verify intra-investigation URLs still work).
No DB change, no model loading, no UX change.

---

## P2 — Carried over from Apr 28 audit, now next in line

These were ranked P2 a day ago; they're still P2, but worth re-listing
so they don't drift further:

* **`get_frames_in_window` helper missing.**  When a key shot lasts
  <1 s and only one keyframe was sampled (e.g. the *Murder Over New
  York* schedule-board pan at t=49.4 s), the agent has no way to
  densify the timeline — `get_frames_around_transcript` only fires off
  a transcript hit, not a generic timestamp.  Add
  `get_frames_in_window(video_id, start, end, max_frames=6)` that
  returns up to N frames sampled raw from the video file (not just the
  ingest keyframes) so `ask_vision` / OCR can re-read at higher density.
  ~50 LOC; needs an FFmpeg seek + decode helper.  Skipped in the
  Apr 29 ask_vision rollout to keep that change focused.
* **OCR text embeddings still use SigLIP.**  Same model-fit issue we
  fixed for transcripts (chunk 4) and captions (chunk 5), but
  lower-impact because OCR queries are dominated by the substring path
  (license plates, exact signs).  When picked up: mirror migration 0005
  against `ocr_texts`, swap `ocr.py` to `bge_encode_text`, optionally
  rewrite `search_ocr` as hybrid (substring → BM25 → semantic) keeping
  substring-first.
* **Open-vocab detector replacement.**  Florence-2 emits no real
  confidence scores (we hard-code `0.8` in `ml.py`).  Once PyTorch ≥ 2.7
  lands, swap to SAM 3.1 (gated HF access already granted) or Grounding
  DINO for calibrated scores.
* **`temporal_cluster` silent length mismatch.**  If `frame_ids` and
  `scores` have different lengths the relative-threshold filter
  collapses silently.  Should raise.
* **`co_presence.time_window_sec`** declared in the schema but unused
  by the implementation.
* **OCR opens the image twice per frame** (`ml.py:673` + `ml.py:677`)
  — wasted I/O.
* **`update_status` always commits**, splitting transactions in
  fragile ways.

---

## P3 — Code health, no behaviour change

* Several search tools share near-identical pgvector boilerplate
  (cosine_distance + order + limit + format CSV) — extract a helper.
* Dead code: `embed_box._key_hash`, `ocr.text_owners`.
* `confirmation.PENDING` global never cleared on timeout (memory leak
  in long-running workers).
* Several places catching bare `Exception: pass` swallow real errors.

---

## Recommended order

1. **Auth-tight middleware (P1 above)** — small patch, removes the
   only real cross-collection leak.  Should ship before any multi-user
   demo.
2. **OCR → BGE migration** — mirrors chunk 4 / chunk 5 exactly, ~30
   min, no surprises.
3. **`temporal_cluster` fail-loud + `co_presence.time_window_sec`** —
   two small correctness fixes that prevent confusing silent bugs.
4. **OVD swap (Florence-2 → SAM 3.1 / Grounding DINO)** — bigger,
   blocked on PyTorch upgrade.  Tackle when the upgrade window opens.
