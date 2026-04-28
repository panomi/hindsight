# Investigation Platform

Agentic search over large camera-footage collections. Investigators bring
100+ hours of video and an open-ended question — *"when do these two people
appear together?"*, *"someone reading a newspaper"*, *"license plate ABC123"*
— and a Claude-powered agent runs a tool-use loop over a pre-indexed
corpus, asks the user to confirm subjects of interest, and re-ranks based
on feedback.

## Quick start

```bash
./start.sh
```

This brings up Postgres+pgvector, Redis, the Celery worker (all stage
queues), the FastAPI server on :8000, and the Vite dev server on :5173.

You'll need:
- Docker
- `uv` (Python package manager)
- Node 20+ / npm
- `ffmpeg` and `ffprobe` on PATH
- `ANTHROPIC_API_KEY` in `backend/.env` (only for live agent runs)

## Two phases

### 1. Preprocessing (Celery DAG, all stages routed by queue)

```
shots → frames → detect_track → embed_global → embed_box →
   transcribe → caption → ocr
```

| Stage | Model | Mode |
|---|---|---|
| Shot boundaries | TransNetV2 | placeholder; fake = uniform 5s shots |
| Frame extraction | ffmpeg | **real** |
| Object detection + tracking | RT-DETR + ByteTrack | placeholder; fake = synthetic person tracks |
| Frame embeddings | SigLIP | placeholder; fake = deterministic seeded vectors |
| Box embeddings | SigLIP on crops | placeholder; fake = clustered by instance_id |
| Transcription | Parakeet (ASR) + BGE-small-en-v1.5 (embeddings) | placeholder; fake = one segment per shot |
| Captioning | Florence-2 + BGE-small-en-v1.5 (embeddings) | placeholder; fake = "Scene contains <classes>" |
| OCR | PaddleOCR (gated) | placeholder; fake = empty |

`USE_FAKE_ML=true` (default in `.env.example`) lets the entire pipeline run
on a dev machine without GPU. Flip to `false` on the GPU box (RTX 3090 /
G4dn / G6) and replace the `_real_*` stubs in each task module.

### 2. Interactive investigation

Claude orchestrates a tool-use loop. Tools are bounded by per-category
asyncio semaphores (`gpu_heavy=1`, `gpu_light=2`, `db_only=16`,
`external=4`) so an eager turn can't brown-out the GPU.

Every tool returns a structured `model_summary` (what Claude sees) and a
separate `ui_payload` (what the frontend renders) so context stays bounded
across many turns.

## Agent tools

| Tool | Purpose |
|---|---|
| `search_visual_embeddings` | Frame-level SigLIP ANN |
| `search_by_image` | Image search at frame OR detection-crop level |
| `search_instances_by_subject` | Two-pass cross-video: ANN match + within-video instance expansion |
| `get_object_detections` | Filter pre-computed RT-DETR by class/conf/time |
| `search_transcript` | Hybrid ASR search: exact substring → BM25 (stem-aware) → semantic (BGE-small) |
| `search_captions` | Hybrid caption search: exact substring → BM25 → semantic (BGE-small) |
| `search_ocr` | Text in scene (license plates, signs, screens) |
| `activity_search` | v1: caption ANN; v2: V-JEPA2 on candidates |
| `open_vocab_detect` | Florence-2 open-vocabulary detector (SigLIP pre-filter → detect on top-k → cached) |
| `co_presence` | Resolve every term to bboxes → proximity-join (center / IoU) |
| `temporal_cluster` | Group hits into events using shot boundaries |
| `scene_assembly` | Enter / dwell / exit timeline for a Subject |
| `caption_frames` | On-demand densification |
| `get_transcript_for_subject` | Transcript segments from videos where a Subject appears |
| `get_frames_around_transcript` | Frames + detections visible while a phrase was spoken |
| `request_user_confirmation` | Pause + emit SSE; wait for confirm/reject |
| `register_subject` | Build a Subject from confirmed detections (≥3); N=20 farthest-point cap |
| `apply_user_feedback` | Confirms expand the reference set; rejects flagged for sibling-subject hint |
| `get_video_clip_url` | Playable URL for evidence playback |

## Skills

Markdown procedural-knowledge files routed per turn:

- **Always loaded** — communication-style, investigation-strategy, result-synthesis, tool-result-budgeting, evidentiary-language, composition-patterns
- **Conditionally injected** — confirmation-ux, subject-registration-protocol, re-ranking-protocol, spatial-reasoning, open-vocab-prompting, temporal-reasoning, negative-result-handling

The `skill_router` selects based on (a) keywords in the user message and
(b) interception of upcoming tool calls. `skill_eval.py` runs a small drift
detector on every model upgrade.

## Subject identity model (the important clarification)

- **Within a video** — ByteTrack assigns `instance_id`. Meaningful only inside one video.
- **Across videos** — there is no shared id. Cross-video matching is SigLIP box-embedding similarity (max over the subject's reference set).
- **Subject** — a session-scoped identity. Stored as a SET of confirmed embeddings (`subject_references` table). Capped at 20 via farthest-point sampling.
- **Propagation** — confirmed/matched `(video_id, instance_id)` tuples are upserted to `subject_instances`. From there, every detection sharing that pair is treated as the subject.

v1 deliberately excludes face/biometric identification. Person identity is
appearance-based (clothing/build/accessories via SigLIP-on-crop) — weaker
than face matching but avoids the legal/ethical surface of biometric ID.

## Audit trail

Every tool call writes an `agent_actions` row:
`(investigation_id, ts, turn_index, tool, params_json, result_summary,
ui_payload_hash, result_count, duration_ms, confirmation_id, parent_action_id)`

To reconstruct a full investigation:
```sql
SELECT * FROM agent_actions
WHERE investigation_id = $1 ORDER BY ts;
```

## Layout

```
investigation-platform/
├── docker-compose.yml      postgres+pgvector + redis
├── start.sh                bring up everything
├── backend/
│   ├── pyproject.toml      uv
│   ├── alembic/            migrations (0001 schema → 0005 captions BGE+tsvector)
│   ├── app/
│   │   ├── api/            collections, videos, ingest, investigations, stream, confirmations
│   │   ├── worker/         Celery app + 8 stage tasks
│   │   ├── agent/          orchestrator, schemas, audit, concurrency, skill_router, skill_eval
│   │   │   ├── tools/      19 tools (registry in __init__.py)
│   │   │   └── skills/     13 markdown skill files
│   │   └── models/         video, detection, subject, agent
│   └── scripts/            smoke_tools.py, backfill_transcript_embeddings.py, backfill_caption_embeddings.py
└── frontend/
    ├── src/
    │   ├── api/            client, collections, investigations, stream
    │   ├── components/     ui/, ToolCallBadge, FrameGrid, InstanceGallery, TimelineBar, SubjectChip, ChatMessage
    │   ├── views/          Dashboard, Collections, Ingest, Investigate
    │   ├── hooks/          useAgentStream
    │   └── store/          Zustand
    └── tailwind.config.ts  white base + coral accent + glass
```

## v2 (deferred)

- TIPSv2 as a second concept index (only if SigLIP underperforms)
- V-JEPA2 as an on-demand `activity_search` over already-filtered candidate clips, with an action-prototype bank
- SAM 3.1 replacing Florence-2 for `open_vocab_detect` (calibrated confidence scores; pending PyTorch ≥ 2.7)
- OCR semantic embeddings (BGE-small, mirroring transcript/caption treatment)
