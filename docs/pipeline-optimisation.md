# Ingest Pipeline Optimisation Notes

## Baseline (2026-04-27, single RTX 3090 24 GB)

Video: ~65 min (3892s) of footage, `DEFAULT_MAX_INTER_FRAME_SECONDS=3.0`

| Stage | Wall (s) | Speed |
|---|---|---|
| shots | 124 | 31× realtime |
| frames | 102 | 38× realtime |
| detect_track | 34 | 115× realtime |
| embed_global | 41 | 94× realtime |
| embed_box | 279 | 14× realtime |
| transcribe | 185 | 21× realtime |
| caption | 207 | 19× realtime |
| **ocr** | **1358** | **2.9× realtime** |
| **TOTAL** | **2337 (~39 min)** | **1.67× realtime** |

→ ~36 min of compute per hour of footage.

---

## Bottleneck Analysis

### 1. OCR — 58% of total wall time
PaddleOCR runs on every extracted frame, sequentially, one frame at a time.
For a 65-min video at 3 s/frame that is ~1300 frames through PaddleOCR.

**Fixes (priority order):**
- **Skip blank frames** — add a fast pre-filter (e.g. text-likelihood score or
  image variance threshold) and skip frames with no detectable text. Most
  surveillance/investigation frames have no on-screen text.
- **Run on a subset** — only run OCR on frames flagged by a lightweight
  text-detector (e.g. EAST or DBNet) rather than all frames.
- **Batch PaddleOCR** — PaddleOCR 3.x supports batch inference; switch from
  per-frame calls to batched calls.
- **Use GPU PaddleOCR** — verify `device="gpu"` is actually used; PaddleOCR
  has a separate CUDA backend that is ~10× faster than CPU.
- **Reduce frame coverage** — OCR only makes sense on frames with visible UI /
  text overlays. Consider running it only on the first frame of each shot.

### 2. embed_box — 12% of total time
SigLIP box-crop embeddings run on every detection crop. For a busy video
(many detections per frame) this can be very large.

**Fixes:**
- **Cap crops per frame** — if RT-DETRv2 returns 50 boxes per frame, we
  probably don't need embeddings for all 50; top-K by confidence (e.g. K=10).
- **Batch with global embed** — see vision_pass section below.

### 3. caption — 9% of total time
Florence-2 runs sequentially, one shot at a time (one `Image.open` + one
`generate` call per shot). Florence-2 supports batches of 4–8.

**Fixes:**
- **Batch Florence-2** — group shot keyframes into batches of 4–8 and run
  `generate` once per batch. Straightforward change to `caption.py`.

---

## Architectural Improvement: Single Vision Pass

**Proposed:** Replace the four separate Celery tasks
(`detect_track → embed_global → embed_box → caption`) with a single
`vision_pass` task.

**Current flow:**
```
frames → detect_track  (Image.open × N, RT-DETRv2 + ByteTrack)
       → embed_global  (Image.open × N, SigLIP global)
       → embed_box     (Image.open × N, SigLIP crops)
       → caption       (Image.open × 1/shot, Florence-2)
```
Each stage independently re-reads the same JPEG files from disk.
Total: **5 disk reads per frame** across the pipeline.

**Proposed flow:**
```
frames → vision_pass   (Image.open × 1 per frame, all models in one pass)
       → transcribe
       → ocr
```

### vision_pass logic (streaming, bounded RAM)

```python
# Group frames into batches of BATCH_SIZE (e.g. 8)
# Identify shot keyframes (first frame per shot) upfront
for batch in chunks(frames, BATCH_SIZE):
    images = [Image.open(f.filepath).convert("RGB") for f in batch]

    # 1. Detect (all frames)
    detections = rtdetr_detect(images)          # list[list[dict]]

    # 2. ByteTrack (sequential, stateful)
    for f, img, dets in zip(batch, images, detections):
        bytetrack_step(tracker, dets, img.width, img.height, np.array(img))

    # 3. Global SigLIP embed (all frames)
    global_embs = siglip_encode_images(images)

    # 4. Box crop SigLIP embed (only frames with detections)
    crop_embs = siglip_encode_images(crops_from(images, detections))

    # 5. Florence-2 caption (only shot keyframes in this batch)
    keyframes = [img for f, img in zip(batch, images) if f.id in keyframe_ids]
    if keyframes:
        captions = [florence2_caption(img) for img in keyframes]  # TODO: batch

    # 6. Write all results to DB (one transaction per batch)
    write_batch_to_db(db, batch, detections, global_embs, crop_embs, captions)
```

### Benefits
| | Current | vision_pass |
|---|---|---|
| Disk reads per frame | 5 | 1 |
| Peak RAM | all frames loaded at once in detect_track | bounded to BATCH_SIZE |
| DB round trips | 4 × SELECT frames | 1 × SELECT frames |
| Florence-2 | sequential, 1/shot | batched 4–8/call |

### Trade-offs
- Loses per-stage retry granularity (if Florence-2 fails, the whole pass fails)
- Slightly more complex single task
- Acceptable for single-GPU dev box; in production, stage queues can still
  be preserved by making vision_pass a single queue (`gpu_vision`)

### Files to change
1. **New** `app/worker/tasks/vision_pass.py`
2. **Update** `app/worker/tasks/ingest.py` — replace 4 stage entries with 1
3. **Update** `app/worker/tasks/ingest.py` — `_EVICT_AFTER` → `{"vision_pass"}`
4. Keep old task files as dead code (or delete)

---

## Next Actions (Prioritised)

### Action 1: OCR text-likelihood pre-filter (est. 5–10× OCR speedup)

Add a fast pre-filter before `paddleocr_run` that skips frames with no
detectable text. Two options:

**Option A — image variance / gradient threshold (zero extra model, ~0 ms/frame):**
```python
import numpy as np
from PIL import Image, ImageFilter

def _has_text_candidate(img_path: str, threshold: float = 800.0) -> bool:
    """Sobel edge energy as a cheap text-presence proxy.
    Frames with little edge content are very unlikely to contain text."""
    img = Image.open(img_path).convert("L").resize((128, 128))
    edges = img.filter(ImageFilter.FIND_EDGES)
    return float(np.array(edges).mean()) > threshold
```
Call this in `ocr.py` before `paddleocr_run(f.filepath)` and skip the frame
if it returns False. Tune the threshold on a representative sample.

**Option B — only OCR the first frame of each shot (structural heuristic):**
Text overlays (subtitles, timestamps, graphics) persist across a shot.
Running OCR on 1 frame/shot instead of all frames cuts frame count by
`DEFAULT_MAX_INTER_FRAME_SECONDS` × (frames/shot average) — roughly 3×.
Change the query in `ocr.py` to join Shot and only select one frame per shot.

Recommend starting with Option B (5 lines of SQL change) then adding Option A
on top.

### Action 2: Cap embed_box to top-K detections per frame (est. 2–3× embed_box speedup)

RT-DETRv2 returns up to 300 boxes per frame (though most are low confidence).
Even after the 0.4 threshold, busy frames can have 20–50 boxes. Embedding all
of them is wasteful.

Fix in `detect_track.py` or `ml.py` — sort detections by confidence and keep
only top K before passing to ByteTrack and embed_box:

```python
MAX_DETS_PER_FRAME = 20  # tune based on your footage density

# In rtdetr_detect, after building dets list:
dets.sort(key=lambda d: d["confidence"], reverse=True)
results.append(dets[:MAX_DETS_PER_FRAME])
```

This also speeds up ByteTrack (fewer boxes to match) and reduces DB rows.

---

## Other Quick Wins

### Frame sampling rate
`DEFAULT_MAX_INTER_FRAME_SECONDS=3.0` → ~0.33 FPS.
Tuning options:
- `1.0` → ~1 FPS, 3× frames, 3× compute for all vision stages
- `2.0` → middle ground
- Per-stage override (e.g. OCR only on 1 frame/shot) not currently supported
  but easy to add

### PaddleOCR angle classifier warning
```
ppocr WARNING: Since the angle classifier is not initialized, it will not be used
```
This fires once per frame because PaddleOCR 3.x removed `use_angle_cls`.
Fix: initialise with `use_textline_orientation=True` (already done) and
suppress the warning by setting `paddleocr` log level.

### Parakeet fp16
Already applied (`model.half()`). Cuts VRAM from ~2.4 GB to ~1.2 GB.

### RT-DETRv2
Already upgraded from RT-DETR (COCO+O365) to RT-DETRv2-R50 (COCO).
Note: v2 covers 80 COCO classes vs 445 (COCO+O365). If broader class coverage
is needed, check if a COCO+O365 v2 checkpoint becomes available, or switch
back to `PekingU/rtdetr_r50vd_coco_o365`.

---

---

## Efficient Video Intelligence — Upgrade Opportunities (April 2026)

Source: "Efficient Video Intelligence in 2026" (Efficient Track Anything / EUPE / LongVU group).

Each item is tagged by where it fits in our pipeline:
- **[INGEST]** — runs at ingest time, result stored in DB for downstream use
- **[AGENT]** — used at query time inside an agent tool step
- **[BOTH]** — applicable at both stages

---

### 1. DINOv2 temporal frame filtering — [INGEST] `frames.py`

**What:** Replace (or augment) our current ffmpeg scene-change filter with
DINOv2-based inter-frame similarity. Within non-overlapping 8-frame windows,
drop frames whose DINOv2 features are cosine-similar to their neighbours above
a threshold. LongVU reports ~45.9% frame retention — significantly reducing
downstream compute without losing informative moments.

**Why better than ffmpeg scene filter:** DINOv2 captures semantic similarity
(two nearly-identical frames of a person standing still register as redundant
even if pixel-level diff is non-zero due to motion blur / compression).

**How:** Load DINOv2 ViT-S/14 (via `torch.hub` or transformers) once at
ingest time; encode all candidate frames at low resolution (224×224); drop
frames above cosine similarity threshold (e.g. 0.95 to the previous kept
frame). Slot directly into `_samples_from_shots` or as a post-filter after
ffmpeg extracts candidates.

**Effort:** Medium — needs DINOv2 as a new lru_cache'd model in `ml.py` and
a refactor of `frames.py` post-extraction filter.

---

### 2. Per-detection Qwen2.5-VL crop descriptions — [INGEST] new stage or extend `embed_box`

**What:** For each RT-DETRv2 detection box, crop the frame and ask
Qwen2.5-VL-3B to describe it in detail:
> "Describe the person in this crop: clothing, approximate age/build, what
>  they are doing, anything they are carrying."

The text is SigLIP-embedded and stored on the Detection row (new column
`description TEXT`, `description_embedding vector(1152)`).

**Why:** Directly enables queries like "man in blue jacket with backpack"
at the individual-object level rather than only at the frame/caption level.
The article explicitly names spatial grounding ("who is the man in the blue
shirt") as one of the hardest unsolved problems; per-box VLM descriptions
are the tractable approximation.

**Combo:** RT-DETRv2 box → Qwen2.5-VL crop → SigLIP text embed → stored
per Detection row. At query time, `object_search` tool searches
`description_embedding` instead of (or alongside) `box_embedding`.

**Effort:** Medium — new Alembic migration for two columns, new ingest stage
or extend `embed_box.py`, Qwen prompt per crop (batched across detections
in a frame).

---

### 3. W4A16 quantization of Qwen2.5-VL — [INGEST] `ml.py`

**What:** Load Qwen2.5-VL-3B in 4-bit NF4 via BitsAndBytes:
```python
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    QWEN_VL_MODEL_ID, quantization_config=bnb_config, device_map={"": device()}
)
```
Cuts VRAM from ~6 GB to ~2.5 GB; inference speed increases ~1.5–2×.
Quality loss is minimal for captioning tasks (NF4 is designed for VLMs).

**Prereq:** None — `TorchAoConfig` is in transformers 4.53.3 and uses PyTorch's
native `torchao` quantization. No extra package required.

**Effort:** Low — 5-line change in `qwen_vl()` factory in `ml.py`.

---

### 4. Query-aware frame pre-selection — [AGENT] tool step

**What:** The Tempo / LongVU pattern: before running expensive Qwen inference
on candidate frames, use a lightweight model (DINOv2 or SigLIP) to select the
top-K frames most relevant to the query. This replaces the current
"retrieve all SigLIP matches, pass to agent" flow with a two-stage funnel:

1. Fast ANN search over SigLIP frame embeddings → top-50 candidates
2. Qwen2.5-VL reads each candidate and answers the specific question →
   filter to top-5 by Qwen confidence

Reduces hallucination on long videos and makes the agent's answers more
precise by grounding them in per-frame VQA rather than embedding similarity.

**How:** New agent tool `frame_vqa(query, frame_ids)` that loads frames,
calls `qwen_vl_caption` with a targeted question prompt, parses yes/no +
explanation.

**Effort:** Medium — new tool in `app/agent/tools/`, update
`investigation-strategy.md` skill to use it as a verification step.

---

### 5. EfficientSAM for open-vocabulary detection — [AGENT] tool replacement

**What:** Replace the broken Florence-2 `open_vocab_detect` tool with
EfficientSAM (ViT-T/S variants, public weights on HuggingFace). EfficientSAM
uses SAMI pretraining (distilled from SAM ViT-H) and is significantly more
reliable than Florence-2's OVD task for:
- Arbitrary object prompts ("find the dog", "locate the red car")
- Returning clean bounding boxes with masks
- Works offline (no HuggingFace connectivity required at runtime)

**Combo with RT-DETRv2:** Use RT-DETRv2 for known-class detection at ingest
(fast, 80 COCO classes), and EfficientSAM at query time for open-vocabulary
prompting on specific frames of interest.

**Models:** `yushan777/EfficientSAM` or `facebook/sam2-hiera-tiny` (SAM 2).

**Effort:** Medium — replace `florence2_open_vocab_detect` in `ml.py` and
update the `open_vocab_detect` agent tool.

---

### 6. EUPE universal encoder — [INGEST] long-term architectural

**What:** Replace SigLIP (semantic embeddings) + Florence-2 (OVD) with a
single <100M parameter EUPE encoder that covers:
- Language-aligned semantics (SigLIP capability)
- Dense spatial features (DINOv2 capability)
- Segmentation prompts (SAM capability)

This collapses 3 loaded models into 1, reducing VRAM and disk reads.

**Status:** EUPE weights are on HuggingFace; the model family includes
ViT-T/S/B and ConvNeXt T/S/B variants. Not yet production-tested with our
pipeline; worth evaluating once EfficientSAM is stable.

**Effort:** High — requires replacing SigLIP embedding dimensions (1152 →
EUPE output dim), Alembic migration, and thorough quality evaluation.

---

### 7. Qwen3-VL-4B-FP8 upgrade — [INGEST] blocked on nemo

**What:** Upgrade captioning from Qwen2.5-VL-3B to Qwen3-VL-4B-Instruct-FP8.
Better spatial reasoning, text reading, and multi-person scene understanding.

**Blocker:** `nemo-toolkit` pins `transformers~=4.53.0`; Qwen3-VL needs
`transformers>=4.54`. Unblocked when nemo releases a version with a looser
transformers constraint.

**Migration:** Two lines in `ml.py`:
```python
QWEN_VL_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct-FP8"
# swap Qwen2VLForConditionalGeneration → AutoModelForCausalLM
# add trust_remote_code=True
```

---

---

### 8. Qwen batched inference — [IMPLEMENTED 2026-04-28]

**What:** Instead of calling `model.generate()` once per frame, collect all
representative frames for the video upfront, then process them in batches of 4
in a single forward pass each.  The processor's `padding=True` handles
variable-length sequences; since all video frames share the same resolution
(and therefore the same image-token count) padding waste is negligible.

Also reduced `max_new_tokens` from 256 → 128. Scene descriptions are typically
40–80 tokens; 128 is a safe ceiling that cuts decode time roughly in half.

**Expected speedup:** 3–4× on the caption stage (was ~73% of total ingest wall
time). End-to-end estimated improvement: ~118 → ~45 min/hour of footage.

**Where:** `ml.py::qwen_vl_caption_batch()`, `tasks/caption.py`.

**Batching principle — apply everywhere:** Any inference loop over N homogeneous
inputs should collect inputs first, then call the model in chunks of
`batch_size` rather than one at a time.  The current pipeline stages that
already do this: `embed_global` (SigLIP), `embed_box` (SigLIP).
The stages that still process one item at a time and could benefit:
OCR (RapidOCR), open_vocab_detect (Florence-2).

---

### Priority Order

| # | Item | Impact | Effort | Status |
|---|---|---|---|---|
| 1 | Qwen batch inference (batch_size=4) | High (3–4× caption speed) | Done | ✅ implemented |
| 2 | W4A16 Qwen quantization | High (VRAM + speed) | Low | blocked (torch 2.5.1) |
| 3 | Per-detection Qwen crop descriptions | High (search precision) | Medium | pending |
| 4 | DINOv2 frame filtering | Medium (ingest speed) | Medium | pending |
| 5 | Query-aware frame VQA tool | High (agent precision) | Medium | pending |
| 6 | EfficientSAM open-vocab detect | Medium (tool reliability) | Medium | pending |
| 7 | Qwen3-VL-4B upgrade | Medium (caption quality) | Low | blocked (nemo/transformers) |
| 8 | EUPE universal encoder | High (architecture) | High | pending |

---

## GPU Memory Notes (RTX 3090 24 GB, dual-card shared machine)

Full model set resident: ~11.4 GiB
- TransNetV2: ~0.5 GiB
- RT-DETRv2 + ByteTrack: ~0.5 GiB
- SigLIP: ~1.8 GiB
- Parakeet: ~2.5 GiB
- Qwen2.5-VL-3B (BF16): ~6.0 GiB

Models stay resident via `lru_cache` across all stages — no reload latency.
OOM risk comes from stale worker processes from previous `./start.sh` runs;
`start.sh` now runs `pkill -f "celery.*celery_app"` before launching workers.
Docker containers use `restart: unless-stopped` to survive machine hiccups.

## Running long ingests without disruption (nohup)

```bash
# Kill any stale processes first, then launch detached:
pkill -f "celery.*celery_app" 2>/dev/null; sleep 1
cd /home/omiros/projects/investigation-platform
nohup ./start.sh > /tmp/invplatform.log 2>&1 &
echo "Started PID $!  —  tail -f /tmp/invplatform.log"
```

To monitor: `tail -f /tmp/invplatform.log`
To stop: `kill $(pgrep -f "start.sh")`  then `pkill -f "celery.*celery_app"`
