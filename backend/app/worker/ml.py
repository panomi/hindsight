"""Singleton ML model loaders, device/dtype management.

All loaders use functools.lru_cache so each Celery worker process loads a
given model exactly once. Stage-specific Celery queues mean a `gpu_embed`
worker only loads SigLIP, a `gpu_asr` worker only loads Parakeet, etc.

When a single worker subscribes to all GPU queues (the dev pattern) every
loader fires once and the models all sit in VRAM together. On a 24 GB
3090 the full set fits with ~10 GB to spare.

Usage:
    from app.worker.ml import siglip_vision, encode_image_batch
    model, processor = siglip_vision()
    embeddings = encode_image_batch(images)  # list[PIL.Image] -> list[list[float]]

Defensive: each loader wraps imports so a missing optional dep doesn't
crash the worker startup; the error surfaces only when that specific
model is actually called.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── device + dtype ─────────────────────────────────────────────────────────────

def _torch():
    import torch
    # Inference-only globals; safe to set at first import.
    torch.set_grad_enabled(False)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
    return torch


def device() -> str:
    t = _torch()
    if t.cuda.is_available():
        return "cuda"
    if hasattr(t.backends, "mps") and t.backends.mps.is_available():
        return "mps"
    return "cpu"


def vision_dtype():
    t = _torch()
    if device() == "cuda":
        return t.float16
    return t.float32


# ── SigLIP (frame embeddings, box embeddings, query encoders) ──────────────────

SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"
# SigLIP-SO400m emits 1152-dim embeddings; our pgvector schema uses Vector(512).
# We project to 512 with a fixed random Gaussian projection, seeded for determinism.
# Same projection is applied to images, crops, and text queries so cosine similarity
# is preserved (Johnson-Lindenstrauss). Avoids a schema migration to widen columns.
SIGLIP_PROJ_SEED = 42
SIGLIP_OUT_DIM = 512


@lru_cache(maxsize=1)
def _siglip_projection_for(in_dim: int):
    """Cached deterministic projection (in_dim → 512) keyed by input dim.
    Re-using lru_cache on a simple closure means we materialize the matrix
    exactly once per dim, even across many embed calls."""
    import numpy as np
    rng = np.random.default_rng(SIGLIP_PROJ_SEED)
    proj = rng.standard_normal((in_dim, SIGLIP_OUT_DIM)).astype("float32")
    proj /= in_dim ** 0.5
    return proj


@lru_cache(maxsize=1)
def siglip():
    """Load SigLIP model + processor (vision and text). Returns (model, processor)."""
    from transformers import AutoModel, AutoProcessor
    t = _torch()
    proc = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID)
    model = AutoModel.from_pretrained(SIGLIP_MODEL_ID, torch_dtype=vision_dtype())
    model.eval().to(device())
    return model, proc


def _project_to_512(features) -> "list[list[float]]":
    """features: torch.Tensor [B, D] → list of length-512 Python float lists.
    Accepts either a raw tensor or a ModelOutput object (transformers>=5.x)."""
    import numpy as np
    # transformers>=5 wraps outputs in dataclass objects; unwrap to the tensor.
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        features = features.pooler_output
    elif hasattr(features, "last_hidden_state"):
        features = features.last_hidden_state[:, 0]  # CLS token
    arr = features.detach().to("cpu", dtype=_torch().float32).numpy()
    proj = _siglip_projection_for(arr.shape[1])
    out = arr @ proj
    out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
    return out.astype("float32").tolist()


def siglip_encode_images(images: "list", batch_size: int = 32) -> list[list[float]]:
    """Encode a list of PIL.Image into 512-dim Python-float embeddings."""
    model, proc = siglip()
    t = _torch()
    out: list[list[float]] = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = proc(images=batch, return_tensors="pt").to(device())
        with t.inference_mode():
            feats = model.get_image_features(**inputs)
        out.extend(_project_to_512(feats))
    return out


def siglip_encode_text(texts: "list[str]", batch_size: int = 32) -> list[list[float]]:
    model, proc = siglip()
    t = _torch()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = proc(
            text=batch, padding="max_length", truncation=True, max_length=64,
            return_tensors="pt",
        ).to(device())
        with t.inference_mode():
            feats = model.get_text_features(**inputs)
        # Cast to match projection dtype expectations
        if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            feats = feats.pooler_output
        elif hasattr(feats, "last_hidden_state"):
            feats = feats.last_hidden_state[:, 0]
        out.extend(_project_to_512(feats))
    return out


# ── BGE-small (text-text retrieval for transcripts) ────────────────────────────
#
# SigLIP's text tower is trained for image-text alignment, not text-text
# similarity, and caps at 64 tokens — both bad for transcript retrieval where
# segments are sentences/paragraphs and queries like "they shouted fire"
# need true semantic matching against natural prose.
#
# BGE-small-en-v1.5 (384-dim, ~33M params) is a top-tier sentence encoder
# at this size, well above SigLIP for text-text on MTEB benchmarks, and
# fast enough to run on CPU if needed.

BGE_MODEL_ID = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def bge_text():
    """Load the BGE sentence encoder. Returns a SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BGE_MODEL_ID, device=device())
    model.eval()
    return model


def bge_encode_text(texts: "list[str]", batch_size: int = 64) -> list[list[float]]:
    """Encode a list of strings into 384-dim L2-normalized embeddings.

    Output is plain Python floats, ready to insert into pgvector(384).
    Empty input returns an empty list.  Empty strings are encoded as the
    zero vector (which yields cosine similarity 0 against anything else).
    """
    if not texts:
        return []
    model = bge_text()
    # SentenceTransformer handles batching/pooling/normalization in one call.
    embs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [[float(x) for x in row] for row in embs]


# ── RT-DETR (object detection) ─────────────────────────────────────────────────

RTDETR_MODEL_ID = "PekingU/rtdetr_v2_r50vd"  # RT-DETRv2 R50 — COCO 80 classes


@lru_cache(maxsize=1)
def rtdetr():
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
    proc = RTDetrImageProcessor.from_pretrained(RTDETR_MODEL_ID)
    model = RTDetrV2ForObjectDetection.from_pretrained(RTDETR_MODEL_ID, torch_dtype=vision_dtype())
    model.eval().to(device())
    return model, proc


def rtdetr_detect(images: "list", threshold: float = 0.4, batch_size: int = 8):
    """Returns list[list[dict]] aligned with `images`. Each dict has class_name,
    confidence (float), bbox (normalized {x1,y1,x2,y2})."""
    model, proc = rtdetr()
    t = _torch()
    id2label = model.config.id2label
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = proc(images=batch, return_tensors="pt").to(device())
        # Cast pixel_values to match model dtype (float16 on CUDA, float32 on CPU)
        inputs["pixel_values"] = inputs["pixel_values"].to(vision_dtype())
        with t.inference_mode():
            outputs = model(**inputs)
        target_sizes = t.tensor([img.size[::-1] for img in batch]).to(device())  # (h, w)
        post = proc.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )
        for img, p in zip(batch, post):
            w, h = img.size
            dets = []
            for score, label, box in zip(p["scores"], p["labels"], p["boxes"]):
                x1, y1, x2, y2 = (float(v) for v in box.tolist())
                dets.append({
                    "class_name": id2label[int(label)],
                    "confidence": float(score),
                    "bbox": {"x1": x1 / w, "y1": y1 / h, "x2": x2 / w, "y2": y2 / h},
                })
            results.append(dets)
    return results


# ── ByteTrack (within-video tracking) ──────────────────────────────────────────

@lru_cache(maxsize=1)
def bytetrack():
    """Returns a fresh tracker each call would be wrong — state is per video.
    Use bytetrack_new_tracker() at the start of each video instead."""
    raise RuntimeError("Use bytetrack_new_tracker() per video; state is stateful")


def bytetrack_new_tracker():
    """Fresh tracker for one video. ByteTrack expects per-frame detections and
    returns track ids that persist across frames within this tracker only."""
    from boxmot.trackers.bytetrack.bytetrack import ByteTrack
    return ByteTrack()


def bytetrack_step(tracker, dets: list[dict], img_w: int, img_h: int,
                   img_numpy=None) -> list[dict]:
    """Feed one frame's detections through the tracker. Mutates `dets` in
    place to add `instance_id` (or None if untracked). Returns the same list.

    img_numpy: uint8 HWC numpy array of the frame — required by boxmot>=11."""
    import numpy as np
    # boxmot>=11 requires a real numpy image; synthesise a blank one as fallback.
    if img_numpy is None:
        img_numpy = np.zeros((img_h, img_w, 3), dtype="uint8")
    if not dets:
        # ByteTrack still wants to be called per frame to advance its state.
        try:
            tracker.update(np.empty((0, 6)), img_numpy)
        except Exception:
            pass
        return dets
    # boxmot expects [x1, y1, x2, y2, conf, cls_id]; we use class_name strings,
    # so map them to a stable per-class int id for the tracker only.
    class_ids = {name: i for i, name in enumerate({d["class_name"] for d in dets})}
    arr = np.array([[
        d["bbox"]["x1"] * img_w, d["bbox"]["y1"] * img_h,
        d["bbox"]["x2"] * img_w, d["bbox"]["y2"] * img_h,
        d["confidence"], class_ids.get(d["class_name"], 0),
    ] for d in dets], dtype="float32")
    tracked = tracker.update(arr, img_numpy)
    # tracked rows: [x1, y1, x2, y2, track_id, conf, cls, det_idx]
    # Map track_id back to the originating detection by IoU (tracker may
    # drop or merge). We'll match by best-IoU.
    for d in dets:
        d["instance_id"] = None
    if tracked is None or len(tracked) == 0:
        return dets
    for row in tracked:
        x1, y1, x2, y2 = row[:4]
        tid = int(row[4])
        # Find the closest input detection by center distance
        best_idx, best_d = -1, 1e9
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        for i, d in enumerate(dets):
            dx = (d["bbox"]["x1"] + d["bbox"]["x2"]) / 2 * img_w - cx
            dy = (d["bbox"]["y1"] + d["bbox"]["y2"]) / 2 * img_h - cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_d:
                best_d, best_idx = dist, i
        if best_idx >= 0:
            dets[best_idx]["instance_id"] = tid
    return dets


# ── TransNetV2 (shot boundaries) ───────────────────────────────────────────────

@lru_cache(maxsize=1)
def transnetv2():
    """Returns the TransNetV2 model if `transnetv2-pytorch` is installed;
    otherwise None. Callers should fall back to `_opencv_shots` in that case."""
    try:
        from transnetv2_pytorch import TransNetV2  # type: ignore
    except ImportError:
        return None
    m = TransNetV2()
    m.eval().to(device())
    return m


def transnetv2_shots(video_path: str, threshold: float = 0.5) -> list[tuple[float, float]]:
    """Shot boundaries as [(start_sec, end_sec), ...]. Uses TransNetV2 when
    available, otherwise an OpenCV HSV histogram-distance heuristic.

    Uses the library's `detect_scenes()` public API which returns structured
    scene dicts with timestamps already in seconds. We run it in a daemon
    thread with a 2-minute hard ceiling because some codecs can cause the
    underlying ffmpeg frame iterator to hang.
    """
    import threading
    import cv2

    m = transnetv2()
    if m is not None:
        result: list = [None]
        exc: list = [None]

        def _detect():
            try:
                result[0] = m.detect_scenes(video_path, threshold=threshold)
            except Exception as e:
                exc[0] = e

        t = threading.Thread(target=_detect, daemon=True)
        t.start()
        t.join(timeout=120)

        if t.is_alive():
            logger.warning("transnetv2 timed out (>120s) for %s — using OpenCV fallback", video_path)
        elif exc[0] is not None:
            logger.warning("transnetv2 failed for %s (%s) — using OpenCV fallback",
                           video_path, exc[0])
        else:
            scenes = result[0] or []
            shots = [
                (float(s["start_time"]), float(s["end_time"]))
                for s in scenes
                if float(s["end_time"]) > float(s["start_time"])
            ]
            if shots:
                return shots
            logger.warning("transnetv2 returned no scenes for %s — using OpenCV fallback", video_path)

    # ── OpenCV histogram fallback ─────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps))  # sample ~1 frame/second

    boundaries_sec: list[float] = [0.0]
    prev_hist: np.ndarray | None = None
    frame_idx = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            corr = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
            if corr < (1.0 - threshold):
                boundaries_sec.append(frame_idx / fps)
        prev_hist = hist
        frame_idx += step

    cap.release()
    end_sec = (total or frame_idx) / fps
    if not boundaries_sec or boundaries_sec[-1] < end_sec - 0.3:
        boundaries_sec.append(end_sec)

    shots = [
        (float(boundaries_sec[i]), float(boundaries_sec[i + 1]))
        for i in range(len(boundaries_sec) - 1)
        if boundaries_sec[i + 1] - boundaries_sec[i] > 0.3
    ]
    return shots or [(0.0, end_sec)]


# ── Florence-2 (open-vocabulary detection only — captioning uses Qwen2.5-VL) ──

FLORENCE_MODEL_ID = "microsoft/Florence-2-base"

# Qwen2.5-VL-3B-Instruct is natively supported in transformers ≥ 4.49.
# Upgrade to Qwen3-VL-4B-FP8 once transformers ships qwen3_vl support.
QWEN_VL_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

_CAPTION_PROMPT = (
    "You are describing one video frame for an investigative search index.\n"
    "CONTEXT — other tools in this pipeline already provide:\n"
    "  - object detections (person, vehicle, bag, phone, etc., with positions)\n"
    "  - per-detection appearance embeddings (clothing colour, build, posture)\n"
    "  - OCR of any text in the scene (gated by what you flag below)\n"
    "  - audio transcription of speech\n"
    "Your job is to fill the gaps those tools cannot — primarily ACTIONS and SETTING.\n"
    "Write ONE neutral paragraph under 50 words. Cover, only when present:\n"
    "- WHAT IS HAPPENING — verbs and activities (handing, reaching, opening,\n"
    "  exiting, arguing, queueing, pointing, running, falling).\n"
    "- INTERACTIONS — who is near whom, who is doing what to whom or with what.\n"
    "- SCENE TYPE — kitchen, lobby, parking lot, alley, checkout counter, sidewalk.\n"
    "- DISTINCTIVE OPEN-VOCAB attributes detectors miss — limp, mask, cast, helmet,\n"
    "  wrapped package, gloves, visible tattoo, prosthetic.\n"
    "- TEXT CARRIER — if any readable text is visible, name its carrier with one of:\n"
    "  sign, screen, paper, plate, label, newspaper, menu.\n"
    "Rules:\n"
    "- Reference people / objects only as needed to anchor the verbs and relations.\n"
    "  Do NOT enumerate everything visible — other tools list object classes already.\n"
    "- Observation only — no intent, mood, identity, or speculation. No metaphors.\n"
    "- One paragraph, no lists, no headings."
)


@lru_cache(maxsize=1)
def qwen_vl():
    """Qwen2.5-VL-3B-Instruct — captions individual video frames.

    Loaded in BF16 (~6 GB VRAM on RTX 3090).
    TorchAo int4 quantization would cut this to ~2.5 GB but requires
    torch >= 2.11.0 (we run 2.5.1); revisit when torch is upgraded.
    """
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(QWEN_VL_MODEL_ID, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_VL_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device()},
    )
    model.eval()
    return model, proc


_QWEN_BATCH_SIZE = 4  # frames per forward pass; safe on 24 GB with 3B model


def qwen_vl_caption(image) -> str:
    """Caption a single PIL image — thin wrapper around the batch function."""
    return qwen_vl_caption_batch([image])[0]


def qwen_vl_caption_batch(images: list, batch_size: int = _QWEN_BATCH_SIZE) -> list[str]:
    """Caption a list of PIL images, processing *batch_size* frames per forward pass.

    Processing N frames together amortises the generate() call overhead and
    keeps the GPU at higher utilisation.  All video frames share the same
    resolution, so padding waste is minimal.

    Returns a list of caption strings in the same order as *images*.
    """
    return _qwen_vl_generate_batch(
        images=images,
        prompts=[_CAPTION_PROMPT] * len(images),
        max_new_tokens=80,
        batch_size=batch_size,
    )


_VQA_SYSTEM_INSTRUCTION = (
    "Answer the user's question about the image with a short, factual response. "
    "Read any visible text exactly as written. If the answer cannot be determined "
    "from the image alone, reply 'unknown'. Do not speculate."
)


def qwen_vl_vqa_batch(
    images: list,
    questions: list[str],
    max_new_tokens: int = 96,
    batch_size: int = _QWEN_BATCH_SIZE,
) -> list[str]:
    """Visual-question-answering: short factual answer per (image, question) pair.

    Distinct from `qwen_vl_caption_batch` in two ways:
      * Per-image prompt — the agent's free-form question, not the fixed
        ingest caption template.
      * Constrained instruction — read text literally, return "unknown" on
        ambiguity, no narration.  Captions are 50-word paragraphs; VQA
        answers are typically <30 tokens (a number, a phrase, "unknown").
    """
    if len(images) != len(questions):
        raise ValueError("qwen_vl_vqa_batch: images and questions must have equal length")
    prompts = [
        f"{_VQA_SYSTEM_INSTRUCTION}\n\nQuestion: {q.strip()}"
        for q in questions
    ]
    return _qwen_vl_generate_batch(
        images=images,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )


def _qwen_vl_generate_batch(
    images: list,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    """Shared VL generate loop used by both captioning and VQA paths."""
    if not images:
        return []

    model, proc = qwen_vl()
    t = _torch()
    results: list[str] = []

    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        batch_prompts = prompts[i : i + batch_size]

        texts = [
            proc.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for img, prompt in zip(batch_imgs, batch_prompts, strict=True)
        ]
        inputs = proc(
            text=texts,
            images=batch_imgs,
            padding=True,
            return_tensors="pt",
        ).to(device())

        with t.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        generated = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids, strict=True)]
        decoded = proc.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        results.extend(s.strip() for s in decoded)

    return results


@lru_cache(maxsize=1)
def florence2():
    from transformers import AutoModelForCausalLM, AutoProcessor
    proc = AutoProcessor.from_pretrained(FLORENCE_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL_ID, trust_remote_code=True, torch_dtype=vision_dtype()
    )
    model.eval().to(device())
    return model, proc


def florence2_caption(image, task: str = "<MORE_DETAILED_CAPTION>") -> str:
    """Caption a single PIL image with Florence-2 (kept as fallback for open-vocab detect)."""
    model, proc = florence2()
    t = _torch()
    inputs = proc(text=task, images=image, return_tensors="pt").to(device(), vision_dtype())
    with t.inference_mode():
        gen = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=256, num_beams=3, do_sample=False,
        )
    text = proc.batch_decode(gen, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(text, task=task, image_size=(image.width, image.height))
    return parsed.get(task) or text


def florence2_open_vocab_detect(image, labels: str, score_threshold: float = 0.3) -> list[dict]:
    """Open-vocabulary detection via Florence-2 OPEN_VOCABULARY_DETECTION task.

    labels: dot-separated noun phrases, e.g. 'person . airplane . cockpit'
    Returns [{bbox: {x1,y1,x2,y2} normalized 0-1, label, score}]

    TODO: swap to SAM 3.1 once pytorch>=2.7 and sam3 package are installed.
    SAM 3 install: pip install git+https://github.com/facebookresearch/sam3.git
    SAM 3 checkpoint: facebook/sam3.1 (gated, request access on HuggingFace)
    """
    model, proc = florence2()
    t = _torch()
    task = "<OPEN_VOCABULARY_DETECTION>"
    inputs = proc(text=task + labels, images=image, return_tensors="pt").to(device(), vision_dtype())
    with t.inference_mode():
        gen = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
    decoded = proc.batch_decode(gen, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(
        decoded, task=task, image_size=(image.width, image.height)
    )
    w, h = image.size
    results = []
    task_out = parsed.get(task, {})
    bboxes = task_out.get("bboxes", [])
    bbox_labels = task_out.get("bboxes_labels", [])
    for box, label in zip(bboxes, bbox_labels):
        x1, y1, x2, y2 = box
        results.append({
            "bbox": {"x1": x1 / w, "y1": y1 / h, "x2": x2 / w, "y2": y2 / h},
            "label": label,
            "score": 0.8,  # Florence-2 OVD does not emit confidence scores
        })
    return results


# ── Parakeet (ASR) ─────────────────────────────────────────────────────────────

PARAKEET_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"


@lru_cache(maxsize=1)
def parakeet():
    """Load Parakeet via NeMo. The first call downloads ~2.5 GB of weights."""
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as e:
        raise RuntimeError(
            "nemo-toolkit[asr] is not installed; required for real Parakeet ASR. "
            "Install with `uv sync --group ml` on Linux."
        ) from e
    model = nemo_asr.models.ASRModel.from_pretrained(PARAKEET_MODEL_ID)
    model.eval()
    if device() == "cuda":
        # fp16 halves VRAM: 0.6B params × 2 bytes ≈ 1.2 GB instead of 2.4 GB
        model = model.to(device()).half()
    return model


def _nemo_transcribe(model, inputs: list, **opts) -> tuple:
    """NeMo transcribe() wrapper that handles bug #15143 (word-confidence
    aggregation RuntimeError). Returns (hypotheses, bug_fired)."""
    try:
        return model.transcribe(inputs, **opts), False
    except RuntimeError as e:
        if "word-level confidence aggregation" not in str(e):
            raise
        # Retry without word confidence to work around the NeMo bug.
        no_conf_opts = {k: v for k, v in opts.items() if k != "word_confidence"}
        return model.transcribe(inputs, **no_conf_opts), True


def _wav_to_tensor(audio_wav: str):
    """Read a 16 kHz mono WAV into a 1-D float32 torch.Tensor."""
    import soundfile as sf
    import torch
    data, _ = sf.read(audio_wav, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    return torch.from_numpy(data)


def parakeet_transcribe_segments(audio_wav: str, progress_callback=None) -> list[dict]:
    """Transcribe a 16kHz mono WAV. Returns list of {text, start_seconds, end_seconds}.

    Manual waveform-level chunking (60 s windows, 10 s overlap) prevents the
    conformer's O(T²) attention from OOMing on long-form audio. Short clips
    (≤ 60 s) are passed through in one shot.
    """
    TARGET_SR = 16_000
    CHUNK_SEC = 60.0
    OVERLAP_SEC = 10.0
    STEP_SEC = CHUNK_SEC - OVERLAP_SEC
    MIN_CHUNK_SEC = 0.5

    m = parakeet()
    waveform = _wav_to_tensor(audio_wav)
    duration_sec = waveform.shape[0] / TARGET_SR

    transcribe_opts = dict(timestamps=True, return_hypotheses=True, batch_size=1)

    def _segments_from_hyp(hyp, time_offset: float = 0.0) -> list[dict]:
        ts = getattr(hyp, "timestamp", None) or {}
        raw = ts.get("segment", []) if isinstance(ts, dict) else []
        text_full = getattr(hyp, "text", None) or str(hyp)
        if not raw:
            # No segment timestamps — emit one segment for the whole window.
            return [{"text": text_full.strip(),
                     "start_seconds": time_offset,
                     "end_seconds": time_offset + duration_sec}]
        return [
            {
                "text": str(s.get("segment", "")).strip(),
                "start_seconds": float(s.get("start", 0.0)) + time_offset,
                "end_seconds": float(s.get("end", 0.0)) + time_offset,
            }
            for s in raw
            if isinstance(s, dict) and str(s.get("segment", "")).strip()
        ]

    # ── Short audio: single pass ────────────────────────────────────────────
    if duration_sec <= CHUNK_SEC:
        hyps, _ = _nemo_transcribe(m, [waveform], **transcribe_opts)
        hyp = hyps[0] if hyps else None
        if hyp is None:
            return []
        return _segments_from_hyp(hyp, time_offset=0.0)

    # ── Long audio: manual chunked pass ────────────────────────────────────
    segments: list[dict] = []
    start_sec = 0.0
    chunk_idx = 0

    while start_sec < duration_sec - 1e-3:
        chunk_idx += 1
        this_dur = min(CHUNK_SEC, duration_sec - start_sec)
        if this_dur < MIN_CHUNK_SEC:
            break
        chunk_wav = waveform[int(start_sec * TARGET_SR): int((start_sec + this_dur) * TARGET_SR)]
        if chunk_wav.numel() == 0:
            break

        if progress_callback:
            progress_callback(start_sec, duration_sec)

        hyps, _ = _nemo_transcribe(m, [chunk_wav], **transcribe_opts)
        hyp = hyps[0] if hyps else None
        if hyp is None:
            start_sec += STEP_SEC
            continue

        ts = getattr(hyp, "timestamp", None) or {}
        raw = ts.get("segment", []) if isinstance(ts, dict) else []

        # For non-first chunks, drop segments that fall entirely within the
        # overlap zone (already captured by the previous chunk).
        filtered = [
            s for s in raw
            if isinstance(s, dict)
            and not (chunk_idx > 1 and float(s.get("end", 0.0)) <= OVERLAP_SEC)
        ]

        for s in filtered:
            text = str(s.get("segment", "")).strip()
            if not text:
                continue
            segments.append({
                "text": text,
                "start_seconds": float(s.get("start", 0.0)) + start_sec,
                "end_seconds": float(s.get("end", 0.0)) + start_sec,
            })

        start_sec += STEP_SEC

    return segments


# ── PaddleOCR ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def rapidocr():
    """RapidOCR via ONNX Runtime — no Paddle/CUDA helper dependencies."""
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def paddleocr_run(image_path: str) -> list[dict]:
    """Run OCR on a single image. Returns list of {text, bbox, confidence}.

    bbox is normalized {x1,y1,x2,y2} relative to image dimensions.
    Uses RapidOCR (ONNX Runtime backend, PP-OCRv4 weights).
    """
    from PIL import Image
    ocr = rapidocr()
    result, _ = ocr(image_path)
    if not result:
        return []

    img = Image.open(image_path)
    w, h = img.size
    out: list[dict] = []
    for poly, text, conf in result:
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        out.append({
            "text": str(text),
            "confidence": float(conf),
            "bbox": {"x1": min(xs) / w, "y1": min(ys) / h,
                     "x2": max(xs) / w, "y2": max(ys) / h},
        })
    return out


# ── SAM 3.1 (open-vocab text-prompted detection + masks, single model) ─────────

SAM3_MODEL_ID = "facebook/sam3.1"


@lru_cache(maxsize=1)
def sam3():
    """SAM 3.1 supports text-prompted detection natively — no separate
    Grounding DINO needed. Loaded via HuggingFace transformers."""
    from transformers import AutoModelForMaskGeneration, AutoProcessor
    proc = AutoProcessor.from_pretrained(SAM3_MODEL_ID)
    model = AutoModelForMaskGeneration.from_pretrained(
        SAM3_MODEL_ID, torch_dtype=vision_dtype()
    )
    model.eval().to(device())
    return model, proc


def sam3_text_detect(image, text_prompt: str, score_threshold: float = 0.3,
                      include_mask_summary: bool = True) -> list[dict]:
    """Text-prompted detection + masks on a single PIL image.
    Returns [{bbox: {x1,y1,x2,y2} normalized, score, label, mask_area_norm}]."""
    model, proc = sam3()
    t = _torch()
    inputs = proc(images=image, text=text_prompt, return_tensors="pt").to(device())
    with t.inference_mode():
        outputs = model(**inputs)

    w, h = image.size
    # post_process is provided by the SAM3 processor and yields per-image dicts
    # with masks (bool tensors), boxes (xyxy in original coords), and scores.
    target_sizes = t.tensor([image.size[::-1]]).to(device())
    candidate_methods = (
        "post_process_grounded_segmentation",
        "post_process_instance_segmentation",
        "post_process_object_detection",
    )
    method = next((m for m in candidate_methods if hasattr(proc, m)), None)
    if method is None:
        raise RuntimeError(
            "SAM 3.1 processor exposes none of the expected post-process methods "
            f"({', '.join(candidate_methods)}). Check the transformers version "
            "and the model card for facebook/sam3.1."
        )
    results = getattr(proc, method)(
        outputs, target_sizes=target_sizes, threshold=score_threshold,
    )[0]

    out = []
    boxes = results.get("boxes")
    scores = results.get("scores")
    masks = results.get("masks")
    labels = results.get("labels") or [text_prompt] * (len(scores) if scores is not None else 0)
    if boxes is None or scores is None:
        return out

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = (float(v) for v in box.tolist())
        item = {
            "bbox": {"x1": x1 / w, "y1": y1 / h, "x2": x2 / w, "y2": y2 / h},
            "score": float(score),
            "label": str(labels[i]) if i < len(labels) else text_prompt,
        }
        if include_mask_summary and masks is not None and i < len(masks):
            m = masks[i]
            try:
                area = float(m.sum().item())
                item["mask_area_norm"] = area / float(w * h)
            except Exception:
                pass
        out.append(item)
    return out


# ── audio extraction (helper for ASR) ──────────────────────────────────────────

def extract_audio_16k_mono(video_path: str, out_wav: str) -> str:
    """ffmpeg → 16 kHz mono PCM. Required by Parakeet."""
    import subprocess
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
         "-sample_fmt", "s16", out_wav],
        check=True, capture_output=True, timeout=600,
    )
    return out_wav