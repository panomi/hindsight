"""Celery app with stage-specific queues per the platform plan.

Each ML stage has its own queue so workers can be sized to their compute
profile (GPU concurrency=1 to avoid VRAM contention; CPU queues higher).
"""
import logging

from celery import Celery
from celery.signals import setup_logging

from app.config import get_settings

settings = get_settings()


# Run worker at INFO so our pipeline's `[shots] / [frames] / [embed_*]` summary
# lines are visible, while silencing chatty third-party libraries that would
# otherwise drown them out. Triggered via @setup_logging so celery's own
# log config doesn't override ours.
_NOISY_LIBS = (
    "nemo",
    "nemo_logger",
    "nemo_toolkit",
    "transformers",
    "transformers.modeling_utils",
    "transformers.tokenization_utils_base",
    "transformers.configuration_utils",
    "huggingface_hub",
    "filelock",
    "urllib3",
    "accelerate",
    "paddle",
    "paddleocr",
    "ppocr",
    "pytorch_lightning",
    "lightning_fabric",
    "matplotlib",
    "PIL",
    "h5py",
)


@setup_logging.connect
def _configure_logging(**_kwargs):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s: %(levelname)s/%(processName)s] %(name)s: %(message)s",
    )
    for name in _NOISY_LIBS:
        logging.getLogger(name).setLevel(logging.WARNING)

celery_app = Celery(
    "invplatform",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.worker.tasks.ingest",
        "app.worker.tasks.shots",
        "app.worker.tasks.frames",
        "app.worker.tasks.detect_track",
        "app.worker.tasks.embed_global",
        "app.worker.tasks.embed_box",
        "app.worker.tasks.transcribe",
        "app.worker.tasks.caption",
        "app.worker.tasks.ocr",
    ],
)

celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_default_queue="cpu",
    # Stage-specific routing — see plan §"Celery: stage-specific queues"
    task_routes={
        "app.worker.tasks.ingest.*":        {"queue": "cpu"},
        "app.worker.tasks.shots.*":         {"queue": "gpu_light"},
        "app.worker.tasks.frames.*":        {"queue": "cpu"},
        "app.worker.tasks.detect_track.*":  {"queue": "gpu_detect"},
        "app.worker.tasks.embed_global.*":  {"queue": "gpu_embed"},
        "app.worker.tasks.embed_box.*":     {"queue": "gpu_embed"},
        "app.worker.tasks.transcribe.*":    {"queue": "gpu_asr"},
        "app.worker.tasks.caption.*":       {"queue": "gpu_vlm"},
        "app.worker.tasks.ocr.*":           {"queue": "cpu_heavy"},
    },
)
