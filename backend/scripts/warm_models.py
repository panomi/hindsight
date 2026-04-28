"""One-shot model warm-up.

Run on the GPU box after `uv sync --group ml` and before the first ingest.
Pre-downloads every weight the pipeline needs so a long-running Celery
task doesn't time out mid-pipeline waiting on a 4 GB pull.

Usage:
    cd backend
    PYTHONPATH=. uv run python scripts/warm_models.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def warm(label: str, fn):
    print(f"  • {label} … ", end="", flush=True)
    t0 = time.monotonic()
    try:
        fn()
        print(f"ok ({time.monotonic() - t0:.1f}s)")
    except Exception as e:
        print(f"FAILED — {e!r}")


def main():
    print("Warming model weights (pulls happen on first call):")

    from app.worker import ml

    print(f"  device={ml.device()}, dtype={ml.vision_dtype()}")

    warm("SigLIP-SO400M (so400m-patch14-384)", lambda: ml.siglip())
    warm("BGE-small-en-v1.5 (transcript text encoder)", lambda: ml.bge_text())
    warm("RT-DETRv2 (r50vd)", lambda: ml.rtdetr())

    # TransNetV2 is optional — if transnetv2-pytorch isn't installed, the
    # shots stage falls back to an OpenCV histogram detector at runtime.
    def _tn():
        m = ml.transnetv2()
        if m is None:
            raise RuntimeError("transnetv2-pytorch not installed — using OpenCV fallback at runtime")
    warm("TransNetV2 (optional; OpenCV fallback if missing)", _tn)

    warm("Florence-2-base", lambda: ml.florence2())
    warm("SAM 3.1 / OWL-v2 (open-vocab detect)", lambda: ml.sam3())

    # Heavy / Linux-only; skip silently if the optional dep isn't installed
    def _parakeet():
        try:
            ml.parakeet()
        except RuntimeError as e:
            print(f"\n      (skipping Parakeet: {e})", end="")
    warm("Parakeet TDT 0.6B v3", _parakeet)

    def _ocr():
        try:
            ml.paddleocr()
        except Exception as e:
            print(f"\n      (skipping PaddleOCR: {e})", end="")
    warm("PaddleOCR (en)", _ocr)

    print("\nDone. You can now flip USE_FAKE_ML=false in backend/.env")


if __name__ == "__main__":
    main()