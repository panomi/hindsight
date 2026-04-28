"""Query-time embedding helpers used by retrieval tools.

Real mode wraps two distinct encoders:
- SigLIP for text/image (visual & caption similarity, 512-dim)
- BGE-small for transcript text (text-text retrieval, 384-dim)

In fake mode every helper returns a deterministic_unit_vector keyed by
content so dev/CI runs without any model download.
"""
import base64
import hashlib

from app.config import get_settings
from app.worker.util import deterministic_unit_vector

settings = get_settings()


def embed_text(query: str) -> list[float]:
    """SigLIP-text embedding (512-dim) for visual/caption similarity searches."""
    if settings.use_fake_ml:
        return deterministic_unit_vector(b"text:" + query.encode())
    return _real_siglip_text(query)  # pragma: no cover


def embed_text_bge(query: str) -> list[float]:
    """BGE-small embedding (384-dim) for text-text retrieval.

    Use this — not `embed_text` — when matching against any text-side
    embedding column generated at ingest time (transcripts, captions).
    SigLIP's text tower is trained for image-text alignment and performs
    poorly on text-text retrieval; BGE-small-en-v1.5 is the right tool.
    """
    if settings.use_fake_ml:
        # 384-dim deterministic vector for fake mode (matches schema).
        return deterministic_unit_vector(b"bge:" + query.encode(), dim=384)
    return _real_bge_text(query)  # pragma: no cover


def embed_image_b64(image_b64: str) -> list[float]:
    if settings.use_fake_ml:
        seed = hashlib.sha256(b"image:" + image_b64.encode()).digest()
        return deterministic_unit_vector(seed)
    return _real_siglip_image(base64.b64decode(image_b64))  # pragma: no cover


def _real_siglip_text(text: str) -> list[float]:
    """Encode a text query with SigLIP; project to 512-dim to match the schema."""
    from app.worker.ml import siglip_encode_text
    return [float(x) for x in siglip_encode_text([text])[0]]


def _real_bge_text(text: str) -> list[float]:
    """Encode a transcript-query string with BGE-small (384-dim, normalized)."""
    from app.worker.ml import bge_encode_text
    return [float(x) for x in bge_encode_text([text])[0]]


def _real_siglip_image(raw: bytes) -> list[float]:
    """Encode a single PIL-decodable image (raw bytes) with SigLIP."""
    import io

    from PIL import Image

    from app.worker.ml import siglip_encode_images
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return [float(x) for x in siglip_encode_images([img])[0]]