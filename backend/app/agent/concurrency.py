"""Per-category concurrency caps. A single eager turn from the model
that fans out 8 parallel SAM 3.1 calls won't brown-out the GPU because
gpu_heavy is bounded to 1.
"""
import asyncio

SEMAPHORES: dict[str, asyncio.Semaphore] = {
    "gpu_heavy": asyncio.Semaphore(1),
    "gpu_light": asyncio.Semaphore(2),
    "db_only":   asyncio.Semaphore(16),
    "external":  asyncio.Semaphore(4),
}

# Map every tool to a category. Default is db_only.
TOOL_CATEGORY: dict[str, str] = {
    # GPU-heavy
    "open_vocab_detect": "gpu_heavy",
    "caption_frames": "gpu_heavy",
    "segment_region": "gpu_heavy",
    # GPU-light
    "search_by_image": "gpu_light",
    # everything else stays db_only
}


def category_for(tool: str) -> str:
    return TOOL_CATEGORY.get(tool, "db_only")
