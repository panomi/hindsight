"""Tool registry exposed to Claude.

Each tool exports `SCHEMA` (Anthropic tool schema) and `run(session, params,
investigation_id) -> ToolResult`. The orchestrator dispatches by name.
"""
from importlib import import_module

# Module name → tool name (matches the JSON tool spec presented to Claude)
TOOL_MODULES: dict[str, str] = {
    "search_visual_embeddings":   "app.agent.tools.visual_search",
    "search_by_image":            "app.agent.tools.image_search",
    "search_instances_by_subject": "app.agent.tools.instance_search",
    "open_vocab_detect":          "app.agent.tools.open_vocab_detect",
    "get_object_detections":      "app.agent.tools.detection_query",
    "search_transcript":          "app.agent.tools.audio_search",
    "search_captions":            "app.agent.tools.caption_search",
    "search_ocr":                 "app.agent.tools.ocr_search",
    "activity_search":            "app.agent.tools.activity_search",
    "co_presence":                "app.agent.tools.co_presence",
    "scene_assembly":             "app.agent.tools.scene_assembly",
    "temporal_cluster":           "app.agent.tools.temporal_cluster",
    "caption_frames":             "app.agent.tools.caption_frames",
    "get_transcript_for_subject": "app.agent.tools.transcript_for_subject",
    "get_frames_around_transcript": "app.agent.tools.frames_around_transcript",
    "request_user_confirmation":  "app.agent.tools.confirmation",
    "register_subject":           "app.agent.tools.subjects",
    "apply_user_feedback":        "app.agent.tools.reranking",
    "get_video_clip_url":         "app.agent.tools.video_clip",
}


def all_schemas() -> list[dict]:
    schemas = []
    for name, mod in TOOL_MODULES.items():
        m = import_module(mod)
        schemas.append(m.SCHEMA)
    return schemas


def get_tool(name: str):
    mod = TOOL_MODULES.get(name)
    if mod is None:
        return None
    return import_module(mod)
