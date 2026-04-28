"""v1 activity search: caption-text ANN over Qwen2.5-VL scene descriptions.

Most activity queries are answered via caption ANN with light keyword expansion.
The agent should fall back to caption_search if this returns nothing.
"""
from app.agent.tools import caption_search

SCHEMA = {
    "name": "activity_search",
    "description": "Find frames where a specific activity is happening "
                   "(running, loitering, handoff, falling, etc.) by searching "
                   "scene descriptions indexed at ingest time.",
    "input_schema": {
        "type": "object",
        "properties": {
            "concept": {"type": "string", "description": "The activity/concept term"},
            "video_ids": {"type": "array", "items": {"type": "string"}},
            "top_k": {"type": "integer", "default": 20, "maximum": 50},
        },
        "required": ["concept"],
    },
}


async def run(session, params, investigation_id):
    # Reuse caption_search; rename param.
    return await caption_search.run(
        session,
        {"query": params["concept"],
         "video_ids": params.get("video_ids", []),
         "top_k": params.get("top_k", 20)},
        investigation_id,
    )
