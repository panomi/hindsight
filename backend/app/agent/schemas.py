"""Tool I/O envelope.

The plan mandates that every tool returns:
  - model_summary: short, structured text the LLM sees
  - ui_payload:    full result detail streamed to the frontend (NOT the model)
  - page_token:    for pagination on large result sets
  - top_k_used:    always capped (default 50)
"""
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    model_summary: str
    ui_payload: dict[str, Any] = Field(default_factory=dict)
    page_token: str | None = None
    top_k_used: int = 0
    error: str | None = None


class ToolError(ToolResult):
    @classmethod
    def of(cls, msg: str) -> "ToolError":
        return cls(model_summary=f"error: {msg}", error=msg)
