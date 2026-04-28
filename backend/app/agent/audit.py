"""Structured agent action log for evidentiary use."""
import hashlib
import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.schemas import ToolResult
from app.models import AgentAction


def _hash_payload(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


async def record(
    session: AsyncSession,
    *,
    investigation_id: UUID,
    turn_index: int,
    tool: str,
    params: dict,
    result: ToolResult,
    duration_ms: int,
    confirmation_id: UUID | None = None,
    parent_action_id: UUID | None = None,
) -> UUID:
    action_id = uuid4()
    row = AgentAction(
        id=action_id,
        investigation_id=investigation_id,
        ts=datetime.now(timezone.utc),
        turn_index=turn_index,
        tool=tool,
        params_json=params,
        result_summary=(result.model_summary or "")[:4000],
        ui_payload_hash=_hash_payload(result.ui_payload),
        result_count=result.top_k_used,
        duration_ms=duration_ms,
        confirmation_id=confirmation_id,
        parent_action_id=parent_action_id,
    )
    session.add(row)
    await session.commit()
    return action_id
