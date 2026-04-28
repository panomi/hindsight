"""Shared utilities for agent tools.

These helpers enforce per-investigation scoping so tools cannot leak data
across collections. Every search tool MUST call `scope_videos` and bail when
it returns an empty list. Every tool that takes a `subject_id` MUST call
`validate_subject` so subjects from another investigation cannot be queried
by guessing UUIDs.
"""
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.schemas import ToolResult
from app.models import Investigation, Subject, Video


async def scope_videos(
    session: AsyncSession,
    investigation_id: UUID,
    explicit_ids: list[str],
) -> list[UUID]:
    """Return the effective video UUID filter for a search tool.

    Behaviour:

    * If the agent explicitly passed `video_ids`, intersect them with the
      investigation's collection so the agent cannot escape its scope by
      guessing UUIDs from another collection.
    * Otherwise auto-restrict to ready videos in the investigation's
      collection.
    * Returns an EMPTY list when there is nothing in scope (investigation
      not found, no ready videos, or all explicit ids belong to other
      collections).  Callers MUST treat an empty list as "0 results"; never
      use it as "no filter" — that's the bug this helper exists to prevent.
    """
    inv = await session.get(Investigation, investigation_id)
    if inv is None:
        return []

    # Always-on guard: only ready videos in this investigation's collection
    # are ever in scope.
    in_scope_stmt = select(Video.id).where(
        Video.collection_id == inv.collection_id,
        Video.status == "ready",
    )
    in_scope = set((await session.execute(in_scope_stmt)).scalars().all())
    if not in_scope:
        return []

    if explicit_ids:
        requested = {UUID(v) for v in explicit_ids}
        # Intersect — silently drop ids that don't belong to this scope.
        return [v for v in requested if v in in_scope]

    return list(in_scope)


async def validate_subject(
    session: AsyncSession,
    subject_id: UUID,
    investigation_id: UUID,
) -> Subject | None:
    """Return the Subject only if it belongs to this investigation.

    Returns None if the subject doesn't exist OR belongs to a different
    investigation. Callers should treat None as "not found" — do NOT leak
    "exists but not yours" via the error message.
    """
    subject: Subject | None = await session.get(Subject, subject_id)
    if subject is None or subject.investigation_id != investigation_id:
        return None
    return subject


def empty_scope_result(tool_name: str, query_hint: str = "") -> ToolResult:
    """Standard 'nothing in scope' tool result.

    Use this when `scope_videos` returns an empty list. The wording mirrors
    a normal '0 results' message so the agent treats it as "nothing matched"
    rather than as an error.
    """
    suffix = f"('{query_hint[:60]}')" if query_hint else ""
    return ToolResult(
        model_summary=(
            f"{tool_name}{suffix}: 0 results — no ready videos in this "
            f"investigation's collection."
        ),
        ui_payload={"results": []},
        top_k_used=0,
    )
