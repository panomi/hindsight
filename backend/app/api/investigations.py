import asyncio
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import orchestrator
from app.api.schemas import (ConfirmationIn, InvestigationCreate, InvestigationMessageIn,
                             InvestigationOut)
from app.database import get_session
from app.models import AgentAction, Collection, Investigation, Message

router = APIRouter(prefix="/api/investigations", tags=["investigations"])


@router.post("", response_model=InvestigationOut, status_code=status.HTTP_201_CREATED)
async def create_investigation(
    payload: InvestigationCreate, session: AsyncSession = Depends(get_session)
) -> InvestigationOut:
    coll = await session.get(Collection, payload.collection_id)
    if coll is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "collection not found")
    inv = Investigation(collection_id=payload.collection_id, title=payload.title)
    session.add(inv)
    await session.commit()
    await session.refresh(inv)
    return InvestigationOut.model_validate(inv)


@router.get("", response_model=list[InvestigationOut])
async def list_investigations(
    session: AsyncSession = Depends(get_session)
) -> list[InvestigationOut]:
    rows = (await session.execute(
        select(Investigation).order_by(Investigation.created_at.desc())
    )).scalars().all()
    return [InvestigationOut.model_validate(r) for r in rows]


@router.get("/{investigation_id}", response_model=InvestigationOut)
async def get_investigation(
    investigation_id: UUID, session: AsyncSession = Depends(get_session)
) -> InvestigationOut:
    inv = await session.get(Investigation, investigation_id)
    if inv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
    return InvestigationOut.model_validate(inv)


@router.get("/{investigation_id}/history")
async def get_history(
    investigation_id: UUID, session: AsyncSession = Depends(get_session)
) -> dict:
    """Return the full conversation history for an investigation.

    Merges persisted user/assistant messages with tool-call audit rows, sorted
    by time, in the shape expected by the frontend ChatMessage store.
    """
    msgs = (await session.execute(
        select(Message)
        .where(Message.investigation_id == investigation_id)
        .order_by(Message.created_at)
    )).scalars().all()

    actions = (await session.execute(
        select(AgentAction)
        .where(AgentAction.investigation_id == investigation_id)
        .order_by(AgentAction.ts)
    )).scalars().all()

    events: list[dict] = []

    for m in msgs:
        content = m.content.get("content", "") if isinstance(m.content, dict) else ""
        if isinstance(content, str) and content.strip():
            events.append({
                "kind": m.role,
                "text": content,
                "ts": int(m.created_at.timestamp() * 1000),
            })

    for a in actions:
        events.append({
            "kind": "tool",
            "id": str(a.id),
            "tool": a.tool,
            "params": a.params_json or {},
            "status": "done",
            "summary": a.result_summary,
            "count": a.result_count,
            "durationMs": a.duration_ms,
            "ts": int(a.ts.timestamp() * 1000),
        })

    events.sort(key=lambda e: e["ts"])
    return {"events": events}


@router.post("/{investigation_id}/message", status_code=status.HTTP_202_ACCEPTED)
async def post_message(
    investigation_id: UUID,
    payload: InvestigationMessageIn,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> dict:
    inv = await session.get(Investigation, investigation_id)
    if inv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
    # Run the agent loop in the background; the client subscribes via SSE.
    background.add_task(_safe_run, investigation_id, payload.content)
    return {"ok": True}


async def _safe_run(investigation_id: UUID, content: str) -> None:
    try:
        await orchestrator.run_investigation_turn(investigation_id, content)
    except Exception as exc:  # pragma: no cover
        from app.agent import sse_bus
        await sse_bus.queue_for(investigation_id).put({
            "event": "error", "data": {"message": repr(exc)},
        })


@router.post("/{investigation_id}/confirm", status_code=status.HTTP_202_ACCEPTED)
async def post_confirm(investigation_id: UUID, payload: ConfirmationIn) -> dict:
    from app.agent.tools.confirmation import resolve_pending
    ok = resolve_pending(payload.confirmation_id, {
        "confirmed_ids": list(payload.confirmed_ids),
        "rejected_ids": list(payload.rejected_ids),
    })
    if not ok:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "confirmation not found / already resolved")
    return {"ok": True}
