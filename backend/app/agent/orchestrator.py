"""Claude tool-use loop orchestrator.

Per the plan:
- parallel tool calls under per-category semaphores
- only `model_summary` enters the model context; full payloads stream to UI
- skill_router picks which skills to inject per turn
- every tool call writes an agent_actions audit row
- request_user_confirmation suspends on an asyncio.Event
"""
from __future__ import annotations

import asyncio
import inspect
import time
from uuid import UUID

from anthropic import AsyncAnthropic
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import audit, skill_router, sse_bus
from app.agent.concurrency import SEMAPHORES, category_for
from app.agent.schemas import ToolResult
from app.agent.tools import all_schemas, get_tool
from app.config import get_settings
from app.database import SessionLocal
from app.models import Message

settings = get_settings()


async def _fetch_history(session: AsyncSession, investigation_id: UUID) -> list[dict]:
    """Replay prior messages from DB so the loop continues from the last state.

    Two storage shapes coexist in `Message.content`:

    * `{"content": "<text>"}`  — display messages (user prompts and the
      agent's final assistant replies).  These render in the chat UI.
    * `{"blocks": [...]}`      — the raw Anthropic content blocks for an
      intermediate turn (assistant `text` + `tool_use`, or user
      `tool_result`).  Invisible to the chat UI but essential for the
      agent loop: without them the model loses every tool ID, score, and
      intermediate observation when the user reloads the page.

    We reconstruct both into the format the Anthropic Messages API expects.
    """
    rows = (await session.execute(
        select(Message).where(Message.investigation_id == investigation_id)
        .order_by(Message.created_at)
    )).scalars().all()
    out: list[dict] = []
    for r in rows:
        if r.role not in ("user", "assistant") or not isinstance(r.content, dict):
            continue
        if "blocks" in r.content:
            out.append({"role": r.role, "content": r.content["blocks"]})
        elif "content" in r.content:
            out.append({"role": r.role, "content": r.content["content"]})
    return out


async def _persist_message(session: AsyncSession, investigation_id: UUID,
                           role: str, content) -> None:
    """Persist a display message (plain text). Renders in the chat UI."""
    session.add(Message(investigation_id=investigation_id, role=role,
                        content={"content": content}))
    await session.commit()


async def _persist_blocks(session: AsyncSession, investigation_id: UUID,
                          role: str, blocks: list[dict]) -> None:
    """Persist a raw-blocks turn (assistant tool_use / user tool_result).

    Hidden from the chat UI but replayed verbatim by `_fetch_history` so the
    agent can pick up an interrupted multi-step investigation.
    """
    session.add(Message(investigation_id=investigation_id, role=role,
                        content={"blocks": blocks}))
    await session.commit()


async def _execute_tool(
    name: str, params: dict, investigation_id: UUID, sse_queue: asyncio.Queue,
) -> ToolResult:
    mod = get_tool(name)
    if mod is None:
        return ToolResult(model_summary=f"unknown tool: {name}",
                          ui_payload={}, error="unknown_tool")
    async with SessionLocal() as session:
        sig = inspect.signature(mod.run)
        kwargs = {}
        if "sse_queue" in sig.parameters:
            kwargs["sse_queue"] = sse_queue
        return await mod.run(session, params, investigation_id, **kwargs)


async def _gathered_with_audit(
    tool_uses, investigation_id: UUID, turn_index: int, sse_queue: asyncio.Queue,
) -> list[ToolResult]:
    async def run_one(tu) -> ToolResult:
        sem = SEMAPHORES[category_for(tu.name)]
        params = tu.input or {}
        await sse_queue.put({"event": "tool_start", "data": {
            "id": tu.id, "tool": tu.name, "params": _summarize_params(params),
        }})
        async with sem:
            t0 = time.monotonic()
            try:
                result = await _execute_tool(tu.name, params, investigation_id, sse_queue)
            except Exception as e:
                result = ToolResult(
                    model_summary=f"{tu.name} failed: {e!r}",
                    ui_payload={}, error=repr(e),
                )
            duration_ms = int((time.monotonic() - t0) * 1000)
        async with SessionLocal() as session:
            await audit.record(
                session, investigation_id=investigation_id, turn_index=turn_index,
                tool=tu.name, params=params, result=result, duration_ms=duration_ms,
            )
        await sse_queue.put({"event": "tool_result", "data": {
            "id": tu.id, "tool": tu.name,
            "summary": result.model_summary, "count": result.top_k_used,
            "ui_payload": result.ui_payload, "duration_ms": duration_ms,
        }})
        return result

    return await asyncio.gather(*(run_one(t) for t in tool_uses))


def _summarize_params(params: dict) -> dict:
    """Strip large fields (image_b64) before sending to UI to keep events tiny."""
    out = {}
    for k, v in params.items():
        if isinstance(v, str) and len(v) > 200:
            out[k] = f"<{len(v)} bytes>"
        else:
            out[k] = v
    return out


# How many times the same (tool, error_code) pair may recur in one investigation
# turn before we replace the next result with a hard "stop" instruction.
# Iterating with different params/IDs is fine; banging on the identical wall is not.
_MAX_IDENTICAL_FAILURES = 2


def _task_anchor_preamble(user_message: str) -> str:
    """Re-anchor the model on the *current* user request at the top of every turn.

    Prevents drift where unfinished sub-goals from prior turns (e.g. a stuck
    subject-registration loop) hijack a new, unrelated user question.
    """
    return (
        "# Current request\n\n"
        "The user's most recent message is:\n"
        f"\"{user_message.strip()}\"\n\n"
        "Stay focused on answering this request. If a sub-goal from a previous "
        "turn is unfinished but not required to answer the current message, "
        "set it aside and address what was just asked. Only continue prior "
        "work if it directly serves the current request."
    )


async def run_investigation_turn(
    investigation_id: UUID, user_message: str, max_turns: int = 20,
) -> None:
    """One full agent turn: persist user message, loop until end_turn or cap."""
    sse_queue = sse_bus.queue_for(investigation_id)
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    tool_schemas = all_schemas()

    async with SessionLocal() as session:
        await _persist_message(session, investigation_id, "user", user_message)
        messages = await _fetch_history(session, investigation_id)

    last_user_message = user_message
    pending_tool_calls: list[str] = []
    last_zero = False
    # (tool_name, error_code) → count of consecutive identical failures this turn.
    failure_counts: dict[tuple[str, str], int] = {}
    # Most recent assistant text emitted, in case we need a fallback persist.
    last_assistant_text: str = ""
    natural_exit = False

    for turn_index in range(max_turns):
        skills = skill_router.for_turn(
            last_user_message, pending_tool_calls=pending_tool_calls,
            last_tool_zero_results=last_zero,
        )
        system_prompt = skill_router.compose_system_prompt(
            skills, extra_preamble=_task_anchor_preamble(last_user_message),
        )

        response = await client.messages.create(
            model=settings.anthropic_model, max_tokens=2048,
            system=system_prompt, tools=tool_schemas, messages=messages,
        )
        text_blocks = [b for b in response.content if b.type == "text"]
        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if text_blocks:
            last_assistant_text = "".join(b.text for b in text_blocks)
            await sse_queue.put({"event": "message", "data": {
                "role": "assistant", "content": last_assistant_text,
            }})

        if response.stop_reason == "end_turn" or not tool_uses:
            async with SessionLocal() as session:
                if last_assistant_text:
                    await _persist_message(session, investigation_id, "assistant",
                                           last_assistant_text)
            natural_exit = True
            break

        results = await _gathered_with_audit(
            tool_uses, investigation_id, turn_index, sse_queue,
        )

        # Loop-detection: if a (tool, error_code) pair recurs >_MAX_IDENTICAL_FAILURES
        # times in this turn, replace the result the model sees with a hard stop.
        # We do NOT cap legitimate iteration (different params, success, or
        # different error) — only the "bang on the same wall" pattern.
        for tu, r in zip(tool_uses, results):
            if r.error is None:
                continue
            key = (tu.name, r.error)
            failure_counts[key] = failure_counts.get(key, 0) + 1
            if failure_counts[key] > _MAX_IDENTICAL_FAILURES:
                r.model_summary = (
                    f"BLOCKED — repeated identical failure: {tu.name} has "
                    f"returned error '{r.error}' {failure_counts[key]} times "
                    f"this turn. Do NOT call {tu.name} with the same approach "
                    f"again. Either change your strategy, ask the user for "
                    f"clarification with a text response, or end your turn."
                )

        pending_tool_calls = [t.name for t in tool_uses]
        last_zero = any(r.top_k_used == 0 and r.error is None for r in results)

        assistant_blocks = [b.model_dump() for b in response.content]
        tool_result_blocks = [
            {"type": "tool_result", "tool_use_id": tu.id, "content": r.model_summary}
            for tu, r in zip(tool_uses, results)
        ]
        messages.append({"role": "assistant", "content": assistant_blocks})
        messages.append({"role": "user", "content": tool_result_blocks})

        # Persist both turns so a page reload mid-investigation doesn't lose
        # the agent's tool history (P1 in the code audit).  Hidden from the
        # chat UI; only the orchestrator loop replays them.
        async with SessionLocal() as _persist_session:
            await _persist_blocks(_persist_session, investigation_id,
                                  "assistant", assistant_blocks)
            await _persist_blocks(_persist_session, investigation_id,
                                  "user", tool_result_blocks)

    # Safety net: if the loop exhausted max_turns without a natural end_turn,
    # persist *something* so the user isn't left staring at an empty UI.
    if not natural_exit:
        fallback = last_assistant_text or (
            f"I used my full reasoning budget ({max_turns} steps) without "
            "reaching a clean answer. The investigation log on the right shows "
            "what I tried. Please refine the question or pick one of the "
            "candidate items shown so I can focus the next pass."
        )
        async with SessionLocal() as session:
            await _persist_message(session, investigation_id, "assistant", fallback)
        await sse_queue.put({"event": "message", "data": {
            "role": "assistant", "content": fallback,
        }})

    await sse_queue.put({"event": "done", "data": {}})
