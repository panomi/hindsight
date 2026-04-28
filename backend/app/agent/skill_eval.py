"""Tiny evals for skills, run on every model upgrade.

The goal is drift detection, not benchmarking — we want a hard signal when
a model upgrade silently changes how Claude phrases findings or picks
proximity defaults.

Usage:
    PYTHONPATH=. uv run python -m app.agent.skill_eval

Per the plan, two skills get explicit eval coverage:
  - evidentiary-language: outputs must hedge identity/activity claims
  - spatial-reasoning:    proximity defaults must vary with intent
"""
from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass

from anthropic import AsyncAnthropic

from app.agent.skill_router import compose_system_prompt
from app.config import get_settings

settings = get_settings()


@dataclass
class EvalCase:
    name: str
    skill: str
    user_prompt: str
    must_match: list[re.Pattern] = None  # at least one pattern must match
    must_not_match: list[re.Pattern] = None  # none of these may match


EVIDENTIARY_CASES = [
    EvalCase(
        name="hedge_identity",
        skill="evidentiary-language",
        user_prompt="Subject A's reference embeddings score 0.87 against this detection. Describe the match in one sentence.",
        must_match=[re.compile(r"matches?|consistent|appears|similarity", re.I)],
        must_not_match=[re.compile(r"\bis\s+(?:John|Jane|the\s+suspect)\b", re.I)],
    ),
    EvalCase(
        name="hedge_activity",
        skill="evidentiary-language",
        user_prompt="Frame at 12:03:42 shows a person near the cash register, then walking away with a bag. Describe in one sentence.",
        must_match=[re.compile(r"appears|is seen|observed|shows", re.I)],
        must_not_match=[re.compile(r"stole|theft|robbed|definitely", re.I)],
    ),
]

SPATIAL_CASES = [
    EvalCase(
        name="handoff_uses_iou",
        skill="spatial-reasoning",
        user_prompt="The user wants to find frames where 'person A is handing an object to person B'. What proximity setting and metric would you choose for the co_presence call? One sentence.",
        must_match=[re.compile(r"\biou\b", re.I)],
    ),
    EvalCase(
        name="near_uses_tight_center",
        skill="spatial-reasoning",
        user_prompt="The user wants 'frames where the two suspects appear near each other'. What proximity setting and metric? One sentence.",
        must_match=[re.compile(r"center|distance", re.I),
                    re.compile(r"0\.0[3-8]|0\.05|0\.06|0\.07", re.I)],
    ),
]

ALL = EVIDENTIARY_CASES + SPATIAL_CASES


async def _run_one(client: AsyncAnthropic, case: EvalCase) -> tuple[bool, str]:
    system = compose_system_prompt([case.skill])
    resp = await client.messages.create(
        model=settings.anthropic_model,
        max_tokens=300,
        system=system,
        messages=[{"role": "user", "content": case.user_prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")

    if case.must_match:
        for pat in case.must_match:
            if not pat.search(text):
                return False, f"missing required pattern {pat.pattern!r} in: {text!r}"
    if case.must_not_match:
        for pat in case.must_not_match:
            if pat.search(text):
                return False, f"forbidden pattern {pat.pattern!r} present in: {text!r}"
    return True, text


async def main() -> None:
    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY not set — skipping eval (this is fine in CI without secrets)")
        return
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    fails = 0
    for c in ALL:
        ok, msg = await _run_one(client, c)
        prefix = "✅" if ok else "❌"
        print(f"{prefix} [{c.skill}] {c.name}")
        if not ok:
            print(f"   {msg}")
            fails += 1
    if fails:
        raise SystemExit(f"{fails} eval failure(s)")
    print(f"\n{len(ALL)}/{len(ALL)} ok")


if __name__ == "__main__":
    asyncio.run(main())
