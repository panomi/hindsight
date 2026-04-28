"""Skill router — picks which markdown skills to inject into the system
prompt for the upcoming turn.

Three always-loaded skills (cross-cutting). Other skills are conditionally
injected based on:
  (a) keyword/intent classification of the user message, and
  (b) interception of the model's most recent response — if it's about to
      call a tool, we inject the tool-specific skill so the next response
      benefits from it.
"""
from __future__ import annotations

from pathlib import Path

SKILLS_DIR = Path(__file__).parent / "skills"

ALWAYS_ON = (
    "communication-style",
    "investigation-strategy",
    "result-synthesis",
    "tool-result-budgeting",
    "evidentiary-language",
    "composition-patterns",
)

# Tool name → skill that should be loaded when the model is about to call it.
# (composition-patterns is in ALWAYS_ON because it must steer tool *choice*,
# not just the call itself.)
TOOL_TRIGGERED = {
    "request_user_confirmation": "confirmation-ux",
    "register_subject": "subject-registration-protocol",
    "apply_user_feedback": "re-ranking-protocol",
    "co_presence": "spatial-reasoning",
    "open_vocab_detect": "open-vocab-prompting",
    "temporal_cluster": "temporal-reasoning",
    "scene_assembly": "temporal-reasoning",
}

# Substring → skill triggered by the user message
MESSAGE_TRIGGERED = [
    (("before", "after", "around", "shortly", "while", "during"), "temporal-reasoning"),
    (("near", "next to", "handoff", "contact", "together"), "spatial-reasoning"),
    (("running", "loitering", "falling", "fight", "struggle"), "negative-result-handling"),
    (("plate", "sign", "newspaper", "screen", "label"), "open-vocab-prompting"),
]


def _load(name: str) -> str:
    path = SKILLS_DIR / f"{name}.md"
    if not path.exists():
        return ""
    return path.read_text()


def always_on() -> list[str]:
    return list(ALWAYS_ON)


def for_turn(
    last_user_message: str | None,
    pending_tool_calls: list[str] | None = None,
    last_tool_zero_results: bool = False,
) -> list[str]:
    """Return the list of skill names to load this turn (in order)."""
    skills: list[str] = list(ALWAYS_ON)
    seen = set(skills)

    def add(name: str) -> None:
        if name and name not in seen:
            skills.append(name)
            seen.add(name)

    if pending_tool_calls:
        for t in pending_tool_calls:
            add(TOOL_TRIGGERED.get(t))

    if last_user_message:
        msg = last_user_message.lower()
        for keywords, skill in MESSAGE_TRIGGERED:
            if any(k in msg for k in keywords):
                add(skill)

    if last_tool_zero_results:
        add("negative-result-handling")

    return skills


def compose_system_prompt(skill_names: list[str], extra_preamble: str = "") -> str:
    """Concatenate skill markdown files into a system prompt."""
    parts = []
    if extra_preamble:
        parts.append(extra_preamble.strip())
    for name in skill_names:
        body = _load(name)
        if body:
            parts.append(f"# Skill: {name}\n\n{body.strip()}")
    return "\n\n---\n\n".join(parts)
