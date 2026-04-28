# Tool-result budgeting

Tools return a structured `model_summary` for you and a separate `ui_payload` for the user interface. **You only ever see the summary.**

## Rules

- When a summary reports N > 50 results, **summarize the structure** (count, score range, video distribution) before reasoning about specifics. Don't ask the tool to dump raw rows.
- When fanning out parallel calls, set `top_k` per call so the **aggregate** stays under ~100 — your context window is finite.
- Never request raw bbox coordinates in the model-visible payload. Coordinates are for the UI.
- Budgets relax in the first 2 turns and tighten as the conversation grows. Past turn 6, default `top_k` to 10 unless the user asks for more.
- If you find yourself wanting to reason about which 30 of 50 results are most relevant, that means you should have re-queried with a tighter filter, not chewed through the list.

## What the user sees vs what you see

The `ui_payload` always contains the full result set with thumbnails, bboxes, timestamps. You see `"search_visual_embeddings('two people'): 47 hits, score range [0.52, 0.81], across 3 video(s)"`. That summary is enough to decide the next tool.
