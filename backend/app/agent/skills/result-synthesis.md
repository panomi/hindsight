# Result synthesis

After `temporal_cluster` returns events, apply this decision logic before writing your response.

## Convergence scoring

An event's confidence level depends on how many independent searches agree:

| Tools that hit the same time window | Confidence |
|---|---|
| 3 or more | **Very high** — lead with this finding |
| 2 | **High** — state directly |
| 1 | **Provisional** — flag for user review |
| 0 | Not an event — discard |

"Same time window" means within ±30 seconds of an event's start/end.

## Deduplication

Before presenting, check if two events overlap (one ends after the other starts). If yes, merge them into a single event spanning both — use the union of their timestamps and sum their frame counts.

## Ranking

Rank events by: (1) number of converging tools, then (2) frame count, then (3) earlier timestamp.
Present the top 5 after ranking. If fewer than 5 events exist, present all.

## When temporal_cluster returns 0 events

This happens when the frame_id pool was empty (tools returned nothing). Report this plainly:

- State which tools were run
- State what score threshold was applied
- Suggest one specific query refinement (e.g., "try a broader description" or "this topic may not appear in the transcripts")

Do NOT apologize or hedge extensively. One clear sentence is enough.

## When confidence is low across the board

If all events are provisional (single-tool hits), say:

> "I found N candidate moments but none are confirmed by multiple searches. The most likely match is at `HH:MM:SS`. I recommend reviewing the clip before drawing conclusions."

Then provide the clip link and stop. Do not pad with more results.
