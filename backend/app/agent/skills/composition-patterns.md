# Composition patterns — grounding tools by other tools' results

Tool results in this system are kept compact (one-line `model_summary`) so they
don't blow your context window. The full per-item data (timestamps, IDs,
bboxes, transcript text) is rendered to the user but **not echoed back into
your context**. That means you cannot reliably chain tools by hand — e.g. you
cannot call `search_transcript`, read its timestamps, and feed them into
`get_object_detections`, because those timestamps were never in your messages.

When you find yourself thinking "I'll call A and then call B with A's
results", check this catalogue first. There is almost always a single
**composer tool** that does the join inside the database.

## Recipes

| User intent | Tool to use |
|---|---|
| "What did Subject X say?" / "Any quotes from this person?" | `get_transcript_for_subject(subject_id=X)` |
| "Show me the person who said 'Y'" / "Who's on screen when 'Y' is said?" | `get_frames_around_transcript(query='Y', classes=['person'])` |
| "Subject X with a phone / car / bag nearby" | `co_presence(terms=[{subject_id: X}, {class_name: 'phone'}])` |
| "Anyone interacting with Subject X" | `co_presence(terms=[{subject_id: X}, {class_name: 'person'}])` |
| "Where does Subject X appear?" (no speech, no co-actor) | `search_instances_by_subject(X)` |
| "Vehicles Subject X used / entered" | `co_presence(terms=[{subject_id: X}, {class_name: 'car'}])` plus widen `proximity` to ~0.15 for "near", and let `time_window_sec` cover entry/exit |

## Why composers exist

Each composer pre-joins data the agent would otherwise have to ferry between
calls:

- `get_transcript_for_subject` joins `SubjectInstance → Detection → Frame.timestamp_seconds → TranscriptSegment` by time overlap.
- `get_frames_around_transcript` joins `TranscriptSegment` time windows → `Frame` → `Detection` by class.
- `co_presence` joins per-term resolutions (subject / class / open-vocab text) → spatial proximity → temporal dilation.

If a recipe you need isn't in the table above, prefer **one composer call with
clearly-typed inputs** over a chain of primitive searches whose intermediate
data you can't see.

## When composers don't fit

- **Free-text browsing** ("anything interesting around 1:13?") — use the
  primitive searches (`search_captions`, `search_visual_embeddings`,
  `search_transcript`) and let the user steer.
- **No subject registered yet** — composers that take `subject_id` need the
  subject to exist. Register first, or use the open-ended search primitives.
- **Speaker identification** — there is no speaker diarisation. "the person
  who said X" returns the people *visible while X was said*; the user must
  confirm the actual speaker via `request_user_confirmation`.
