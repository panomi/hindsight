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
| **"What does the sign / board / screen / paper / plate say?"** — read text on a known surface | `open_vocab_detect("schedule board" / "license plate" / "sign" ...)` to locate it → `ask_vision(frame_ids=[...], question="...", bbox_xyxy=<from step 1>)` to read it |
| **"What time / number / value is shown?"** — specific data field | `ask_vision(frame_ids=[chosen frame], question="<focused question>", bbox_xyxy=<region if known>)` |
| **OCR found the entity but missed the data field** (e.g. "CHICAGO" but no time) | DO NOT retry `search_ocr` with rephrasings. `ask_vision` on the same frame, ideally with a bbox for the surface |
| **Stored caption from `caption_frames` is too generic for the question** | `ask_vision(frame_ids=[same frames], question="<the specific thing>")`. Re-calling `caption_frames` returns the same paragraph; the prompt is fixed at ingest |

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

## Reading pixels: which tool when

The "read pixels" surface has three distinct primitives — picking the wrong
one is the most common failure mode in this codebase:

- **`search_ocr`** — retrieval over text recognised at ingest. Fast, free,
  but: (a) only runs on frames flagged as text-carriers, (b) struggles with
  small / stylised / tabular text, (c) word-level — no semantic understanding
  of what the text *means*. Use for "find the frame that has the word X".
- **`caption_frames`** — returns the stored 50-word ingest paragraph.
  Generic scene description, fixed prompt. Use for "what's roughly happening
  in this frame?". CANNOT answer custom questions.
- **`ask_vision`** — runs the VLM with a free-form question on specific
  frames, optionally cropped to a bbox. Metered (~0.5–2s per frame). Use for
  "what does the sign actually say?", "what time is shown?", "is the door
  open?" — anything OCR missed and stored captions don't carry. Pair with
  `open_vocab_detect`'s bbox when the answer is in a small region of a busy
  scene.

**Escalation rule.** If `search_ocr` or `caption_frames` returned something
*on the right frame* but not *the data field the user asked for*, the next
call is `ask_vision` — never another OCR/caption retry with the same frames.
