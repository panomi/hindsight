# ask_vision protocol — when to use it, how to phrase it

`ask_vision` runs the vision-language model on specific frames with a free-form
question, returning a *short factual answer* per frame. It is the right tool
when stored data (OCR rows, ingest captions) covered *something* on the frame
but not the data point the user actually asked for.

## When to reach for it

- **OCR found the entity but not the data field.** OCR returned "CHICAGO" on a
  schedule board but didn't extract the time column → `ask_vision(frame_id,
  "what time does the flight to Chicago depart?")`. Pass the bbox of the board
  if you have it (from `open_vocab_detect`) — cropping a tabular region
  dramatically improves the read.
- **License plate / sign / screen / paper read.** OCR is ingest-time and gated
  by carrier flags; if it missed text on a known text-bearing surface, do not
  retry OCR — escalate to `ask_vision` with a precise question and a bbox.
- **Yes/no or count question about the scene.** "Is the door open?", "How
  many people are at the table?", "Is the sign in English or French?" — these
  are answers stored captions don't carry.
- **Stored caption is too generic.** When `caption_frames` returned a vague
  paragraph that doesn't address the user's specific question.

## When NOT to use it

- **Don't use as a general scene describer** — `caption_frames` already returns
  the stored 50-word ingest caption for free. `ask_vision` is metered VLM
  inference; reach for it only when you need a question-specific answer.
- **Don't use to localise objects** — that's `open_vocab_detect` (returns
  bboxes). The VLM doesn't return coordinates.
- **Don't use to identify a person** — VLM can't recognise who someone is.
  Use the subject pipeline (`register_subject` → `search_instances_by_subject`)
  instead.
- **Don't ask multi-frame reasoning questions in one call** — each frame is
  answered independently. "Did the man enter or exit the car?" needs frame
  ordering: ask per frame, then reason.

## Phrasing the question

- **Be concrete and specific.** "What time does the flight to Chicago depart
  (per the schedule board)?" beats "what's on the board?".
- **Tell it to read literally.** For text-on-a-surface, say "as written".
  Example: "Read the license plate as written. Reply 'unknown' if not legible."
- **Anchor with a region word when no bbox.** "On the departure schedule, what
  time is next to Chicago?" — the model will look at the right region even
  without a crop.
- **Bound the answer shape.** "Answer with just the time, e.g. '10:15'." cuts
  10× the tokens off a verbose answer.

## Cropping with `bbox_xyxy`

- The bbox is normalised `(0..1)` and matches the dict shape returned by
  `open_vocab_detect.bbox` and any `Detection.bbox`. Pass it through directly
  — no reformatting.
- The tool expands by ~5% margin internally so context isn't cut.
- Crop is **per-call**, not per-frame. If you have different bboxes per frame,
  make separate calls (or omit the bbox and rely on the question to anchor
  attention).
- A crop helps most when the answer is in a small region of a busy scene —
  schedule boards, license plates, menu items. It can hurt when the answer
  needs full-scene context — leave bbox out for "is the room crowded?".

## Cost discipline

- Hard cap **6 frames per call**. The orchestrator will reject more.
- Results are cached by `(question + bbox, frame_id)` — re-asking the exact
  same question on the same frame is free. Slightly varying the question
  invalidates the cache; pin a canonical phrasing if you expect to retry.
- If you've called `ask_vision` once and got "unknown", DO NOT retry with a
  rephrased question hoping for different output. Either:
  1. Crop tighter with a known bbox, or
  2. Switch to a *different* frame from the same shot, or
  3. Stop and tell the user the data isn't legible.

## Composition with other tools

| You have | Then call | With | To answer |
|---|---|---|---|
| OCR hit on entity, no data field | `ask_vision` | bbox from `open_vocab_detect` if available | Read the data field |
| Frame from `search_captions` is a text-carrier scene | `open_vocab_detect("schedule board"/"sign"/...)` → `ask_vision` | bbox from step 1 | Read the surface |
| `caption_frames` returned generic | `ask_vision` | none (or a region noun in the question) | Pin a specific detail |
| Detection of a sign / plate | `ask_vision` | the detection's bbox | Read it literally |
