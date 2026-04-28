# Spatial reasoning

`co_presence`, `open_vocab_detect`, and any tool with proximity / bbox parameters need consistent defaults so results are reproducible. **Do not pick the schema's default just because it is the default** — pick per intent.

## Per-intent proximity (normalized coords, 0–1)

| Intent | metric | proximity | Notes |
|---|---|---|---|
| handoff / contact / handing object | `iou` | ≥ 0.10 | bboxes must actually overlap |
| next to / shoulder-to-shoulder | `center` | 0.03–0.05 | very tight |
| near / nearby | `center` | 0.05–0.08 | default if user just says "near" |
| in the same area / same room | `center` | 0.15–0.25 | only when scene is intentionally wide |
| anywhere in frame | omit proximity entirely | — | use `co_presence` without proximity, or just intersect frame ids |

## Pick the metric to fit the intent

- **IoU** captures contact/overlap. Use for handoff, struggle, holding the same object.
- **Center distance** captures rough closeness. Use for "next to", "near", "approaching".
- **Containment** (one bbox inside another) is rare in practice — skip unless user is explicit.

## Bounded prompts to `open_vocab_detect`

When resolving a semantic term, use a canonical noun phrase ("person carrying red backpack"), not a full sentence — this improves SAM 3.1 precision and `prompt_cache` hit rate.
