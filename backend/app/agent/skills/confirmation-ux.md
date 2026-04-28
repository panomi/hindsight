# Confirmation UX

The user is time-pressured. Bad confirmation UX cascades into bad subjects and bad results.

## Phrasing

Wrong: *"Are these the right people?"* (vague, no action)
Right: *"Subject A appears in 8 candidate tracks. Confirm matches and reject any false positives — I'll then expand to all detections sharing those tracks."*

A good question states (a) what's shown, (b) what action you want, (c) what happens next.

## Modes

- `mode="instances"` — proposing tracks for `register_subject` or `apply_user_feedback`. UI shows per-track image strips so the user can scan motion/pose variety.
- `mode="frames"` — when retrieval is ambiguous and you want the user to pick relevant frames.
- `mode="events"` — confirming temporal groupings before presenting them as findings.

## Batch size

8–12 items per confirmation. 30 items overwhelms; 3 items is wasteful (just present them as findings). Split larger sets across multiple confirmations only when you genuinely need each batch's outcome to decide the next batch.
