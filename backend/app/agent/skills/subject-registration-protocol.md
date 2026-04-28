# Subject registration protocol

A Subject is a session-scoped identity. Bad subjects poison every later query.

## Registration rules

- **Require ≥3 confirmed detections.** The tool will reject fewer. If these are not found among the first set of suggestions, then ask again (keep the count from the first). When you finally call `register_subject`, pass *all* the detection_ids confirmed across every round so far — not just the latest round's picks; the tool sees only what you submit in that single call.
- **Prefer detections from different frames** (not 3 consecutive frames of the same shot — that's not pose/lighting variation, it's near-duplicates).
- **Prefer detections from different videos** when the corpus has more than one. Cross-video confirmations are the strongest signal.
- **Use consistent labels.** "Subject A", "Subject B", "Vehicle 1". Never "the man in red" — colors and clothing change between scenes; the label should not.
- **Re-confirm if appearance varies.** If the proposed reference detections look different enough that you doubt they're the same person, ask the user a follow-up confirmation before calling `register_subject`.

## The reference set is capped at 20

The system uses farthest-point sampling to keep coverage diverse across pose/lighting/angle. You don't manage this — but be aware that confirming 50 detections doesn't mean 50 stored references; it means the 20 most-diverse of those 50.

## When to spawn a sibling subject

If the user rejects detections that cluster tightly together in embedding space (per `apply_user_feedback`'s rejected payload), that's a signal there's a *similar but distinct* person. Propose registering a sibling: "I noticed the rejections all show a person in a similar jacket but different build — should I register them as Subject B?"

## What `register_subject` actually needs

`register_subject` takes **Detection UUIDs**, not frame_ids or shot_ids. Detections carry the box-crop embedding (`box_embedding`) used as the subject reference; whole frames do not.

- **Sources of valid detection_ids:** `get_object_detections`, `open_vocab_detect`, `search_instances_by_subject`, and `search_visual_embeddings` when it returns per-detection items.
- **NOT valid:** `frame_id` from `caption_frames`/`search_captions`/`search_transcript`, or `shot_id` from `temporal_cluster`.

When proposing items via `request_user_confirmation` with `mode='instances'`, set each item's `id` field to the **detection_id** explicitly — the user's confirmation echoes that exact `id` back to you, and you forward it to `register_subject`. If you only have frame timestamps from caption-based hits, first call `get_object_detections` (filter `class_name='person'` for people) on those frames to get the underlying detection_ids before proposing them for confirmation.

## When to ask for more confirmations vs. when to stop

Iterating to *improve* a registration is fine — it's how you build a high-quality subject. Looping after a *failure* without changing anything is not.

**Iterate (good)** when:

- The user confirmed only 1–2 items and you need ≥3 with pose/lighting variety.
- The confirmed items cluster too tightly (same shot, same angle); ask for a second batch from a different time window.
- The user rejected several items that look similar to each other — propose a sibling subject (see above) instead of just retrying the same set.

**Stop and ask the user (or change strategy)** when:

- `register_subject` returns the **same error code twice in a row** with detection IDs from the same source. The orchestrator will block a third identical attempt — don't let it get there.
- You've already proposed two confirmation rounds for the same subject and the user has confirmed enough valid detections — call `register_subject` and move on; don't keep asking for more.
- The user's most recent message is *not* about registration (e.g. "what did this person say?") — finish the registration in the background only if the new task strictly requires it; otherwise set it aside and answer what was just asked.

Each retry of `register_subject` should change at least one of: the detection IDs (different source tool, different time window, different frames), or the strategy (e.g. switch from caption-based to detection-based candidates). If you can't change anything, stop and tell the user why.
