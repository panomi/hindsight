# Re-ranking protocol

`apply_user_feedback` translates the user's confirm/reject signals into changes to a Subject's reference set. How you interpret those signals matters.

## Confirm signals

A confirmation expands the reference set. The system then prunes back to N=20 via farthest-point sampling, so adding 5 confirmations doesn't grow the set by 5 — it adds 5 candidates and re-selects the most diverse 20 from the new pool. This is a feature: it lets users keep confirming without diluting the set with near-duplicates.

After a confirm round, **re-run `search_instances_by_subject`** — the expanded reference set will catch matches the previous one missed, and the `subject_instances` table will pick up new (video, instance_id) pairs.

## Reject signals

Rejections never modify the reference embeddings. They only:
- record which detections were wrong
- surface clustering hints (does the rejected set cluster tightly in embedding space?)

## When to spawn a sibling subject

If `apply_user_feedback`'s `rejected` payload reports many rejections from a small number of `(video_id, instance_id)` tuples — i.e. the same person was repeatedly mistaken for the subject — propose a sibling: *"The 6 rejections all came from track 7 in video 2. That's likely a different person who looks similar to Subject A. Should I register them as Subject B so we can track both separately?"*

Don't try to "fix" Subject A by adding the rejected detections as negatives — the system stores positive references only. Spawn a sibling instead.

## After feedback

Always re-issue the originating retrieval (`search_instances_by_subject`, `co_presence`, etc.) and present the *delta* from the previous result set, not the full new set: *"Subject A now appears in 12 events, +4 from before."*
