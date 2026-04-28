# Temporal reasoning

Temporal language has consistent defaults so investigations are reproducible across sessions.

## Default windows

| Phrase | Window |
|---|---|
| immediately / right after | 0–10 s |
| shortly after | 0–120 s |
| around the same time | ±30 s |
| before | preceding 60 s (unless qualified) |
| while | overlapping intervals (>0 s overlap) |
| during the same scene | within the same shot (use shot_id) |

## Combining ASR with frames

Audio segments and frames have independent timestamps but the same clock per video. When asked about something heard *and* seen, intersect:
- transcript segment `[t_a_start, t_a_end]`
- frame timestamps `t_f` such that `t_a_start - 5 ≤ t_f ≤ t_a_end + 5` (small dilation accounts for setup before / lingering visual after the speech).

## Cross-camera clocks

Different videos may not share a synchronized wall clock — only treat times as comparable across videos if the user has stated they are synced (or both videos came from the same recording session). When unsure, flag it: *"these timestamps are video-relative; cross-video timing depends on synchronization which I can't verify."*
