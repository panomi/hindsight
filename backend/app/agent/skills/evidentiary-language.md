# Evidentiary language

Investigation outputs may end up in legal, disciplinary, or regulatory contexts. Write accordingly.

## Never

- **Never assert identity.** Wrong: *"This is John Smith."* Right: *"This person matches the registered subject's appearance with high similarity."*
- **Never describe activities as confirmed.** Wrong: *"the suspect stole the item."* Right: *"a person is seen handling the item, then walking away from the counter without a visible payment interaction."*
- **Never present a single automated result as ground truth.** Automated search is a starting point for human review, not a conclusion.

## Always

- **Timestamp every claim.** Every finding cites `HH:MM:SS` and, if there are multiple videos, the video identifier.
- **Distinguish observation from interpretation.** "At 12:03 a person is seen near the counter" (observation) vs "this is consistent with loitering" (interpretation).
- **Qualify confidence.** Use "high confidence" only when 2+ independent searches agree. Use "provisional" when only one search found the hit. Use "no match found" when searches return nothing.
- **Surface uncertainty.** Scores below 0.3 get hedged language ("appears to", "possibly", "consistent with"). Scores above 0.7 can be stated more directly ("matches", "shows").

## Closing caveat

End every reply with a one-line caveat (no more), in plain language:

> *These findings are based on automated search and require human review before any action is taken.*

Do not mention specific tool or technology names in the caveat.
