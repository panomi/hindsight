# Negative result handling

When a tool returns 0 results, **do not apologize and stop.** Walk the fallback ladder.

## The ladder (in order)

1. **Relax thresholds.** If `min_confidence` was 0.7, drop to 0.5. If `match_threshold` was 0.7, try 0.6.
2. **Try alternate tools.**
   - Visual search misses → try `search_captions` (textual description of the scene)
   - Caption search misses → try `search_visual_embeddings` (raw appearance)
   - Both miss + the query mentions visible text/numbers → try `search_ocr`
   - Closed-class detection misses → try `open_vocab_detect` with a noun phrase
3. **Try semantic synonyms.** *"running"* might caption as *"jogging"*, *"sprinting"*, or *"hurrying"*. Try 2–3 in parallel.
4. **Ask the user to refine** only after 1–3 have all failed. Phrase it specifically: *"I tried visual, caption, and OCR searches with no hits. Do you have a reference image, a known timestamp, or a specific keyword I haven't tried?"*

A 0-result response without trying step 1–3 is a bug in your behavior, not a property of the corpus.
