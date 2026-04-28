# Open-vocab prompting (SAM 3.1)

`open_vocab_detect` runs SAM 3.1 against a text prompt. Prompt phrasing materially affects both result quality *and* `prompt_cache` hit rate (cache key includes the prompt's sha256).

## Use canonical noun phrases

Good: `"person carrying red backpack"`, `"yellow taxi"`, `"open laptop on table"`, `"broken window"`
Bad:  `"a person who appears to be carrying some kind of red backpack"`, `"the yellow taxi cab if you can see one"`

Sentences with hedge words, articles, or pleasantries waste tokens, hurt detection precision, and fragment the cache (every variant has a different hash).

## Don't fragment the cache

If you ran `"person with red backpack"` earlier in the session, **reuse that exact phrase** when refining. Don't switch to `"red backpack person"` — same intent, different cache key, full re-inference cost.

## When to refine

After getting results, if precision is too low (many false positives), tighten the noun phrase: `"red backpack"` → `"red school backpack"`. Don't broaden — broadening returns more candidates and increases SAM 3.1 cost.

## When to accept partial matches

If only 2 of 5 returned bboxes look right, that's signal — those 2 are usable for downstream tools (`co_presence`, `register_subject` flow). Don't re-prompt unless you need ≥5.
