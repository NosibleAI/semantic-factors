# Changelog

## 0.1.3 — 2026-08-06

- Keep annotation guide wires and markers inside the visible chart rectangle, including when a
  raw daily spike sits above the trend-based axis ceiling or an episode date is out of range.
- Require automatic annotations to carry an event headline instead of rendering a meaningless
  date-only label.
- Shorten automatic headlines to keep the two annotation rows readable.

## 0.1.2 — 2026-08-05

- Correct the public release source to the reviewed package used by the NOSIBLE website build.
- Promote the release-fix client, cache, filter, embedding-provider, return-data, and reliability
  changes from the authoritative monorepo branch.
- Make `plot()` match the website recipe out of the box: tanh display squash, a 30-day geometric
  mean, and its 7-day rolling average as the single default trend.
- Resolve OpenRouter endpoints and provider model names automatically, while retaining explicit
  `embed_base_url` control.
- Reject absolute cosine floors below 0.30 and expand API boundary tests.

## 0.1.1 — 2026-08-05

- Add the documented `embed_base_url` argument to `compute()`, `calibrate()`, and CLI specs so
  OpenRouter and other OpenAI-compatible embedding providers work end to end.

## 0.1.0 — 2026-08-05

First release.

- `SemanticFactor`: define a factor with anchors, poles and filters; build a daily series with
  `compute()`; chart it with `plot()`; measure exposures with `regress()`.
- One authenticated data path over `POST /world/v1/search`: date-grouped multi-vector retrieval,
  exact candidate reranking, lean factor fields, and exact eligible-universe denominator totals.
- Quantile floors. With max aggregation, `floor="p99.9"` is resolved once against evenly spaced,
  unfiltered archive samples and then held fixed. `calibrate()` reports the resolvable
  high-score tail before you commit to one; mean and softmax require an explicit cosine floor.
- Filter vocabulary covering all 74 filterable World fields, accepting nested names
  (`event.region`), documented aliases (`n_sources`), and the operator DSL. Unknown fields are
  rejected locally with a suggestion.
- Poles as matched `(high, low)` pairs, one per topic, with per-event best-pair selection. Each
  event's weight is multiplied by `(1 + polarity) / 2`, so the poles move the indicator itself
  rather than producing a second series beside it. `polarity_weighted=False` opts out.
- Six weighting schemes with matching exact denominators. Unsupported callable weights fail
  before making a charged request.
- Corpus-growth stabilisation against a trailing 365-day baseline.
- Contemporaneous exposures by default, with Newey-West standard errors when a forward horizon
  makes the observations overlap, and point-in-time rolling loadings.
- Content-addressed caching of definition embeddings and immutable World search chunks.
