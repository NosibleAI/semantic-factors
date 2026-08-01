# Changelog

## 0.1.0 — unreleased

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
