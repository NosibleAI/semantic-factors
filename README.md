# semantic-factors

Build auditable quantitative factors from NOSIBLE World by describing events in plain English.

```python
from semantic_factors import SemanticFactor

factor = SemanticFactor(
    name="crop_infestation",
    anchors=[
        "Locust swarms destroy crops and threaten regional food supplies.",
        "Agricultural pest outbreaks cause severe losses in cultivated fields.",
    ],
    pole_hi=["The infestation is spreading and becoming more severe."],
    pole_lo=["The infestation is contained and crop conditions are recovering."],
    filters={"language": "en"},
    floor="p99.9",
)

daily = factor.compute(
    weights="netlocs",
    start_from="2015-01-01",
    stop_at="2026-06-30",
)
factor.plot(path="crop_infestation.png")
```

The package is deliberately small. NOSIBLE World retrieves date-local semantic candidates and
returns a lean event projection plus exact eligible-universe totals. This library performs the
factor arithmetic, diagnostics, time-series transformation, plotting, and exposure analysis.

## Install

```bash
python -m pip install semantic-factors
```

Plotting is optional:

```bash
python -m pip install "semantic-factors[plot]"
```

Python 3.10 or newer is required.

## Credentials

A build uses two independent credentials:

```bash
export NOSIBLE_API_KEY=nos_sk_...
export OPENAI_API_KEY=sk-...
```

- `NOSIBLE_API_KEY` authorizes the date-grouped NOSIBLE World archive requests.
- `OPENAI_API_KEY` embeds only your anchor and pole text with `text-embedding-3-large`. No
  NOSIBLE event text is sent to the embedding provider.

Pass `api_key=` and `embed_api_key=` instead if environment variables are inconvenient. An
OpenAI-compatible embedding endpoint can be configured through the lower-level embedding API.

## Define a factor

`SemanticFactor` has three semantic inputs:

- `anchors` describe which events are relevant. Alternative phrasings normally use
  `aggregate="max"`; use `mean` when every anchor is a facet that should be present together.
  `mean` and `softmax` require an explicit cosine floor because a per-anchor candidate union
  cannot reliably calibrate their aggregate quantiles.
- `poles` describe direction. Each `(high, low)` pair should express the same topic in opposite
  states. A factor spanning several topics should use one matched pair per topic.
- `filters` define the eligible universe using NOSIBLE World fields and operators.

An unsigned factor needs only anchors:

```python
factor = SemanticFactor(
    anchors=["Workers begin a strike that disrupts production or transport."],
    filters={"country": "United States", "language": "en"},
    floor=0.35,
)
```

Several pole pairs are explicit:

```python
factor = SemanticFactor(
    anchors=["Trade policy creates material uncertainty for companies."],
    poles=[
        (["Tariffs are threatened or expanded."], ["Tariffs are withdrawn or suspended."]),
        (["Trade talks break down."], ["A binding trade agreement is signed."]),
    ],
)
```

## The arithmetic

For each eligible event, World returns exact cosines against every anchor retained by the
date-local search. The package aggregates those cosines into relevance `r`, keeps events where
`r >= floor`, and applies the selected event weight `w`.

When poles are present, the event uses the matched pair that best describes it. Its high-minus-low
cosine gap is mapped through `tanh(gap / tanh_scale)` to polarity `p`. With the default
`polarity_weighted=True`, the numerator contribution is:

```text
w * (1 + p) / 2
```

The daily factor is the sum of those contributions divided by the matching exact denominator
over every eligible event that day. `stabilize=True` replaces the same-day denominator with its
trailing 365-day mean, preserving genuine event spikes while controlling for corpus growth.

## Weighting schemes

| `weights=` | Matching-event numerator | Eligible-universe denominator |
|---|---|---|
| `equal` | 1 per event | event count |
| `coverage` | published article count | total article count |
| `netlocs` | distinct publisher count | total distinct-publisher count |
| `materiality` | materiality score | total materiality score |
| `sentiment` | negative 1, neutral 0.5, positive 0 | the same sentiment mass |
| `relevance` | relevance cosine | event count |

Custom callable weights are not accepted by the World retrieval path: the client could weight
matched rows, but it could not produce the corresponding exact denominator without exporting the
whole eligible corpus. The package fails before making a charged request instead of returning a
plausible but inconsistent series.

## Floors and calibration

An absolute cosine is reproducible and cheap:

```python
SemanticFactor(anchors=[...], floor=0.35)
```

A high quantile is easier to reason about across definitions:

```python
SemanticFactor(anchors=[...], floor="p99.9")
```

Quantile floors use `aggregate="max"` and are estimated once from the approximate high-score
tails of 24 evenly spaced, unfiltered archive dates, then held fixed across the requested window.
They are never recalculated per day. Use an explicit cosine floor with `mean` or `softmax`.
Inspect the available high quantiles before a build:

```python
print(factor.calibrate(start_from="2015-01-01", stop_at="2026-06-30"))
```

Calibration and factor chunks are content-addressed on disk, so an interrupted or repeated run
continues from cached immutable responses. Set `SEMANTIC_FACTORS_CACHE` to relocate the cache.

## Filters

Mappings use implicit `and`:

```python
filters={
    "language": "en",
    "country": {"in": ["United States", "Canada"]},
    "materiality_score": {"gte": 0.6},
}
```

Logical composition is explicit when needed:

```python
filters={
    "and": [
        {"language": "en"},
        {"or": [{"entity_org": "NVIDIA"}, {"ticker": "NVDA.US"}]},
    ]
}
```

Unknown fields and invalid operators are rejected locally where possible, and indexed World
filters are pushed into HNSW traversal. Run `semantic-factors fields` to inspect the vocabulary.

## Results and diagnostics

`compute()` returns a Polars frame and stores it as `factor.daily`:

| Column | Meaning |
|---|---|
| `date` | archive date |
| `value` | stabilised factor level |
| `polarity` | base-weighted mean pole direction |
| `n_events` | matching events |
| `n_eligible` | eligible events before semantic matching |
| `raw` | numerator contribution |
| `volume` | exact eligible denominator mass |
| `baseline` | trailing denominator mean |
| `share` | `raw / volume` |

The heaviest retained events are available in `factor.events`. `factor.diagnostics` records the
resolved floor, requested and effective archive window, query budget, request and candidate
counts, filters, weighting, and elapsed time.

Convenience methods include:

```python
factor.weekly()
factor.monthly()
factor.to_csv("factor.csv")
factor.to_parquet("factor.parquet")
factor.regress(tickers=["CTVA.US", "FMC.US"])
factor.longs
factor.shorts
```

## Retrieval accuracy and cost

World search uses approximate date-local HNSW retrieval followed by exact reranking. It is not an
exhaustive scan. The default retains 1,000 candidates per anchor per date; pre-release broad- and
narrow-factor measurements recovered about 99% of exact above-floor matches at that budget on the
sampled dates. That measurement is evidence for the default, not a guarantee for every factor.

Increase `per_date_limit` (up to 10,000) for unusually broad or high-stakes definitions and inspect
the sensitivity of the resulting series. The chosen budget is always recorded in diagnostics.

A full 11.5-year build uses roughly 35 search requests, plus 24 cached calibration requests when a
quantile floor is used. The client honors `Retry-After`, retries bounded transient failures, and
stores each 120-day response independently.

## Explicit limitations

- `residualize=True` is not supported by the public grouped-search contract in this release and
  raises `ConfigurationError`. It is never silently skipped.
- Retrieval is approximate, as described above.
- Factor values are research signals, not investment advice. Exposure statistics do not establish
  causality and should be validated out of sample.
- NOSIBLE World data access is governed separately from this source package.

## Command line

```bash
semantic-factors compute --spec factor.json --out ./output
semantic-factors fields --search materiality
semantic-factors schemes
```

The compute command writes daily and monthly CSV files, a diagnostic JSON record, retained events,
and (with the plot extra) a PNG chart.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy src
python -m build
```

See `examples/` for complete trade-policy, labour-action, geopolitical-risk, and crop-infestation
definitions. Contribution guidance is in [`CONTRIBUTING.md`](CONTRIBUTING.md); security reports
should follow [`SECURITY.md`](SECURITY.md).

## License

The package metadata currently declares Apache-2.0. The root `LICENSE` file will be added after
maintainer confirmation before the repository becomes public. NOSIBLE World data is licensed
separately. The remaining public-release gates are tracked in
[`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md).
