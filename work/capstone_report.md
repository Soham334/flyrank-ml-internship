# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Soham Shukla
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/Soham334/flyrank-ml-internship
- **Date:** 10 August 2026

## 0. Abstract

This capstone asks whether an interpretable supervised model can rank content pages by decline risk more usefully
than the existing Week-4 transparent "stale + visible → REFRESH_REVIEW" rule, when both are evaluated the same way
on the same 30,000-row anonymized starter slice (32 pseudonymized clients). A regularized logistic regression,
trained on a leakage-checked, allowlisted feature set, is compared against the rule using grouped 5-fold
cross-validation by client, with all metrics computed on concatenated out-of-fold predictions. The model reaches an
out-of-fold ROC-AUC of 0.665 and average precision of 0.668 against a 0.542 base rate; the baseline rule reaches a
ROC-AUC of 0.499 (chance level) and average precision of 0.542 (equal to the base rate) for this specific label —
a genuine, checked finding, not an error, and it is reported honestly rather than reframed. The output is a ranked
human-review queue with interpretable reason codes, intended as decision support for a content team, not an
automatic refresh trigger.

## 1. Problem framing

The decision this supports is which content pages a human reviewer should look at first when review capacity is
limited. The unit of analysis is one pseudonymized content item. The output is a ranked score plus a reason code
that a reviewer can read and check. A false positive costs review time on a page that didn't need it; a false
negative can leave a genuinely declining page unreviewed — since review time is the scarce resource, ranking
quality near the top of the queue matters more than plain classification accuracy, which is why Precision@K sits
alongside ROC-AUC and average precision throughout this report. ML is useful here because the decision depends on
combining several observed signals (visibility, engagement, freshness, content properties) rather than one
threshold, while the Week-4 rule remains a fully transparent point of comparison.

## 2. Data safety

The analysis uses the anonymized `data/raw/content_refresh_anonymized.csv` starter slice (30,000 rows, 32
pseudonymized clients) that ships with this repo — no external data access was needed. Model features are the
allowlisted `MODEL_NUMERIC_FEATURES` (18 fields) and `MODEL_CATEGORICAL_FEATURES` (8 fields) already defined in
`scripts/ml_utils.py`, the repo's single shared source of truth for what the pipeline is allowed to learn from.
`trend_direction` and `trend_pct` (the label source), `provider_used`/`model_used` (content-generation metadata,
excluded by design), and the pseudonymous `content_id`/`client_id` are never in that allowlist. IDs are used only
for de-duplication and for defining grouped cross-validation folds. No client names, domains, URLs, private
queries, or credentials appear anywhere in this repository's public artifacts.

## 3. Baseline

The baseline is the rule already built and leakage-checked in `work/notebooks/w04_baseline_score.ipynb` (ML-07),
reused here without modification: an item is **stale** when `days_since_last_update >= 180` and **visible** when
`impressions_90d > 0`; both together trigger `REFRESH_REVIEW` (reason code `stale_visible`). It flags 174 of 30,000
rows (0.58%). It is transparent, reproducible from two thresholds, and was already checked against `trend_direction`,
`trend_pct`, and the pseudonymous IDs in Week 4.

One honest observation from evaluating it against this capstone's target: on this starter slice, every row already
has `impressions_90d > 0` by construction (documented in `docs/data-dictionary.md`), so the "visible" half of the
rule never actually excludes anything here — in practice the rule reduces to the staleness threshold alone on this
particular dataset. That's a property of the starter data, not a flaw in the rule design.

## 4. Model / analysis

**Method:** regularized logistic regression (`class_weight="balanced"`, `max_iter=2000`, seed 42), standardized
numeric features, one-hot encoded categorical features. A decision tree or random forest was deliberately not used
— per the repo's `training-honest-models` skill, the smallest defensible model for a yes/no observed label is
logistic regression, and a readable, auditable ranking model fits the Week-1 framing (decision support, not a
black box) better than a marginally stronger opaque one.

**Target:** `is_declining_label = (trend_direction == "down")` — the same definition used throughout this repo.

**Features:** exactly the 18 numeric + 8 categorical fields in `scripts/ml_utils.py`'s `MODEL_NUMERIC_FEATURES` /
`MODEL_CATEGORICAL_FEATURES`. No label-derived fields, no IDs, no product-generated flags.

## 5. Evaluation

**Split:** grouped 5-fold cross-validation by `client_id` (`GroupKFold`, seed 42); every metric below is computed
on concatenated out-of-fold predictions across all 30,000 rows, not an in-sample fit.

A single 80/20 grouped split was tried first and rejected: the 174 baseline-flagged rows are concentrated in only
10 of 32 clients, so a single random grouped split has a real chance of putting zero baseline-flagged clients into
the test fold. That's exactly what happened on the first attempt — the baseline's test-set ROC-AUC came out as a
meaningless 0.5 computed on a fold with zero positive baseline flags. Five-fold grouped CV, evaluated out-of-fold
across every row, is the smallest defensible fix: every client (and therefore every baseline-flagged row) ends up
in exactly one out-of-fold evaluation.

| Metric | Base rate (reference) | Baseline (`stale_visible`) | Model (logistic regression) |
|---|---:|---:|---:|
| ROC-AUC | — | 0.499 | **0.665** |
| Average precision | 0.542 | 0.542 | **0.668** |
| Precision@20 | 0.542 | 0.550 | **0.750** |
| Precision@50 | 0.542 | 0.460 | **0.740** |
| Precision@100 | 0.542 | 0.490 | **0.810** |

Read plainly: the baseline's ROC-AUC (0.499) is indistinguishable from chance, and its average precision (0.542)
sits exactly at the base rate — the stale+visible rule carries no measurable discriminative signal for *this*
label. That is a genuine result, confirmed independently in Section 3 (decline rate among baseline-flagged rows is
47.1%, slightly *below* the 54.2% overall base rate) — not a scoring artifact. The rule was built to answer "is
this page stale and still visible?", not "is this page's traffic currently declining?"; this evaluation is the
first time the two questions have been measured against the same target, and they turn out to be only weakly
related. The model shows a real, if modest, lift over both the baseline and the base rate on every metric.

**Error analysis** (out-of-fold scores, threshold 0.5): 18,857 correct, 6,729 false positives, 4,414 false
negatives, for a recall of 0.729 and precision of 0.638 at that threshold. False negatives are the costlier error
per the Section 1 framing (a declining page slips through unreviewed); false positives cost review time on a page
that turns out fine. 0.5 is a default threshold, not a tuned one — a real deployment would tune it against actual
reviewer capacity (see Limitations).

## 6. Interpretation

Coefficients come from a logistic regression refit on the full 30,000-row population for interpretability and
ranking (performance claims above use only the out-of-fold scores, never this full-data fit). The largest
coefficients: higher recent visibility (`log_impressions_90d`) is associated with *higher* predicted decline
probability; being in the `position_tier_top_3` or `impression_tier_excellent` buckets is associated with *lower*
predicted decline probability; higher `log_clicks_90d` is associated with *lower* decline probability. Read
directionally, this is a plausible, checkable pattern — already-strong pages read as comparatively stable, while
engaged traffic reads as protective — and nothing in the coefficient list approaches the near-perfect,
suspiciously-dominant signature that would suggest leakage. The baseline's lack of signal (Section 5) is itself
treated as a valid, reportable result rather than something to explain away.

## 7. Recommendation

Ranked queue reason codes (only codes the implementation actually produces):

- `MODEL_HIGH_AND_STALE_VISIBLE` — top-20% model score *and* meets the baseline rule (31 rows). Highest-confidence review candidates.
- `MODEL_HIGH` — top-20% model score only (5,969 rows). Worth reviewing even though the simple rule wouldn't have caught it.
- `STALE_VISIBLE` — meets the baseline rule but not in the model's top 20% (143 rows). Kept as an interpretable cross-check given Section 5's finding.
- `REVIEW_IF_NEEDED` — neither signal fires (23,857 rows). Lowest priority.

1. Review `MODEL_HIGH_AND_STALE_VISIBLE` and `MODEL_HIGH` items first.
2. Treat `STALE_VISIBLE`-only items as a separate, interpretable check rather than folding them into the model
   ranking, given the two signals measure related but distinct things.
3. Route model/baseline disagreements to human review — this is decision support, not an automatic action.
4. Re-run the ranking when the data window changes materially; both the rule and the model were built on one
   static snapshot.

Confidence is directional and decision-support only. This does not establish causality and does not claim to
predict Google's ranking algorithm.

## 8. Reproducibility

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace work/notebooks/capstone.ipynb
```

- **Random seed:** `42` (logistic regression, `GroupKFold`, and the seeded precision@K tie-break).
- **Split:** `GroupKFold(n_splits=5)` by pseudonymized `client_id`; all reported metrics are out-of-fold.
- **Dependencies:** `requirements.txt` at the repo root.
- **Expected input:** `data/raw/content_refresh_anonymized.csv` (ships with the repo).
- **Outputs (committed as reproducibility receipts):**
  - `work/outputs/capstone_metrics.json`
  - `work/outputs/capstone_top20.json`
  - `work/figures/model_vs_baseline_average_precision.png`
- The dataset itself is not committed beyond the small anonymized starter slice already tracked in this repo.

## 9. Acknowledgments & data credit

Built on the [FlyRank ML Internship dataset](https://flyrank.ai).

This work is observational and intended for decision support. It does not establish causality, does not identify
clients, domains, or search queries, and does not claim to predict Google's ranking algorithm.

## Limitations & honest framing

- **Starter slice, not the full warehouse.** 30,000 rows / 32 clients is a teaching slice, not FlyRank's full
  pseudonymized warehouse (~79M rows). Results here should not be read as warehouse-scale findings.
- **Baseline and model measure related but different constructs.** The Week-4 rule targets staleness with
  visibility; the model targets an observed short-term decline label. Their weak relationship (Section 5) is a
  finding about the rule's fit to this particular label, not a general verdict on the rule's usefulness for its
  original purpose.
- **Cross-sectional label, not a forecast.** `trend_direction` compares two trailing 30-day windows within the same
  snapshot; it is an observed pattern, not a validated future outcome, and no forward-looking claim is made.
- **Untuned decision threshold.** The 0.5 threshold used for error analysis is a default, not tuned against real
  reviewer capacity or cost asymmetry between false positives and false negatives.
- **Grouped CV, not a held-out future period.** Client-grouped folds test generalization across clients, not across
  time; a genuinely time-aware split was not possible on this single-snapshot slice.
- **No causal claims.** All associations are directional and observational. This work does not claim any feature
  causes decline, does not claim refreshing a page will improve its performance, and does not claim to model or
  predict Google's search ranking algorithm.
