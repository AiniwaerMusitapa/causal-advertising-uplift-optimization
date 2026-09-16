# Methodology and metric interpretation

This page describes the implementation behind the committed offline results. It does not establish the validity of the publisher's randomization or independently replicate the raw-data analysis.

## Experimental question

The estimand is the effect of assigned treatment on site visits in the released sample. The full-data ATE is the treated visit mean minus the control visit mean. Assignment, not realized exposure, defines the treatment. The feature set is `f0`–`f11`; post-treatment `exposure` is excluded.

The publisher's non-uniform subsampling prevents direct interpretation as the original campaign population effect. Features are anonymized, so customer personas and substantive explanations of specific features cannot be inferred from this dataset.

## Development and evaluation

The deterministic model sample uses `hash(row_id,2026)%7=0`. Train, validation and test membership uses `hash(row_id,42)%100` with cutoffs at 70 and 85. These are hash-based partitions, not temporal holdouts, because the release has no event timestamps.

S-, T- and X-Learners use LightGBM. CausalForestDML uses a deterministic 250,000-row training subset, rather than the entire training partition. The response baseline estimates visit probability without treatment as a feature. Validation Qini selects the causal ranker; the test partition is reserved for reporting frozen policies. A full-data ATE audit is separate from predictive-model development.

## Uplift, gain, AUUC and Qini

Let `e` be the observed treatment fraction of the evaluation partition. The transformed outcome is:

```text
Z = Y * (T/e - (1-T)/(1-e))
```

Rows are sorted by descending model score with stable tie ordering. For each targeting fraction `q`:

```text
k = ceil(q * n)
uplift(q) = mean(Z in the first k ranked rows)
gain(q) = q * uplift(q)
AUUC = integral of gain(q) over q
Qini = integral of [gain(q) - q * mean(Z)] over q
```

The implementation uses a 100-point grid from 0.01 to 1.00, adds the origin, and applies trapezoidal integration. Qini here is an area coefficient, not a normalized Qini score. Comparing values from other libraries requires matching their definitions and scaling.

The recorded gain uses the requested fraction `q`, not exactly `k/n`. Consequently, displayed estimated incremental outcomes `n * gain(q)` can differ slightly from `k * uplift(q)` due to rounding. Top-10% uplift is the propensity-adjusted transformed-outcome mean, not necessarily the raw treated-minus-control mean within that segment. A global assignment proportion is used rather than fitting a personalized propensity model.

## Two different bootstrap procedures

The full-sample ATE interval uses 2,000 arm-stratified binary-outcome resamples. The decile intervals use 300 multinomial resamples of the empirical transformed-outcome values within each ranked decile, keeping the evaluation-partition propensity fixed. The latter is not an arm-stratified bootstrap, despite the historical training script's generated description using that term.

Decile intervals condition on the fitted models, partition and ranking. They do not capture retraining, model-selection or publisher-sampling uncertainty. They are not confidence intervals for the difference between causal and response targeting. The observed +1.2148 percentage-point policy difference is therefore not presented as statistically significant.

## What the lightweight checks establish

`python -m unittest discover -s tests -v` checks recorded curve arithmetic, Qini/AUUC integration, population counts, decile reconciliation, model-summary CSV consistency, source checksums and local links. `python scripts/05_snapshot_summary.py` reads the same saved results without data or ML dependencies.

These checks detect internal inconsistencies in the snapshot. They do not recreate rankings from raw data or verify counterfactual individual outcomes. Full replication requires the download, audit, training and finalization stages documented in the README. Production use additionally requires cost/value inputs and prospective randomized validation.
