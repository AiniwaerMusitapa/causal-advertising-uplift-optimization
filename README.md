# Causal Advertising Uplift Experiment & Targeting Optimization

Completed causal uplift study using the corrected, unbiased Criteo Uplift v2.1 randomized advertising dataset.

## Verified scope

- Official Criteo organization mirror archive: 311,422,618 bytes; SHA256 `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`.
- Full-data audit and ATE: 13,979,592 rows, 12 anonymized baseline features, treatment, visit, conversion and post-treatment exposure.
- Model sample: deterministic 1-in-7 hash sample, 1,397,678 train / 299,325 validation / 299,251 test rows.
- Models actually trained: LightGBM S-Learner, T-Learner, X-Learner, response model; EconML CausalForestDML on a deterministic 250,000-row training subset.
- Every baseline model feature is one of `f0`–`f11`. `exposure` is excluded because it is post-treatment.

## Key results

- Treatment/control rows: 11,889,655 / 2,089,937; treatment ratio 85.0501%.
- Visit ATE: 1.0342 percentage points; 95% bootstrap CI [1.0060, 1.0623] percentage points.
- Conversion ATE: 0.1152 percentage points; 95% bootstrap CI [0.1083, 0.1222] percentage points.
- Baseline-feature treatment prediction validation AUC: 0.510361; maximum full-data absolute SMD: 0.048836.
- Validation Qini selected S-Learner. Test Qini / AUUC / visit uplift@10%: 0.00468973 / 0.00963994 / 9.4848%.
- Response-model test Qini / AUUC / visit uplift@10%: 0.00431221 / 0.00926242 / 8.2700%.

The uplift metrics use a propensity-adjusted transformed outcome. Mathematical definitions are recorded in `reports/uplift_evaluation.md`. The dataset publisher non-uniformly subsampled the data, so the measured sample ATE is not presented as original-population campaign lift. No revenue/cost fields exist and no ROI is claimed.

## Reproduce

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\00_download_data.py
.\.venv\Scripts\python.exe scripts\01_prepare_and_audit.py
.\.venv\Scripts\python.exe scripts\02_train_and_evaluate.py
.\.venv\Scripts\python.exe scripts\03_finalize_and_verify.py
```

Every stage preserves machine-readable reports. `experiment_manifest.json` records the data version, hashes, split/sample rules, features, seed, selected model and model artifacts.

## Reports and figures

- `reports/experiment_validation.md`: sample ratio, outcomes, SMD, distribution checks and treatment predictability.
- `reports/ate_analysis.md`: full-data visit/conversion ATE, standard errors and normal/bootstrap confidence intervals.
- `reports/uplift_evaluation.md`: Qini, AUUC, uplift@K, deciles, model comparison and definitions.
- `reports/targeting_policy.md`: response-vs-causal policies for visit and conversion.
- `reports/artifact_audit.json`: model/source integrity and environment versions.
- `reports/powerbi/`: verified CSV inputs; no PBIX artifact is claimed.
- `figures/`: balance, outcomes, ATE, Qini, uplift, decile, targeting and model-comparison plots.

Source: Criteo AI Lab, “Criteo Uplift Prediction Dataset,” and the official `criteo/criteo-uplift` mirror. The corrected 13.98M-row v2.1 release is used instead of the older approximately 25M-row release for which Criteo reports a leakage issue.
