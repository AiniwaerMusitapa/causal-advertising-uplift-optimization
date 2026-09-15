# Randomized experiment validation

Dataset: corrected Criteo Uplift v2.1. Local rows: 13,979,592; columns: 17.

Treatment: 11,882,655 (85.000013%); control: 2,096,937. Visit rate: 4.699200%; conversion rate: 0.291668%.

Maximum absolute SMD across 12 baseline features: 0.048836; mean absolute SMD: 0.020695.

| Feature | Control mean | Treatment mean | SMD | KS statistic |
| --- | ---: | ---: | ---: | ---: |
| f0 | 19.651705 | 19.614755 | -0.006866 | 0.013624 |
| f1 | 10.067935 | 10.070337 | 0.023994 | 0.001570 |
| f2 | 8.448173 | 8.446302 | -0.006241 | 0.007431 |
| f3 | 4.232821 | 4.169412 | -0.048836 | 0.004618 |
| f4 | 10.336526 | 10.339245 | 0.007966 | 0.002471 |
| f5 | 4.039339 | 4.026602 | -0.030557 | 0.004530 |
| f6 | -3.999880 | -4.182792 | -0.040448 | 0.009560 |
| f7 | 5.080284 | 5.105555 | 0.021270 | 0.004574 |
| f8 | 3.934652 | 3.933392 | -0.022433 | 0.007489 |
| f9 | 15.886253 | 16.052589 | 0.024001 | 0.006522 |
| f10 | 5.331898 | 5.333660 | 0.010563 | 0.001889 |
| f11 | -0.170868 | -0.170985 | -0.005160 | 0.002374 |

Treatment assignment is imbalanced in sample size; covariate balance is assessed by SMD rather than equal group counts. Assignment predictability is added by the modeling pipeline.

**Post-treatment safeguard:** `exposure` is observed after treatment assignment and is excluded from preprocessing, response models and every CATE estimator. Including it would condition on a treatment consequence and can bias causal effects.

Split: `hash(row_id,42)%100: train 0-69, validation 70-84, test 85-99`; counts: {'test': 2096507, 'train': 9788408, 'validation': 2094677}. Seed 42.

Assignment predictability validation ROC-AUC (baseline features only): **0.510361**. This diagnostic complements, but does not replace, balance and design evidence.
