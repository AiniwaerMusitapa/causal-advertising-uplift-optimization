# Uplift evaluation

Outcome: visit. Model sample: 1,996,254 rows from 13,979,592 total rows. Train/validation/test: {'train': 1397678, 'validation': 299325, 'test': 299251}.

Treatment assignment prediction validation AUC: 0.510361. Treatment proportion in model train: 85.003985%.

Validation Qini selected `s_learner` before test reporting.

| Ranker | Test Qini | Test AUUC | Uplift@10% | Uplift@20% | Uplift@30% |
| --- | ---: | ---: | ---: | ---: | ---: |
| s_learner | 0.00468973 | 0.00963994 | 9.484796% | 4.651524% | 3.277426% |
| t_learner | 0.00370830 | 0.00865851 | 7.611378% | 4.159797% | 2.986094% |
| x_learner | 0.00454095 | 0.00949116 | 9.014649% | 4.719832% | 3.168520% |
| causal_forest | 0.00440022 | 0.00935043 | 8.527189% | 4.728026% | 3.201156% |
| response_model | 0.00431221 | 0.00926242 | 8.270001% | 4.617093% | 3.171296% |

## Mathematical definition

Sort descending score. With test propensity e, transformed outcome Z=Y[T/e-(1-T)/(1-e)]. uplift(q)=mean(Z|top q); gain(q)=q*uplift(q); AUUC=integral gain dq; Qini=integral [gain-q*mean(Z)] dq.

The response baseline ranks by P(visit|X); causal models rank by estimated E[Y(1)-Y(0)|X]. High response propensity is therefore tested rather than assumed to equal incremental value.

## Treatment imbalance and safeguards

The treated arm is much larger than control. T-learner has fewer control examples; X-learner combines arm-specific imputed-effect models using the observed training propensity to reflect unequal precision.
`exposure` is excluded because it is post-treatment. All splits and model samples use fixed hashes. Preprocessing for Causal Forest is fit on its training subset only. Test metrics were computed after every estimator was frozen.

## Stability and uncertainty

Each test uplift decile includes a 95% stratified binary bootstrap interval (300 replicates). Full decile results, pairwise Spearman correlations, top-10% Jaccard overlap, and segment sizes are in `uplift_metrics.json`.

## Limitations

- Dataset is non-uniformly subsampled by publisher, so sample ATE is not the original campaign population lift.
- No monetary values/costs; no ROI claim.
- One fixed development split and one model seed.
