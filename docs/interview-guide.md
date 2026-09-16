# Interview guide

## 60-second project overview

I built a causal advertising analysis pipeline using the corrected Criteo Uplift v2.1 randomized dataset. Rather than only predicting who would visit, I estimated whose behavior could change because of assigned advertising. I audited about 13.98 million records, estimated full-sample treatment effects, and compared S/T/X-Learners and Causal Forest on a fixed modeling sample. The S-Learner was selected by validation Qini before evaluating the test partition. In the top-10% audience segments, its propensity-adjusted visit uplift was 9.4848%, versus 8.2700% for the response baseline. The findings support incremental-value ranking as an offline targeting approach, but they are not a deployed campaign result or a population-level ROI estimate.

## Questions to prepare

### Why causal uplift instead of response prediction?

Response prediction estimates `P(visit | X)`; uplift estimates the change in outcome under treatment versus control. A user likely to visit without advertising may be a poor candidate for incremental targeting. The two policies select different users and must be evaluated using treatment-aware outcomes.

### Why exclude exposure?

Exposure occurs after assigned treatment. Conditioning on it can change the estimand and introduce post-treatment selection bias. The baseline covariates are only `f0`–`f11`; assigned treatment is toggled as an intervention input in the S-Learner.

### What does the ATE number mean?

The sample visit ATE is +1.0342 **percentage points**, not +1.0342% relative lift. It is the assigned-treatment difference in mean visit rates in the released sample. The separately recorded 95% bootstrap interval is [1.0060, 1.0623] percentage points.

### Why did you choose the S-Learner?

It had the highest Qini coefficient on the fixed validation partition. Selection did not use the test partition. This is evidence about this development setup, not proof that S-Learners outperform other methods generally.

### How are Qini, AUUC, and uplift@K calculated?

Rank users by model score. With evaluation-partition treatment propensity `e`, calculate `Z = Y[T/e − (1−T)/(1−e)]`. Uplift@K is the selected-segment mean of `Z`; cumulative gain is targeting fraction times that mean. AUUC integrates the gain curve; Qini integrates gain above the random-targeting reference. These definitions must be checked before comparing numbers from different implementations.

### Is the top-10% gain statistically significant?

That has not been established for the difference between the two policies. The 1.2148-percentage-point difference is an observed offline point-estimate comparison. Full-sample ATE confidence intervals do not establish uncertainty for the targeting-policy difference.

### What would you do before production use?

Repeat development across prespecified splits/seeds, estimate policy-difference uncertainty with a suitable paired resampling procedure, examine overlap and segment stability, incorporate campaign costs and outcome values, and validate the chosen policy prospectively through a randomized test.

## Claims to avoid

- Do not describe publisher-subsampled effects as original campaign-population lift.
- Do not claim revenue uplift, ROI improvement, production deployment, or an online A/B-test win.
- Do not say the control balance checks prove randomization.
- Do not treat observed uplift as an individual causal ground-truth label.
- Do not describe this anonymous-feature dataset as a rich demographic customer-profile analysis.

See the [experiment validation](../reports/experiment_validation.md), [ATE analysis](../reports/ate_analysis.md), and [uplift evaluation](../reports/uplift_evaluation.md) for the recorded evidence.
