# Targeting policy analysis

Selected on validation Qini: `s_learner`. Test sample: 299,251 users.

| Outcome | Strategy | Target rate | Targeted users | Uplift | Estimated incremental outcomes in model test |
| --- | --- | ---: | ---: | ---: | ---: |
| visit | causal_s_learner | 10% | 29,926 | 9.484796% | 2838.33 |
| visit | causal_s_learner | 20% | 59,851 | 4.651524% | 2783.95 |
| visit | causal_s_learner | 30% | 89,776 | 3.277426% | 2942.32 |
| visit | response_model | 10% | 29,926 | 8.270001% | 2474.81 |
| visit | response_model | 20% | 59,851 | 4.617093% | 2763.34 |
| visit | response_model | 30% | 89,776 | 3.171296% | 2847.04 |
| conversion | causal_s_learner | 10% | 29,926 | 1.090959% | 326.47 |
| conversion | causal_s_learner | 20% | 59,851 | 0.599426% | 358.76 |
| conversion | causal_s_learner | 30% | 89,776 | 0.399950% | 359.06 |
| conversion | response_model | 10% | 29,926 | 0.997214% | 298.42 |
| conversion | response_model | 20% | 59,851 | 0.557931% | 333.92 |
| conversion | response_model | 30% | 89,776 | 0.396841% | 356.27 |

Incremental outcomes are randomized-test estimates within the sampled test partition, not production forecasts or financial impact. Conversion is a secondary evaluation using visit-trained rankings; no conversion-specific model selection occurred.

Exclusive selected-score segments: {'high_positive': 29926, 'moderate_positive': 66340, 'near_zero_0_to_0.0005': 197882, 'negative': 5103}.

No cost or value fields are available, so no ROI is reported.
