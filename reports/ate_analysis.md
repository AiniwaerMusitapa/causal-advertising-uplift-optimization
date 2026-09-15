# Average treatment effect

Estimand: intention-to-treat difference in means on all 13,979,592 rows.

| Outcome | Treatment rate | Control rate | ATE | Relative lift | SE | 95% normal CI | 95% bootstrap CI | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| visit | 4.854336% | 3.820096% | 1.034240% | 27.074% | 0.00014632 | [1.005562%, 1.062918%] | [1.006016%, 1.062278%] | 0 |
| conversion | 0.308946% | 0.193759% | 0.115187% | 59.449% | 0.00003437 | [0.108450%, 0.121924%] | [0.108340%, 0.122153%] | 3.19e-246 |

The bootstrap resamples binary outcomes independently within treatment arms, which is the exact row-level nonparametric bootstrap distribution for group means when rows are independent and outcomes are binary.
No revenue or advertising-cost fields exist, so these effects are not ROI estimates. `exposure` is not used in the estimand.
