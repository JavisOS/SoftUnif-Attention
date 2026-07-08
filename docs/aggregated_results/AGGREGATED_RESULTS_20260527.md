# Aggregated TRUA Results

Generated from the completed `/vepfs` experiment folders on 2026-05-27. Values are `mean +/- sample-std` over seeds `0/1/42` unless noted.

## Completion Status

- Additional depth/seed checks: `14/14 done, 0 failed`.
- Seed-42 completion queue: `22/22 done, 0 failed`.
- NLProofS formal test is still running separately and has not produced a final result file.

## CLUTRR data_089907f8

| Backbone | Variant | Seeds | Best Overall | Best Short | Best Long >=6 | Final Overall | Final Long >=6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deberta | full | 3 | 0.6262 +/- 0.0214 | 0.8117 +/- 0.0181 | 0.4198 +/- 0.0335 | 0.5777 +/- 0.0083 | 0.3654 +/- 0.0147 |
| deberta | label_only | 3 | 0.4706 +/- 0.0298 | 0.6602 +/- 0.0276 | 0.3414 +/- 0.0293 | 0.4171 +/- 0.0146 | 0.2719 +/- 0.0257 |
| deberta | no_consistency | 3 | 0.6222 +/- 0.0023 | 0.8214 +/- 0.0056 | 0.4109 +/- 0.0270 | 0.6073 +/- 0.0123 | 0.3779 +/- 0.0082 |
| roberta | full | 3 | 0.5512 +/- 0.0176 | 0.7327 +/- 0.0312 | 0.3930 +/- 0.0334 | 0.5137 +/- 0.0219 | 0.3432 +/- 0.0376 |
| roberta | label_only | 3 | 0.4887 +/- 0.0482 | 0.6580 +/- 0.0327 | 0.3779 +/- 0.0449 | 0.4593 +/- 0.0547 | 0.3235 +/- 0.0819 |
| roberta | no_consistency | 3 | 0.5785 +/- 0.0154 | 0.7587 +/- 0.0179 | 0.4127 +/- 0.0041 | 0.5398 +/- 0.0394 | 0.3824 +/- 0.0490 |

## CLUTRR data_db9b8f04 2/3/4-Hop Train Check

| Backbone | Variant | Seeds | Best Overall | Best Short | Best Long >=6 | Final Overall | Final Long >=6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deberta | label_only | 3 | 0.5862 +/- 0.0173 | 0.7555 +/- 0.0076 | 0.4925 +/- 0.0314 | 0.5471 +/- 0.0182 | 0.4338 +/- 0.0300 |
| deberta | tsra | 3 | 0.7455 +/- 0.0167 | 0.8596 +/- 0.0177 | 0.6508 +/- 0.0181 | 0.7074 +/- 0.0373 | 0.6131 +/- 0.0553 |

## ProofWriter Main TRUA-Prop

| Backbone | Model | Seeds | Depth-3 Acc | Depth-5 Acc | Depth-3 Trace@1 | Depth-5 Trace@1 | ByDepth-3 | ByDepth-5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bert | baseline | 0,1,42 | 0.9596 +/- 0.0029 | 0.8818 +/- 0.0086 | 0.2053 +/- 0.0237 | 0.1459 +/- 0.0061 | 0.8583 +/- 0.0206 | 0.7038 +/- 0.0097 |
| bert | tsra | 0,1,42 | 0.9601 +/- 0.0030 | 0.8877 +/- 0.0073 | 0.8315 +/- 0.0016 | 0.7841 +/- 0.0007 | 0.8738 +/- 0.0184 | 0.7114 +/- 0.0229 |
| roberta | baseline | 0,1,42 | 0.8703 +/- 0.1225 | 0.8064 +/- 0.0783 | 0.2080 +/- 0.0046 | 0.1694 +/- 0.0178 | 0.6745 +/- 0.1167 | 0.5738 +/- 0.0510 |
| roberta | tsra | 0,1,42 | 0.9453 +/- 0.0123 | 0.8531 +/- 0.0157 | 0.8430 +/- 0.0053 | 0.7934 +/- 0.0030 | 0.7585 +/- 0.0908 | 0.6006 +/- 0.0317 |
| deberta | baseline | 0,1,42 | 0.8663 +/- 0.1197 | 0.8146 +/- 0.0880 | 0.2498 +/- 0.0585 | 0.1687 +/- 0.0231 | 0.7938 +/- 0.2219 | 0.7629 +/- 0.2148 |
| deberta | tsra | 0,1,42 | 0.8740 +/- 0.1258 | 0.8068 +/- 0.0787 | 0.6197 +/- 0.3338 | 0.5896 +/- 0.3034 | 0.7372 +/- 0.1971 | 0.6570 +/- 0.1829 |

## RuleTaker GFaiR Split Main TRUA-Prop

| Backbone | Model | Seeds | Test Acc | Trace@1 | Depth Note |
| --- | --- | --- | --- | --- | --- |
| bert | baseline | 0,1,42 | 0.9624 +/- 0.0015 | 0.6090 +/- 0.0957 | official test depth metadata unavailable (-1 bucket) |
| bert | tsra | 0,1,42 | 0.9646 +/- 0.0022 | 0.0226 +/- 0.0017 | official test depth metadata unavailable (-1 bucket) |
| roberta | baseline | 0,1,42 | 0.9559 +/- 0.0030 | 0.5065 +/- 0.1179 | official test depth metadata unavailable (-1 bucket) |
| roberta | tsra | 0,1,42 | 0.9615 +/- 0.0009 | 0.0128 +/- 0.0036 | official test depth metadata unavailable (-1 bucket) |
| deberta | baseline | 0,1,42 | 0.7215 +/- 0.0000 | 0.4201 +/- 0.1834 | official test depth metadata unavailable (-1 bucket) |
| deberta | tsra | 0,1,42 | 0.9664 +/- 0.0016 | 0.0899 +/- 0.1329 | official test depth metadata unavailable (-1 bucket) |

## RuleTaker Raw Strict QDep 1/2 Train -> 1-5 Test

| Backbone | Model | Seeds | Overall | QDep1 | QDep2 | QDep3 | QDep4 | QDep5 | Trace@1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deberta | baseline | 0,1,42 | 0.5234 +/- 0.0000 | 0.5399 +/- 0.0000 | 0.5081 +/- 0.0000 | 0.5027 +/- 0.0000 | 0.5038 +/- 0.0000 | 0.5019 +/- 0.0000 | 0.1913 +/- 0.0051 |
| deberta | tsra | 0,1,42 | 0.7783 +/- 0.1353 | 0.8188 +/- 0.2180 | 0.8463 +/- 0.1650 | 0.6414 +/- 0.1981 | 0.5263 +/- 0.1471 | 0.4944 +/- 0.2509 | 0.7444 +/- 0.2414 |

## PrOntoQA-OOD

| Backbone | Model | Seeds | OOD Acc | Trace@1 | Depth3 | Depth4 |
| --- | --- | --- | --- | --- | --- | --- |
| bert | baseline | 0,1,42 | 1.0000 +/- 0.0000 | 0.2133 +/- 0.0448 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| bert | tsra | 0,1,42 | 1.0000 +/- 0.0000 | 0.4667 +/- 0.0404 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| roberta | baseline | 0,1,42 | 1.0000 +/- 0.0000 | 0.1733 +/- 0.0333 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| roberta | tsra | 0,1,42 | 1.0000 +/- 0.0000 | 0.4856 +/- 0.0876 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| deberta | baseline | 0,1,42 | 1.0000 +/- 0.0000 | 0.2511 +/- 0.0366 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| deberta | tsra | 0,1,42 | 1.0000 +/- 0.0000 | 0.5378 +/- 0.1482 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

## Notes

- CLUTRR values use the best logged evaluation point for main comparison; final-epoch values are also included for audit.
- The strongest stable CLUTRR signal is trace/next-hop supervision versus label-only, especially for DeBERTa on long-hop examples.
- RuleTaker raw strict QDep shows a large TRUA gain over label-only on overall accuracy, but seed-to-seed variance is high and should be reported transparently.
- PrOntoQA label accuracy is saturated in this processed split; trace@1 is the more informative internal-reasoning metric.
- ProofWriter DeBERTa seed42 baseline and TRUA are identical in the additional check; treat that cell as an audit flag rather than a strong conclusion.
