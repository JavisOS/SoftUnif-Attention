# CREST-style CLUTRR Counterfactual Baseline

Run root: `/vepfs/tsra_outputs/crest_clutrr/crest_clutrr_20260626_002032`

Method label: `crest_style_query_reverse_rename`.
Official CREST code was not found in the public search performed for DOI `10.1016/j.ipm.2025.104418`; this is a fair-input adaptation using final-label supervision plus counterfactual query reversal/entity-renaming.

Values are `mean +/- sample-std` over completed seeds.

## data_089907f8 / deberta-v3

| Selection | Seeds | Overall | Short 2-3 | Long >=6 | Rename consistency | Reverse-CF accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Best | 3 | 0.6428 +/- 0.0013 | 0.9394 +/- 0.0107 | 0.5216 +/- 0.0059 | 0.8141 +/- 0.0333 | 0.5451 +/- 0.0111 |
| Final | 3 | 0.6082 +/- 0.0277 | 0.9510 +/- 0.0185 | 0.4690 +/- 0.0406 | 0.8138 +/- 0.0252 | 0.5151 +/- 0.0251 |

### Per-hop Best

| Hop | Accuracy |
| ---: | ---: |
| 2 | 0.9649 +/- 0.0402 |
| 3 | 0.9302 +/- 0.0055 |
| 4 | 0.7702 +/- 0.0299 |
| 5 | 0.7050 +/- 0.0327 |
| 6 | 0.5826 +/- 0.0235 |
| 7 | 0.5394 +/- 0.0145 |
| 8 | 0.4289 +/- 0.0204 |
| 9 | 0.5658 +/- 0.0463 |
| 10 | 0.5182 +/- 0.0540 |

