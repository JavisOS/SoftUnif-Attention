# CLUTRR Per-Hop Representative Rerun

Generated on 2026-06-01 22:37:12 from `/vepfs/tsra_outputs/clutrr_llama32_3b_directional_precore/clutrr_llama32_3b_directional_precore_20260601_135027`.

Values are `mean +/- sample-std` over seeds `0/1/42` when all three seeds are complete.
`Best` selects the epoch with the highest overall accuracy for each seed; `Final` uses the last logged epoch.

## data_089907f8 Best Epoch

| Backbone | Variant | Seeds | Overall | Short 2-3 | Long >=6 | Hop 2 | Hop 3 | Hop 4 | Hop 5 | Hop 6 | Hop 7 | Hop 8 | Hop 9 | Hop 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama3-2-3b | tsra | 3 | 0.2205 +/- 0.0175 | 0.4654 +/- 0.0295 | 0.1203 +/- 0.0134 | 0.4371 +/- 0.0599 | 0.4802 +/- 0.0276 | 0.1540 +/- 0.0233 | 0.1188 +/- 0.0227 | 0.1140 +/- 0.0355 | 0.1259 +/- 0.0213 | 0.1016 +/- 0.0550 | 0.1630 +/- 0.0340 | 0.1145 +/- 0.0722 |

## data_089907f8 Final Epoch

| Backbone | Variant | Seeds | Overall | Short 2-3 | Long >=6 | Hop 2 | Hop 3 | Hop 4 | Hop 5 | Hop 6 | Hop 7 | Hop 8 | Hop 9 | Hop 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama3-2-3b | tsra | 3 | 0.2048 +/- 0.0192 | 0.4329 +/- 0.0406 | 0.0980 +/- 0.0319 | 0.4340 +/- 0.0378 | 0.4323 +/- 0.0627 | 0.1540 +/- 0.0088 | 0.1205 +/- 0.0282 | 0.0712 +/- 0.0275 | 0.0986 +/- 0.0387 | 0.1057 +/- 0.0550 | 0.1259 +/- 0.0128 | 0.1354 +/- 0.0180 |

## data_db9b8f04 Best Epoch

_No completed rows._

## data_db9b8f04 Final Epoch

_No completed rows._

## Configuration Notes

- `data_089907f8`: primary CLUTRR split, 2/3-hop training and 2-10-hop testing.
- `data_db9b8f04`: follow-up CLUTRR split, 2/3/4-hop training and long-hop testing.
- `vanilla`: plain story+query Transformer classifier trained with final-label cross entropy only.
- `tsra`: TRUA architecture with `lambda_nexthop=1.0`, `lambda_edge=1.0`, `lambda_consistency=5.0`.
