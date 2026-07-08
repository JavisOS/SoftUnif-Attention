# CLUTRR Per-Hop Representative Rerun

Generated on 2026-06-01 17:47:07 from `/vepfs/tsra_outputs/clutrr_qwen17_directional_precore/clutrr_qwen17_directional_precore_20260601_134459`.

Values are `mean +/- sample-std` over seeds `0/1/42` when all three seeds are complete.
`Best` selects the epoch with the highest overall accuracy for each seed; `Final` uses the last logged epoch.

## data_089907f8 Best Epoch

| Backbone | Variant | Seeds | Overall | Short 2-3 | Long >=6 | Hop 2 | Hop 3 | Hop 4 | Hop 5 | Hop 6 | Hop 7 | Hop 8 | Hop 9 | Hop 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-1-7b-base | tsra | 3 | 0.3284 +/- 0.0237 | 0.5822 +/- 0.0244 | 0.1845 +/- 0.0237 | 0.5786 +/- 0.0144 | 0.5842 +/- 0.0325 | 0.2964 +/- 0.0361 | 0.2492 +/- 0.0206 | 0.1795 +/- 0.0086 | 0.1667 +/- 0.0294 | 0.1545 +/- 0.0281 | 0.2371 +/- 0.0714 | 0.2604 +/- 0.0180 |

## data_089907f8 Final Epoch

| Backbone | Variant | Seeds | Overall | Short 2-3 | Long >=6 | Hop 2 | Hop 3 | Hop 4 | Hop 5 | Hop 6 | Hop 7 | Hop 8 | Hop 9 | Hop 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-1-7b-base | tsra | 3 | 0.2993 +/- 0.0038 | 0.5541 +/- 0.0150 | 0.1738 +/- 0.0351 | 0.5566 +/- 0.0189 | 0.5528 +/- 0.0318 | 0.2621 +/- 0.0344 | 0.1914 +/- 0.0200 | 0.1567 +/- 0.0550 | 0.1497 +/- 0.0257 | 0.1870 +/- 0.0186 | 0.2148 +/- 0.0256 | 0.2187 +/- 0.0625 |

## data_db9b8f04 Best Epoch

_No completed rows._

## data_db9b8f04 Final Epoch

_No completed rows._

## Configuration Notes

- `data_089907f8`: primary CLUTRR split, 2/3-hop training and 2-10-hop testing.
- `data_db9b8f04`: follow-up CLUTRR split, 2/3/4-hop training and long-hop testing.
- `vanilla`: plain story+query Transformer classifier trained with final-label cross entropy only.
- `tsra`: TRUA architecture with `lambda_nexthop=1.0`, `lambda_edge=1.0`, `lambda_consistency=5.0`.
