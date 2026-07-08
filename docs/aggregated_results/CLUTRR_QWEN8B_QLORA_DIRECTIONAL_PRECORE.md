# CLUTRR Per-Hop Representative Rerun

Generated on 2026-06-02 22:44:16 from `/vepfs/tsra_outputs/clutrr_qwen8b_qlora_directional_precore/clutrr_qwen8b_qlora_directional_precore_20260601_134459`.

Values are `mean +/- sample-std` over seeds `0/1/42` when all three seeds are complete.
`Best` selects the epoch with the highest overall accuracy for each seed; `Final` uses the last logged epoch.

## data_089907f8 Best Epoch

| Backbone | Variant | Seeds | Overall | Short 2-3 | Long >=6 | Hop 2 | Hop 3 | Hop 4 | Hop 5 | Hop 6 | Hop 7 | Hop 8 | Hop 9 | Hop 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-8b-base | tsra | 3 | 0.5625 +/- 0.0444 | 0.7695 +/- 0.0394 | 0.3788 +/- 0.0382 | 0.8019 +/- 0.0432 | 0.7525 +/- 0.0373 | 0.6514 +/- 0.0746 | 0.4719 +/- 0.0397 | 0.4188 +/- 0.0476 | 0.3265 +/- 0.0102 | 0.4553 +/- 0.0428 | 0.3556 +/- 0.0445 | 0.2291 +/- 0.0902 |

## data_089907f8 Final Epoch

| Backbone | Variant | Seeds | Overall | Short 2-3 | Long >=6 | Hop 2 | Hop 3 | Hop 4 | Hop 5 | Hop 6 | Hop 7 | Hop 8 | Hop 9 | Hop 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-8b-base | tsra | 3 | 0.5617 +/- 0.0433 | 0.7543 +/- 0.0248 | 0.3850 +/- 0.0469 | 0.7987 +/- 0.0393 | 0.7310 +/- 0.0234 | 0.6450 +/- 0.0649 | 0.4868 +/- 0.0488 | 0.4331 +/- 0.0686 | 0.3435 +/- 0.0312 | 0.4431 +/- 0.0254 | 0.3630 +/- 0.0559 | 0.2187 +/- 0.0827 |

## data_db9b8f04 Best Epoch

_No completed rows._

## data_db9b8f04 Final Epoch

_No completed rows._

## Configuration Notes

- `data_089907f8`: primary CLUTRR split, 2/3-hop training and 2-10-hop testing.
- `data_db9b8f04`: follow-up CLUTRR split, 2/3/4-hop training and long-hop testing.
- `vanilla`: plain story+query Transformer classifier trained with final-label cross entropy only.
- `tsra`: TRUA architecture with `lambda_nexthop=1.0`, `lambda_edge=1.0`, `lambda_consistency=5.0`.
