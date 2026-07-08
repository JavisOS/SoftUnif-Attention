# CLUTRR Per-Hop Representative Rerun

Generated on 2026-06-02 07:38:04 from `/vepfs/tsra_outputs/unified_core_regression/unified_core_regression_20260601_144036`.

Values are `mean +/- sample-std` over seeds `0/1/42` when all three seeds are complete.
`Best` selects the epoch with the highest overall accuracy for each seed; `Final` uses the last logged epoch.

## data_089907f8 Best Epoch

| Backbone | Variant | Seeds | Overall | Short 2-3 | Long >=6 | Hop 2 | Hop 3 | Hop 4 | Hop 5 | Hop 6 | Hop 7 | Hop 8 | Hop 9 | Hop 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RoBERTa | tsra | 3 | 0.5718 +/- 0.0199 | 0.7457 +/- 0.0075 | 0.4189 +/- 0.0409 | 0.7893 +/- 0.0553 | 0.7228 +/- 0.0179 | 0.6196 +/- 0.0297 | 0.5281 +/- 0.0372 | 0.4302 +/- 0.0404 | 0.3401 +/- 0.0386 | 0.4512 +/- 0.0761 | 0.5407 +/- 0.0129 | 0.3646 +/- 0.0360 |

## data_089907f8 Final Epoch

| Backbone | Variant | Seeds | Overall | Short 2-3 | Long >=6 | Hop 2 | Hop 3 | Hop 4 | Hop 5 | Hop 6 | Hop 7 | Hop 8 | Hop 9 | Hop 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RoBERTa | tsra | 3 | 0.5355 +/- 0.0351 | 0.7056 +/- 0.0513 | 0.3770 +/- 0.0258 | 0.7547 +/- 0.0661 | 0.6799 +/- 0.0498 | 0.5763 +/- 0.0588 | 0.5165 +/- 0.0151 | 0.3875 +/- 0.0247 | 0.3095 +/- 0.0257 | 0.4146 +/- 0.0732 | 0.4889 +/- 0.0445 | 0.2917 +/- 0.0478 |

## data_db9b8f04 Best Epoch

_No completed rows._

## data_db9b8f04 Final Epoch

_No completed rows._

## Configuration Notes

- `data_089907f8`: primary CLUTRR split, 2/3-hop training and 2-10-hop testing.
- `data_db9b8f04`: follow-up CLUTRR split, 2/3/4-hop training and long-hop testing.
- `vanilla`: plain story+query Transformer classifier trained with final-label cross entropy only.
- `tsra`: TRUA architecture with `lambda_nexthop=1.0`, `lambda_edge=1.0`, `lambda_consistency=5.0`.
