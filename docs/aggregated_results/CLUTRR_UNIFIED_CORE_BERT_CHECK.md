# CLUTRR Per-Hop Representative Rerun

Generated on 2026-06-01 13:31:36 from `/vepfs/tsra_outputs/clutrr_unified_core_bert_check/clutrr_unified_core_bert_check_20260601_111350`.

Values are `mean +/- sample-std` over seeds `0/1/42` when all three seeds are complete.
`Best` selects the epoch with the highest overall accuracy for each seed; `Final` uses the last logged epoch.

## data_089907f8 Best Epoch

| Backbone | Variant | Seeds | Overall | Short 2-3 | Long >=6 | Hop 2 | Hop 3 | Hop 4 | Hop 5 | Hop 6 | Hop 7 | Hop 8 | Hop 9 | Hop 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bert | tsra | 3 | 0.4165 +/- 0.0113 | 0.6028 +/- 0.0082 | 0.3093 +/- 0.0397 | 0.6667 +/- 0.0197 | 0.5693 +/- 0.0050 | 0.4096 +/- 0.0176 | 0.3399 +/- 0.0200 | 0.2735 +/- 0.0617 | 0.2857 +/- 0.0204 | 0.3334 +/- 0.0186 | 0.4074 +/- 0.0513 | 0.3125 +/- 0.1127 |

## data_089907f8 Final Epoch

| Backbone | Variant | Seeds | Overall | Short 2-3 | Long >=6 | Hop 2 | Hop 3 | Hop 4 | Hop 5 | Hop 6 | Hop 7 | Hop 8 | Hop 9 | Hop 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bert | tsra | 3 | 0.3973 +/- 0.0261 | 0.5985 +/- 0.0049 | 0.2816 +/- 0.0430 | 0.6383 +/- 0.0218 | 0.5776 +/- 0.0187 | 0.3918 +/- 0.0268 | 0.3119 +/- 0.0423 | 0.2308 +/- 0.0444 | 0.2585 +/- 0.0513 | 0.3008 +/- 0.0626 | 0.4074 +/- 0.0513 | 0.3125 +/- 0.0313 |

## data_db9b8f04 Best Epoch

_No completed rows._

## data_db9b8f04 Final Epoch

_No completed rows._

## Configuration Notes

- `data_089907f8`: primary CLUTRR split, 2/3-hop training and 2-10-hop testing.
- `data_db9b8f04`: follow-up CLUTRR split, 2/3/4-hop training and long-hop testing.
- `vanilla`: plain story+query Transformer classifier trained with final-label cross entropy only.
- `tsra`: TRUA architecture with `lambda_nexthop=1.0`, `lambda_edge=1.0`, `lambda_consistency=5.0`.
