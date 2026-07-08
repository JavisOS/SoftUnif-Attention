# CLUTRR Per-Hop Representative Rerun

Generated on 2026-06-01 11:30:30 from `/vepfs/tsra_outputs/clutrr_gpt2_precore/clutrr_gpt2_precore_20260601_111210`.

Values are `mean +/- sample-std` over seeds `0/1/42` when all three seeds are complete.
`Best` selects the epoch with the highest overall accuracy for each seed; `Final` uses the last logged epoch.

## data_089907f8 Best Epoch

| Backbone | Variant | Seeds | Overall | Short 2-3 | Long >=6 | Hop 2 | Hop 3 | Hop 4 | Hop 5 | Hop 6 | Hop 7 | Hop 8 | Hop 9 | Hop 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2 | tsra | 3 | 0.2876 +/- 0.0033 | 0.5292 +/- 0.0198 | 0.1408 +/- 0.0094 | 0.4969 +/- 0.0238 | 0.5462 +/- 0.0200 | 0.2621 +/- 0.0023 | 0.2244 +/- 0.0103 | 0.1311 +/- 0.0099 | 0.1225 +/- 0.0177 | 0.1260 +/- 0.0140 | 0.2074 +/- 0.0256 | 0.1771 +/- 0.0181 |

## data_089907f8 Final Epoch

| Backbone | Variant | Seeds | Overall | Short 2-3 | Long >=6 | Hop 2 | Hop 3 | Hop 4 | Hop 5 | Hop 6 | Hop 7 | Hop 8 | Hop 9 | Hop 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2 | tsra | 3 | 0.2792 +/- 0.0114 | 0.5206 +/- 0.0160 | 0.1373 +/- 0.0041 | 0.4937 +/- 0.0288 | 0.5347 +/- 0.0178 | 0.2519 +/- 0.0101 | 0.2096 +/- 0.0223 | 0.1254 +/- 0.0099 | 0.0986 +/- 0.0059 | 0.1341 +/- 0.0000 | 0.2444 +/- 0.0223 | 0.1562 +/- 0.0313 |

## data_db9b8f04 Best Epoch

_No completed rows._

## data_db9b8f04 Final Epoch

_No completed rows._

## Configuration Notes

- `data_089907f8`: primary CLUTRR split, 2/3-hop training and 2-10-hop testing.
- `data_db9b8f04`: follow-up CLUTRR split, 2/3/4-hop training and long-hop testing.
- `vanilla`: plain story+query Transformer classifier trained with final-label cross entropy only.
- `tsra`: TRUA architecture with `lambda_nexthop=1.0`, `lambda_edge=1.0`, `lambda_consistency=5.0`.
