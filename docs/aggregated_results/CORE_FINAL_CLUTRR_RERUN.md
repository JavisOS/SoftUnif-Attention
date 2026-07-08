# CLUTRR Per-Hop Representative Rerun

Generated on 2026-06-03 07:51:27 from `/vepfs/tsra_outputs/core_final_rerun/core_final_rerun_20260602_094419`.

Values are `mean +/- sample-std` over seeds `0/1/42` when all three seeds are complete.
`Best` selects the epoch with the highest overall accuracy for each seed; `Final` uses the last logged epoch.

## data_089907f8 Best Epoch

| Backbone | Variant | Seeds | Overall | Short 2-3 | Long >=6 | Hop 2 | Hop 3 | Hop 4 | Hop 5 | Hop 6 | Hop 7 | Hop 8 | Hop 9 | Hop 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeBERTa | tsra | 3 | 0.6262 +/- 0.0214 | 0.8117 +/- 0.0181 | 0.4198 +/- 0.0335 | 0.8459 +/- 0.0055 | 0.7937 +/- 0.0281 | 0.7366 +/- 0.0077 | 0.5825 +/- 0.0223 | 0.4644 +/- 0.0470 | 0.3435 +/- 0.0156 | 0.4350 +/- 0.0735 | 0.5259 +/- 0.0559 | 0.3021 +/- 0.0361 |
| DeBERTa-v3 | tsra | 3 | 0.6786 +/- 0.0368 | 0.8095 +/- 0.0199 | 0.5169 +/- 0.0622 | 0.8616 +/- 0.0381 | 0.7822 +/- 0.0131 | 0.7837 +/- 0.0281 | 0.6419 +/- 0.0372 | 0.5271 +/- 0.0953 | 0.4490 +/- 0.0530 | 0.5569 +/- 0.0550 | 0.6074 +/- 0.0462 | 0.4583 +/- 0.0361 |

## data_089907f8 Final Epoch

| Backbone | Variant | Seeds | Overall | Short 2-3 | Long >=6 | Hop 2 | Hop 3 | Hop 4 | Hop 5 | Hop 6 | Hop 7 | Hop 8 | Hop 9 | Hop 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeBERTa | tsra | 3 | 0.5777 +/- 0.0083 | 0.7857 +/- 0.0117 | 0.3654 +/- 0.0147 | 0.8333 +/- 0.0054 | 0.7607 +/- 0.0151 | 0.6807 +/- 0.0022 | 0.5198 +/- 0.0179 | 0.3989 +/- 0.0215 | 0.3367 +/- 0.0368 | 0.3659 +/- 0.0322 | 0.4296 +/- 0.0340 | 0.2396 +/- 0.0650 |
| DeBERTa-v3 | tsra | 3 | 0.6693 +/- 0.0303 | 0.8095 +/- 0.0316 | 0.4974 +/- 0.0578 | 0.8491 +/- 0.0432 | 0.7888 +/- 0.0286 | 0.7939 +/- 0.0138 | 0.6122 +/- 0.0200 | 0.5185 +/- 0.0808 | 0.4490 +/- 0.0467 | 0.5203 +/- 0.0812 | 0.5555 +/- 0.0385 | 0.4271 +/- 0.0478 |

## data_db9b8f04 Best Epoch

| Backbone | Variant | Seeds | Overall | Short 2-3 | Long >=6 | Hop 2 | Hop 3 | Hop 4 | Hop 5 | Hop 6 | Hop 7 | Hop 8 | Hop 9 | Hop 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeBERTa-v3 | tsra | 3 | 0.7741 +/- 0.0127 | 0.8263 +/- 0.0076 | 0.7370 +/- 0.0228 | 0.8602 +/- 0.0168 | 0.8058 +/- 0.0129 | 0.8016 +/- 0.0079 | 0.7440 +/- 0.0315 | 0.8255 +/- 0.0422 | 0.7167 +/- 0.0144 | 0.6589 +/- 0.0641 | 0.8125 +/- 0.0361 | 0.6306 +/- 0.0156 |

## data_db9b8f04 Final Epoch

| Backbone | Variant | Seeds | Overall | Short 2-3 | Long >=6 | Hop 2 | Hop 3 | Hop 4 | Hop 5 | Hop 6 | Hop 7 | Hop 8 | Hop 9 | Hop 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeBERTa-v3 | tsra | 3 | 0.6667 +/- 0.0289 | 0.8152 +/- 0.0061 | 0.5176 +/- 0.0605 | 0.8656 +/- 0.0259 | 0.7848 +/- 0.0074 | 0.7593 +/- 0.0321 | 0.6598 +/- 0.0089 | 0.6012 +/- 0.0442 | 0.4861 +/- 0.1005 | 0.5271 +/- 0.0374 | 0.4722 +/- 0.0939 | 0.4144 +/- 0.0312 |

## Configuration Notes

- `data_089907f8`: primary CLUTRR split, 2/3-hop training and 2-10-hop testing.
- `data_db9b8f04`: follow-up CLUTRR split, 2/3/4-hop training and long-hop testing.
- `vanilla`: plain story+query Transformer classifier trained with final-label cross entropy only.
- `tsra`: TRUA architecture with `lambda_nexthop=1.0`, `lambda_edge=1.0`, `lambda_consistency=5.0`.
