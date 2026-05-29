# CLUTRR Vanilla Classifier Audit

Generated from:

- `/vepfs/tsra_outputs/clutrr_vanilla_classifier_audit/clutrr_vanilla_classifier_audit_20260529_044530`
- `/vepfs/tsra_outputs/clutrr_vanilla_classifier_missing_backbones/clutrr_vanilla_classifier_missing_backbones_20260529_053155`

Completion: `15/15 done`, `0 failed`.

Dataset: `data_089907f8`, training on 2/3-hop CLUTRR files and testing on 2-10 hops.

Entry point: `python -m clutrr.cli.baseline`.

Setting: story + query input, Hugging Face encoder + linear classifier, final-label cross entropy only. This audit does not use entity spans, TSRA relation attention, trace/path supervision, renamed-input CE augmentation, or consistency losses.

Values are `mean +/- sample-std` over seeds `0/1/42`. `Best` is selected by best logged overall accuracy across 10 epochs; `Final` is epoch 10.

| Backbone | Seeds | Best Overall | Best Short | Best Long >=6 | Final Overall | Final Short | Final Long >=6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT | 3 | 0.2874 +/- 0.0103 | 0.9697 +/- 0.0040 | 0.1946 +/- 0.0254 | 0.2542 +/- 0.0232 | 0.9697 +/- 0.0245 | 0.1445 +/- 0.0323 |
| RoBERTa | 3 | 0.3249 +/- 0.0068 | 0.9650 +/- 0.0070 | 0.2223 +/- 0.0054 | 0.2880 +/- 0.0123 | 0.9603 +/- 0.0040 | 0.1732 +/- 0.0086 |
| DeBERTa | 3 | 0.3438 +/- 0.0186 | 0.9487 +/- 0.0176 | 0.2415 +/- 0.0122 | 0.2821 +/- 0.0229 | 0.9440 +/- 0.0305 | 0.1685 +/- 0.0265 |
| DeBERTa-v3 | 3 | 0.3610 +/- 0.0465 | 0.9580 +/- 0.0121 | 0.2426 +/- 0.0750 | 0.3310 +/- 0.0487 | 0.9580 +/- 0.0121 | 0.2222 +/- 0.0625 |
| ModernBERT | 3 | 0.2958 +/- 0.0263 | 0.9301 +/- 0.0210 | 0.1873 +/- 0.0235 | 0.2822 +/- 0.0201 | 0.9487 +/- 0.0106 | 0.1690 +/- 0.0071 |

## Per-Seed Details

| Backbone | Seed | Best Epoch | Best Overall | Best Short | Best Long >=6 | Final Overall | Final Short | Final Long >=6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BERT | 0 | 2 | 0.2993 | 0.9650 | 0.2238 | 0.2417 | 0.9930 | 0.1221 |
| BERT | 1 | 7 | 0.2818 | 0.9720 | 0.1784 | 0.2400 | 0.9441 | 0.1299 |
| BERT | 42 | 10 | 0.2810 | 0.9720 | 0.1815 | 0.2810 | 0.9720 | 0.1815 |
| RoBERTa | 0 | 6 | 0.3229 | 0.9650 | 0.2254 | 0.2836 | 0.9580 | 0.1815 |
| RoBERTa | 1 | 7 | 0.3194 | 0.9720 | 0.2254 | 0.2784 | 0.9580 | 0.1643 |
| RoBERTa | 42 | 4 | 0.3325 | 0.9580 | 0.2160 | 0.3019 | 0.9650 | 0.1737 |
| DeBERTa | 0 | 2 | 0.3639 | 0.9510 | 0.2551 | 0.2557 | 0.9091 | 0.1393 |
| DeBERTa | 1 | 1 | 0.3272 | 0.9301 | 0.2316 | 0.2958 | 0.9650 | 0.1909 |
| DeBERTa | 42 | 4 | 0.3403 | 0.9650 | 0.2379 | 0.2949 | 0.9580 | 0.1753 |
| DeBERTa-v3 | 0 | 9 | 0.3255 | 0.9720 | 0.1737 | 0.3255 | 0.9510 | 0.2113 |
| DeBERTa-v3 | 1 | 7 | 0.3438 | 0.9510 | 0.2316 | 0.2853 | 0.9510 | 0.1659 |
| DeBERTa-v3 | 42 | 5 | 0.4136 | 0.9510 | 0.3224 | 0.3822 | 0.9720 | 0.2895 |
| ModernBERT | 0 | 5 | 0.2679 | 0.9301 | 0.1628 | 0.2627 | 0.9510 | 0.1674 |
| ModernBERT | 1 | 2 | 0.2993 | 0.9510 | 0.1894 | 0.2810 | 0.9580 | 0.1628 |
| ModernBERT | 42 | 1 | 0.3202 | 0.9091 | 0.2097 | 0.3028 | 0.9371 | 0.1768 |

## Interpretation

This audit confirms that the CLUTRR same-backbone `label-only` rows in `CLUTRR_BACKBONE_SWEEP_20260529.md` are not plain vanilla classifier baselines. Those rows are TSRA-architecture label-only ablations. The pure classifier baselines are much lower on the same `data_089907f8` shallow-train/deep-test split across all five checked backbones.
