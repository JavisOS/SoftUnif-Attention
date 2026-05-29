# CLUTRR Vanilla Classifier Audit

Generated from `/vepfs/tsra_outputs/clutrr_vanilla_classifier_audit/clutrr_vanilla_classifier_audit_20260529_044530`.

Completion: `6/6 done`, `0 failed`.

Dataset: `data_089907f8`, training on 2/3-hop CLUTRR files and testing on 2-10 hops.

Entry point: `python -m clutrr.cli.baseline`.

Setting: story + query input, Hugging Face encoder + linear classifier, final-label cross entropy only. This audit does not use entity spans, TSRA relation attention, trace/path supervision, renamed-input CE augmentation, or consistency losses.

Values are `mean +/- sample-std` over seeds `0/1/42`. `Best` is selected by best logged overall accuracy across 10 epochs; `Final` is epoch 10.

| Backbone | Seeds | Best Overall | Best Short | Best Long >=6 | Final Overall | Final Short | Final Long >=6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RoBERTa | 3 | 0.3249 +/- 0.0068 | 0.9650 +/- 0.0070 | 0.2223 +/- 0.0054 | 0.2880 +/- 0.0123 | 0.9603 +/- 0.0040 | 0.1732 +/- 0.0086 |
| DeBERTa-v3 | 3 | 0.3610 +/- 0.0465 | 0.9580 +/- 0.0121 | 0.2426 +/- 0.0750 | 0.3310 +/- 0.0487 | 0.9580 +/- 0.0121 | 0.2222 +/- 0.0625 |

## Per-Seed Details

| Backbone | Seed | Best Epoch | Best Overall | Best Short | Best Long >=6 | Final Overall | Final Short | Final Long >=6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RoBERTa | 0 | 6 | 0.3229 | 0.9650 | 0.2254 | 0.2836 | 0.9580 | 0.1815 |
| RoBERTa | 1 | 7 | 0.3194 | 0.9720 | 0.2254 | 0.2784 | 0.9580 | 0.1643 |
| RoBERTa | 42 | 4 | 0.3325 | 0.9580 | 0.2160 | 0.3019 | 0.9650 | 0.1737 |
| DeBERTa-v3 | 0 | 9 | 0.3255 | 0.9720 | 0.1737 | 0.3255 | 0.9510 | 0.2113 |
| DeBERTa-v3 | 1 | 7 | 0.3438 | 0.9510 | 0.2316 | 0.2853 | 0.9510 | 0.1659 |
| DeBERTa-v3 | 42 | 5 | 0.4136 | 0.9510 | 0.3224 | 0.3822 | 0.9720 | 0.2895 |

## Interpretation

This audit confirms that the CLUTRR same-backbone `label-only` rows in `CLUTRR_BACKBONE_SWEEP_20260529.md` are not plain vanilla classifier baselines. Those rows are TSRA-architecture label-only ablations. The pure RoBERTa and DeBERTa-v3 classifier baselines are much lower on the same `data_089907f8` shallow-train/deep-test split.
