# TRUA Experiment Report

## 1. Goal

This report records the current experiment status, configurations, artifacts, and metrics for **TRUA: Transition-Regularized Unit Attention**.

The experiment question recorded for context is:

> Does training-time supervision over gold reasoning traces change Transformer performance and trace-selection metrics in query-conditioned multi-step textual reasoning, especially under long-hop, high-depth, and OOD settings?

The comparison scope is:

- same-backbone Transformer classifiers trained only with final labels;
- Transformer/attention/reasoning architectures that do not receive trace supervision;
- proof-reasoning methods that decompose natural-language reasoning into rule/fact selection or proof steps.

Evaluation emphasizes shallow-train / deep-test settings:

- train on short reasoning chains or low proof depth;
- test on longer chains, deeper proof depth, or OOD systematic/compositional splits;
- report hop/depth grouped metrics, not only overall accuracy.

## 0. Latest Status Snapshot

Updated on **2026-06-26** after the CREST-style CLUTRR counterfactual baseline was added to the completed LoGiPT, AAI, CODI-GPT2, CODI-Llama1B official train/distill, CLUTRR API LLM references, CLUTRR per-hop representative reruns, and unified-core regression checks. This section is the current authoritative summary. Values are `mean +/- sample-std` over seeds `0/1/42` unless stated otherwise.

### Completion Status

- **CLUTRR backbone sweep is complete:** `/vepfs/tsra_outputs/clutrr_backbone_sweep/latest` is `42/42 done, 0 failed`.
- **CLUTRR per-hop representative rerun is complete:** `/vepfs/tsra_outputs/clutrr_perhop_representative/latest` is `24/24 done, 0 failed`. This rerun compares plain vanilla story+query Transformer classifiers against TRUA and logs hop-2 through hop-10 accuracy for each seed/epoch.
- **CLUTRR API LLM reference results are recorded:** GPT-5.2 API and Gemini 3.1 Pro API were evaluated by raw label prompting on the public `data_089907f8` test set, with no CLUTRR task fine-tuning. These rows are scale-oriented LLM references, not same-input encoder baselines. GPT-5.2 reaches overall `0.6981` and long-hop `0.6385`; Gemini 3.1 Pro reaches overall `0.8010` and long-hop `0.7825`.
- **CLUTRR label-only audit note:** the CLUTRR `label-only` rows in the same-backbone tables are **TRUA-architecture label-only ablations**, not plain vanilla RoBERTa/DeBERTa classifier fine-tuning. They run through `clutrr.cli.train` / `TruaReasonerModel` with trace, edge, and consistency losses set to zero, but still use entity spans, query-conditioned pair/relation attention, and renamed-input label CE.
- **True vanilla CLUTRR classifier audit is complete:** RoBERTa/DeBERTa-v3 audit is `6/6 done, 0 failed`; missing-backbone audit at `/vepfs/tsra_outputs/clutrr_vanilla_classifier_missing_backbones/latest` is `9/9 done, 0 failed`. The full BERT/RoBERTa/DeBERTa/DeBERTa-v3/ModernBERT vanilla table is summarized in `docs/aggregated_results/CLUTRR_VANILLA_CLASSIFIER_AUDIT_20260529.md`.
- **CREST-style CLUTRR counterfactual baseline is complete:** official public code was not found for DOI `10.1016/j.ipm.2025.104418` after exact-title/DOI/GitHub searches. A fair-input adaptation was run on CLUTRR `data_089907f8` with DeBERTa-v3, 10 epochs, seeds `0/1/42`, final-label training plus entity-renaming consistency and query-reversal counterfactual labels derived from the CLUTRR relation schema. It uses raw story+query input at test time and no gold trace/path. Best overall is `0.6428 +/- 0.0013`; best long-hop >=6 is `0.5216 +/- 0.0059`.
- **TRUA/backbone runs are complete:** additional depth/seed checks are `14/14 done, 0 failed`; seed-42 completion is `22/22 done, 0 failed`.
- **Final aggregated artifacts:** `docs/aggregated_results/AGGREGATED_RESULTS_20260527.md`, `docs/aggregated_results/aggregated_results_20260527.json`, `docs/aggregated_results/CLUTRR_BACKBONE_SWEEP_20260529.md`, `docs/aggregated_results/clutrr_backbone_sweep_20260529.json`, `docs/aggregated_results/CLUTRR_PERHOP_REPRESENTATIVE.md`, `docs/aggregated_results/CLUTRR_PERHOP_REPRESENTATIVE.json`, and `docs/aggregated_results/CREST_CLUTRR_COUNTERFACTUAL.md/json`.
- **Aggregation scripts:** `scripts/aggregate_experiment_results.py`, `scripts/aggregate_clutrr_backbone_sweep.py`, and `scripts/aggregate_clutrr_perhop.py`.
- **Same-input non-CLUTRR attention baselines are complete:** `scripts/fair_attention_prop.py` finished DAT / Dual Attention and Abstractor/RCA on ProofWriter, RuleTaker raw-QDep, and PrOntoQA-OOD using the same raw-text/no-external-solver setting as TRUA. All reported rows below use DeBERTa-base, 10 epochs, and seeds `0/1/42`.
- **Same-input non-CLUTRR attention baseline summary:** DAT / Dual Attention and Abstractor/RCA results are recorded for ProofWriter, RuleTaker raw-QDep, and PrOntoQA-OOD. On PrOntoQA-OOD, binary label accuracy is saturated for all methods; recorded trace@1 values are TRUA-DeBERTa `0.5378 +/- 0.1482`, DAT `0.2400 +/- 0.0850`, and Abstractor/RCA `0.2044 +/- 0.0158`.
- **Coconut PrOntoQA-OOD run is complete:** official Coconut code ran on the converted TRUA PrOntoQA-OOD data and finished with validation accuracy `80/80 = 1.0000`; recorded CoT match is `0/80 = 0.0000`.
- **CODI official train/distill is complete at two scales:** using CODI's official PrOntoQA training/distillation branch on the converted TRUA PrOntoQA-OOD shallow/deep split (`train` depth 1/2, `test` depth 3/4), CODI-GPT2 reaches exact final-statement accuracy `0.8144` over `2700` test examples: depth-3 `1.0000`, depth-4 `0.7217`. The CODI-Llama1B run uses `/vepfs/tsra_models/hf/Llama-3.2-1B-Instruct` and reaches overall `0.8730`: depth-3 `1.0000`, depth-4 `0.8094`.
- **LoGiPT and AAI are complete:** LoGiPT CodeLlama-13B on the raw ProofWriter depth-3/depth-5 adapter reaches macro depth accuracy `0.6515` (`0.6466` on depth-3, `0.6564` on depth-5). The reported-style Logic-LM ProofWriter test run reaches `0.4517` accuracy over `600` examples. AAI Qwen3-32B reaches `0.8350` without attention intervention and `0.8267` with the released attention-aware intervention setting on the same `600`-example ProofWriter test file.
- **NLProofS formal RuleTaker test is complete:** final result file is at `/vepfs/tsra_outputs/official_external/latest_nlproofs_ruletaker_test/prover_test/lightning_logs/version_0/results_test.json`. Reported test metrics: answer overall `0.6796`, proof overall `0.9187`.
- **Unified TRUA core regression is complete:** `/vepfs/tsra_outputs/unified_core_regression/latest` is `12/12 done, 0 failed`. The run validates the refactor to a shared transition-regularized unit-attention core with task-specific adapters: CLUTRR uses the entity-path adapter; ProofWriter, RuleTaker, and PrOntoQA use the proposition-step adapter. Aggregates are stored in `docs/aggregated_results/CLUTRR_ROBERTA_UNIFIED_CORE_REGRESSION.md/json` and `docs/aggregated_results/UNIFIED_CORE_PROP_REGRESSION.md/json`. The check shows no material performance degradation compared with the pre-core implementation under matched settings.

### Unified Core Regression Check

This check was run after refactoring TRUA into a shared reasoning core plus task-specific adapters. It is a regression test for implementation safety, not a new method variant. The intended method name remains **TRUA**. The previous pre-core implementation remains recoverable from git history; the current code path should be treated as the maintained implementation.

| Dataset / setting | Backbone | Metric | Pre-core | Unified core | Status |
| --- | --- | --- | ---: | ---: | --- |
| CLUTRR `data_089907f8` | RoBERTa TRUA | Best overall | `0.5567 +/- 0.0311` | `0.5718 +/- 0.0199` | no degradation |
| CLUTRR `data_089907f8` | RoBERTa TRUA | Best long-hop >=6 | `0.3904 +/- 0.0602` | `0.4189 +/- 0.0409` | no degradation |
| ProofWriter depth-3 | DeBERTa TRUA-Prop | Accuracy | `0.8740 +/- 0.1258` | `0.8749 +/- 0.1265` | no degradation |
| ProofWriter depth-5 | DeBERTa TRUA-Prop | Accuracy | `0.8068 +/- 0.0787` | `0.8081 +/- 0.0799` | no degradation |
| ProofWriter depth-5 | DeBERTa TRUA-Prop | Trace@1 | `0.5896 +/- 0.3034` | `0.6039 +/- 0.3151` | no degradation |
| RuleTaker raw-QDep test | DeBERTa TRUA-Prop | Accuracy | `0.7783 +/- 0.1353` | `0.7783 +/- 0.1353` | unchanged |
| RuleTaker raw-QDep test | DeBERTa TRUA-Prop | Trace@1 | `0.7444 +/- 0.2414` | `0.7444 +/- 0.2414` | unchanged |
| PrOntoQA-OOD | DeBERTa TRUA-Prop | Accuracy | `1.0000 +/- 0.0000` | `1.0000 +/- 0.0000` | unchanged |
| PrOntoQA-OOD | DeBERTa TRUA-Prop | Trace@1 | `0.5378 +/- 0.1482` | `0.5178 +/- 0.1343` | small variance-range change |

Interpretation for engineering record: the unified core is safe to keep as the maintained implementation. It preserves the public TRUA method identity while reducing duplicate implementations between entity-path and proposition-step settings.

### Core-Final Targeted Rerun Queue

Started on **2026-06-02** to refresh the main paper TRUA rows under the maintained unified-core implementation without rerunning all historical ablations. This is a targeted final-results queue, not a new method variant.

- **Run root:** `/vepfs/tsra_outputs/core_final_rerun/latest`.
- **Concrete run directory at launch:** `/vepfs/tsra_outputs/core_final_rerun/core_final_rerun_20260602_094419`.
- **Launcher:** `scripts/run_core_final_rerun_queue.sh`.
- **Default GPUs:** `0,1,2,3,4,5,7`; GPU6 was left for the ongoing Qwen3-8B QLoRA CLUTRR seed42 run.
- **Total tasks:** 30 completed; 0 failed.
- **Finished:** 2026-06-03 07:51 UTC.
- **Aggregated outputs:** `docs/aggregated_results/CORE_FINAL_CLUTRR_RERUN.md/json` and `docs/aggregated_results/CORE_FINAL_PROP_RERUN.md/json`.

Task scope:

| Group | Tasks |
| --- | --- |
| CLUTRR `data_089907f8` | DeBERTa TRUA seeds `0/1/42`; DeBERTa-v3 TRUA seeds `0/1/42` |
| CLUTRR `data_db9b8f04` | DeBERTa-v3 TRUA seeds `0/1/42` |
| ProofWriter | BERT TRUA-Prop seeds `0/1/42`; RoBERTa TRUA-Prop seeds `0/1/42` |
| RuleTaker GFaiR split | BERT/RoBERTa/DeBERTa TRUA-Prop seeds `0/1/42` |
| PrOntoQA-OOD | BERT/RoBERTa TRUA-Prop seeds `0/1/42` |

Reason for not rerunning every historical row: the completed unified-core regression already showed no material degradation on all four datasets. This queue refreshes the rows most likely to be cited in the main paper while avoiding a full rerun of all label-only, no-consistency, external-baseline, and diagnostic experiments.

### Completed TRUA Main Results

#### CLUTRR `data_089907f8`

This is the primary CLUTRR split used throughout TRUA, with 2/3-hop training. This table replaces the older draft same-backbone CLUTRR table whose configuration was not reliable. In this CLUTRR table, `label-only` means the TRUA architecture trained with final-label CE only; it is not the plain backbone classifier baseline.

| Backbone | Variant | Seeds | Best Overall | Best Short | Best Long >=6 | Final Overall | Final Long >=6 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BERT | label-only | 3 | 0.3685 +/- 0.0149 | 0.5996 +/- 0.0150 | 0.2727 +/- 0.0278 | 0.3505 +/- 0.0277 | 0.2433 +/- 0.0508 |
| BERT | tsra | 3 | 0.4282 +/- 0.0315 | 0.6266 +/- 0.0117 | 0.3066 +/- 0.0615 | 0.3880 +/- 0.0214 | 0.2540 +/- 0.0377 |
| RoBERTa | label-only | 3 | 0.4887 +/- 0.0482 | 0.6580 +/- 0.0327 | 0.3779 +/- 0.0449 | 0.4593 +/- 0.0547 | 0.3235 +/- 0.0819 |
| RoBERTa | tsra | 3 | 0.5512 +/- 0.0176 | 0.7327 +/- 0.0312 | 0.3930 +/- 0.0334 | 0.5137 +/- 0.0219 | 0.3432 +/- 0.0376 |
| RoBERTa | no-consistency | 3 | 0.5785 +/- 0.0154 | 0.7587 +/- 0.0179 | 0.4127 +/- 0.0041 | 0.5398 +/- 0.0394 | 0.3824 +/- 0.0490 |
| DeBERTa | label-only | 3 | 0.4706 +/- 0.0298 | 0.6602 +/- 0.0276 | 0.3414 +/- 0.0293 | 0.4171 +/- 0.0146 | 0.2719 +/- 0.0257 |
| DeBERTa | tsra | 3 | 0.6262 +/- 0.0214 | 0.8117 +/- 0.0181 | 0.4198 +/- 0.0335 | 0.5777 +/- 0.0083 | 0.3654 +/- 0.0147 |
| DeBERTa | no-consistency | 3 | 0.6222 +/- 0.0023 | 0.8214 +/- 0.0056 | 0.4109 +/- 0.0270 | 0.6073 +/- 0.0123 | 0.3779 +/- 0.0082 |
| DeBERTa-v3 | label-only | 3 | 0.6617 +/- 0.0419 | 0.7521 +/- 0.0414 | 0.5294 +/- 0.0520 | 0.6227 +/- 0.0551 | 0.4893 +/- 0.0468 |
| DeBERTa-v3 | tsra | 3 | 0.6786 +/- 0.0368 | 0.8095 +/- 0.0199 | 0.5169 +/- 0.0622 | 0.6693 +/- 0.0303 | 0.4974 +/- 0.0578 |
| ModernBERT | label-only | 3 | 0.4424 +/- 0.0114 | 0.6786 +/- 0.0203 | 0.2986 +/- 0.0137 | 0.3965 +/- 0.0294 | 0.2424 +/- 0.0310 |
| ModernBERT | tsra | 3 | 0.5529 +/- 0.0234 | 0.7370 +/- 0.0234 | 0.3859 +/- 0.0147 | 0.5433 +/- 0.0201 | 0.3788 +/- 0.0319 |

Notes: in this table, BERT, DeBERTa, RoBERTa, and ModernBERT have higher `Best Long >=6` in the `tsra` row than in the `label-only` row. DeBERTa-v3 has higher `Best Overall` in the `tsra` row and lower `Best Long >=6` than its `label-only` row. RoBERTa `no-consistency` has higher `Best Overall` and `Best Long >=6` than RoBERTa `tsra`. These rows are TRUA-architecture ablations, not plain Transformer classifier baselines.

#### CLUTRR True Vanilla Classifier Audit

This audit uses the plain `clutrr.cli.baseline` entry point: story + query input, encoder + linear classifier, final-label CE only. It does not use entity spans, TRUA relation attention, trace/path supervision, or consistency losses.

| Backbone | Seeds | Best Overall | Best Short | Best Long >=6 | Final Overall | Final Long >=6 |
|---|---:|---:|---:|---:|---:|---:|
| BERT | 3 | 0.2874 +/- 0.0103 | 0.9697 +/- 0.0040 | 0.1946 +/- 0.0254 | 0.2542 +/- 0.0232 | 0.1445 +/- 0.0323 |
| RoBERTa | 3 | 0.3249 +/- 0.0068 | 0.9650 +/- 0.0070 | 0.2223 +/- 0.0054 | 0.2880 +/- 0.0123 | 0.1732 +/- 0.0086 |
| DeBERTa | 3 | 0.3438 +/- 0.0186 | 0.9487 +/- 0.0176 | 0.2415 +/- 0.0122 | 0.2821 +/- 0.0229 | 0.1685 +/- 0.0265 |
| DeBERTa-v3 | 3 | 0.3610 +/- 0.0465 | 0.9580 +/- 0.0121 | 0.2426 +/- 0.0750 | 0.3310 +/- 0.0487 | 0.2222 +/- 0.0625 |
| ModernBERT | 3 | 0.2958 +/- 0.0263 | 0.9301 +/- 0.0210 | 0.1873 +/- 0.0235 | 0.2822 +/- 0.0201 | 0.1690 +/- 0.0071 |

Notes: this audit uses the plain classifier entry point and is recorded separately from the TRUA-architecture `label-only` rows above. The `Best Overall` values in the plain classifier audit are lower than the corresponding TRUA-architecture `label-only` values for the checked backbones.

#### CLUTRR `data_db9b8f04` 2/3/4-Hop Train Check

This follow-up trains on 2/3/4-hop examples and tests long-hop generalization. As above, `label-only` means the TRUA architecture with trace/edge/consistency losses disabled, not a plain backbone classifier.

| Backbone | Variant | Seeds | Best Overall | Best Short | Best Long >=6 | Final Overall | Final Long >=6 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BERT | label-only | 3 | 0.4942 +/- 0.0132 | 0.6919 +/- 0.0106 | 0.3986 +/- 0.0203 | 0.4863 +/- 0.0173 | 0.3735 +/- 0.0182 |
| BERT | tsra | 3 | 0.5223 +/- 0.0130 | 0.6869 +/- 0.0175 | 0.4255 +/- 0.0095 | 0.4930 +/- 0.0361 | 0.3727 +/- 0.0541 |
| RoBERTa | label-only | 3 | 0.6292 +/- 0.0242 | 0.7798 +/- 0.0155 | 0.5444 +/- 0.0450 | 0.5722 +/- 0.0086 | 0.4849 +/- 0.0176 |
| RoBERTa | tsra | 3 | 0.7099 +/- 0.0150 | 0.8414 +/- 0.0076 | 0.6139 +/- 0.0192 | 0.6924 +/- 0.0251 | 0.5812 +/- 0.0320 |
| DeBERTa | label-only | 3 | 0.5862 +/- 0.0173 | 0.7555 +/- 0.0076 | 0.4925 +/- 0.0314 | 0.5471 +/- 0.0182 | 0.4338 +/- 0.0300 |
| DeBERTa | tsra | 3 | 0.7455 +/- 0.0167 | 0.8596 +/- 0.0177 | 0.6508 +/- 0.0181 | 0.7074 +/- 0.0373 | 0.6131 +/- 0.0553 |
| DeBERTa-v3 | label-only | 3 | 0.7449 +/- 0.0209 | 0.8081 +/- 0.0155 | 0.7010 +/- 0.0262 | 0.7115 +/- 0.0339 | 0.6340 +/- 0.0510 |
| DeBERTa-v3 | tsra | 3 | 0.7741 +/- 0.0127 | 0.8263 +/- 0.0076 | 0.7370 +/- 0.0228 | 0.6667 +/- 0.0289 | 0.5176 +/- 0.0605 |
| ModernBERT | label-only | 3 | 0.6546 +/- 0.0353 | 0.7939 +/- 0.0320 | 0.5737 +/- 0.0406 | 0.6349 +/- 0.0187 | 0.5486 +/- 0.0167 |
| ModernBERT | tsra | 3 | 0.7068 +/- 0.0289 | 0.8253 +/- 0.0206 | 0.6273 +/- 0.0442 | 0.6778 +/- 0.0482 | 0.5611 +/- 0.0804 |

Notes: in this 2/3/4-hop train split, the `tsra` row has higher `Best Long >=6` than the `label-only` row for BERT, RoBERTa, DeBERTa, DeBERTa-v3, and ModernBERT.

#### ProofWriter

Train depth is 0/1/2; test uses depth-3 and depth-5. Accuracy and trace@1 are reported separately.

| Backbone | Model | Seeds | Depth-3 Acc | Depth-5 Acc | Depth-3 Trace@1 | Depth-5 Trace@1 |
|---|---|---:|---:|---:|---:|---:|
| BERT | baseline | 3 | 0.9596 +/- 0.0029 | 0.8818 +/- 0.0086 | 0.2053 +/- 0.0237 | 0.1459 +/- 0.0061 |
| BERT | TRUA | 3 | 0.9591 +/- 0.0031 | 0.8842 +/- 0.0037 | 0.8296 +/- 0.0030 | 0.7863 +/- 0.0029 |
| RoBERTa | baseline | 3 | 0.8703 +/- 0.1225 | 0.8064 +/- 0.0783 | 0.2080 +/- 0.0046 | 0.1694 +/- 0.0178 |
| RoBERTa | TRUA | 3 | 0.9439 +/- 0.0054 | 0.8511 +/- 0.0064 | 0.8368 +/- 0.0114 | 0.7890 +/- 0.0086 |
| DeBERTa | baseline | 3 | 0.8663 +/- 0.1197 | 0.8146 +/- 0.0880 | 0.2498 +/- 0.0585 | 0.1687 +/- 0.0231 |
| DeBERTa | TRUA | 3 | 0.8740 +/- 0.1258 | 0.8068 +/- 0.0787 | 0.6197 +/- 0.3338 | 0.5896 +/- 0.3034 |

Notes: BERT and RoBERTa have higher TRUA trace@1 than baseline trace@1 at both depth-3 and depth-5. DeBERTa has high seed variance in trace@1. A separate seed-42 check produced identical DeBERTa baseline/TRUA behavior and is recorded as an audit item.

#### RuleTaker GFaiR Split

This uses the GFaiR RuleTaker-3ext-sat split. Official test depth metadata is not available in the exported test bucket, so depth grouping is reported separately in the raw-QDep check below.

| Backbone | Model | Seeds | Test Acc | Trace@1 |
|---|---|---:|---:|---:|
| BERT | baseline | 3 | 0.9624 +/- 0.0015 | 0.6090 +/- 0.0957 |
| BERT | TRUA | 3 | 0.9623 +/- 0.0026 | 0.0226 +/- 0.0015 |
| RoBERTa | baseline | 3 | 0.9559 +/- 0.0030 | 0.5065 +/- 0.1179 |
| RoBERTa | TRUA | 3 | 0.9594 +/- 0.0020 | 0.0130 +/- 0.0018 |
| DeBERTa | baseline | 3 | 0.7215 +/- 0.0000 | 0.4201 +/- 0.1834 |
| DeBERTa | TRUA | 3 | 0.9667 +/- 0.0011 | 0.0176 +/- 0.0078 |

Notes: DeBERTa has the largest baseline-to-TRUA difference in test accuracy on this split. The exported GFaiR test bucket does not preserve comparable depth/trace metadata, so trace@1 values in this table are recorded together with the raw-QDep check below.

#### RuleTaker Raw Strict QDep 1/2 Train -> 1-5 Test

This stricter raw-data setting filters by question-level proof depth (`QDep`) and records RuleTaker depth-generalization metrics.

| Backbone | Model | Seeds | Overall | QDep1 | QDep2 | QDep3 | QDep4 | QDep5 | Trace@1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeBERTa | baseline | 3 | 0.5234 +/- 0.0000 | 0.5399 +/- 0.0000 | 0.5081 +/- 0.0000 | 0.5027 +/- 0.0000 | 0.5038 +/- 0.0000 | 0.5019 +/- 0.0000 | 0.1913 +/- 0.0051 |
| DeBERTa | TRUA | 3 | 0.7783 +/- 0.1353 | 0.8188 +/- 0.2180 | 0.8463 +/- 0.1650 | 0.6414 +/- 0.1981 | 0.5263 +/- 0.1471 | 0.4944 +/- 0.2509 | 0.7444 +/- 0.2414 |

Notes: DeBERTa TRUA has higher mean overall accuracy and trace@1 than the DeBERTa baseline in this raw-QDep setting. Reported standard deviations are large for several QDep buckets.

#### PrOntoQA-OOD

The processed PrOntoQA-OOD label task is saturated, so label accuracy and trace@1 are recorded as separate metrics.

| Backbone | Model | Seeds | OOD Acc | Trace@1 |
|---|---|---:|---:|---:|
| BERT | baseline | 3 | 1.0000 +/- 0.0000 | 0.2133 +/- 0.0448 |
| BERT | TRUA | 3 | 1.0000 +/- 0.0000 | 0.4667 +/- 0.0404 |
| RoBERTa | baseline | 3 | 1.0000 +/- 0.0000 | 0.1733 +/- 0.0333 |
| RoBERTa | TRUA | 3 | 1.0000 +/- 0.0000 | 0.4856 +/- 0.0876 |
| DeBERTa | baseline | 3 | 1.0000 +/- 0.0000 | 0.2511 +/- 0.0366 |
| DeBERTa | TRUA | 3 | 1.0000 +/- 0.0000 | 0.5378 +/- 0.1482 |

Notes: OOD label accuracy is `1.0000 +/- 0.0000` for all rows in this processed split. Trace@1 is recorded separately because it varies across methods.

### Completed External Baselines

| Method | Dataset | Status | Result |
|---|---|---|---|
| EdgeTransformer | CLUTRR `data_089907f8` | completed | overall `0.809951`; short-hop `0.976191`; long-hop 6-10 `0.684677`. |
| RAT | CLUTRR `data_089907f8` | completed | overall `0.575493`; short-hop `0.976191`; long-hop 6-10 `0.348255`. |
| CREST-style counterfactual baseline | CLUTRR `data_089907f8` | completed adapted run | DeBERTa-v3, 10 epochs, seeds `0/1/42`: best overall `0.6428 +/- 0.0013`; best long-hop >=6 `0.5216 +/- 0.0059`. Official CREST code not found; this is a same-input local adaptation. |
| FaiRR end-to-end | ProofWriter | completed | answer acc `98.403099`; proof acc `97.174721`. |
| GFaiR selector2 official XLNet | RuleTaker | completed | top1 `0.984560`; top2 `0.997896`; invalid ratio `0.000597`. |
| GFaiR full official pipeline | RuleTaker | completed | proof_acc_total `0.908629`; faithful_total `0.992208`. |
| IBR | RuleTaker depth-5 | completed | QA `0.994153`; proof `0.937416`; full `0.937169`. |
| NLProofS | RuleTaker depth-3ext | completed | answer overall `0.6796`; proof overall `0.9187`. |
| Coconut | PrOntoQA-OOD | completed reference | converted-data validation acc `1.0000`; CoT match `0.0000`. |
| CODI-GPT2 official train/distill | PrOntoQA-OOD | completed small-model reference | official PrOntoQA train/distill on converted train depth 1/2 -> test depth 3/4: overall `0.8144`; depth-3 `1.0000`; depth-4 `0.7217`. |
| CODI-Llama1B official train/distill | PrOntoQA-OOD | completed reference | official PrOntoQA train/distill on converted train depth 1/2 -> test depth 3/4: overall `0.8730`; depth-3 `1.0000`; depth-4 `0.8094`. |
| Abstractor/RCA adapted | CLUTRR | completed diagnostic | 3-epoch unfrozen raw-text adapter: overall `0.1571`; short `0.4336`; long `0.1095`. |
| Dual Attention adapted | CLUTRR | completed diagnostic | 3-epoch unfrozen raw-text adapter: overall `0.2548`; short `0.9580`; long `0.1424`. |

### Same-Input Architecture Baselines

The official FaiRR/GFaiR/IBR/NLProofS pipelines use task-specific proof generation modules, selectors, symbolic conversions, or pipeline assumptions that differ from TRUA's raw-text setting. This report separates:

- **Same-input main comparisons:** same raw text + query input, final-label training only, no gold trace/proof/graph at test time, no external symbolic solver.
- **Reference comparisons:** official end-to-end proof/pipeline systems, reported PrOntoQA baselines, and structured CLUTRR graph-edge methods.

Same-input architecture baseline runner:

- **Script:** `scripts/fair_attention_prop.py`.
- **Supervisor:** `scripts/run_fair_attention_prop_supervisor.sh`.
- **Run root:** `/vepfs/tsra_outputs/fair_attention_prop/latest`.
- **Backbone:** DeBERTa-base from `/vepfs/tsra_models/hf/deberta-base`.
- **Datasets:** ProofWriter, RuleTaker raw-QDep, PrOntoQA-OOD.
- **Methods:** DAT / Dual Attention, Abstractor/RCA, and optional MAC-style compositional attention.
- **Training:** 10 epochs, seeds `0/1/42`, final-label loss only.
- **Evaluation:** label accuracy, depth/QDep grouping where available, and trace@1 as an unsupervised alignment diagnostic.

Current same-input adapter status:

| Method | Dataset | Setting | Result |
|---|---|---|---|
| DAT / Dual Attention | ProofWriter | train depth 0/1/2 -> test depth 3/5 | completed; depth-3 acc `0.6630 +/- 0.3436`, depth-5 acc `0.6390 +/- 0.3111`, depth-5 trace@1 `0.1740 +/- 0.0634` |
| Abstractor/RCA | ProofWriter | train depth 0/1/2 -> test depth 3/5 | completed; depth-3 acc `0.8761 +/- 0.1275`, depth-5 acc `0.8221 +/- 0.0930`, depth-5 trace@1 `0.1803 +/- 0.0495` |
| DAT / Dual Attention | RuleTaker raw-QDep | train depth 1/2, QDep 1/2 -> test QDep 1-5 | completed; overall `0.5583 +/- 0.0605`, QDep5 `0.6121 +/- 0.1909`, trace@1 `0.1624 +/- 0.0038` |
| Abstractor/RCA | RuleTaker raw-QDep | train depth 1/2, QDep 1/2 -> test QDep 1-5 | completed; overall `0.5603 +/- 0.0639`, QDep5 `0.6203 +/- 0.2051`, trace@1 `0.2082 +/- 0.0588` |
| DAT / Dual Attention | PrOntoQA-OOD | train 1/2-hop ProofsOnly -> test 3/4-hop and OOD composed | completed; OOD acc `1.0000 +/- 0.0000`, trace@1 `0.2400 +/- 0.0850` |
| Abstractor/RCA | PrOntoQA-OOD | train 1/2-hop ProofsOnly -> test 3/4-hop and OOD composed | completed; OOD acc `1.0000 +/- 0.0000`, trace@1 `0.2044 +/- 0.0158` |

Source papers for these adapters:

- DAT / Dual Attention follows `Disentangling and Integrating Relational and Sensory Information in Transformer Architectures` (`https://arxiv.org/abs/2405.16727`), which explicitly adds sensory and relational attention mechanisms to Transformers.
- Abstractor/RCA follows `Abstractors and Relational Cross-Attention: An Inductive Bias for Explicit Relational Reasoning in Transformers` (`https://proceedings.iclr.cc/paper_files/paper/2024/hash/6b070264f20d25fbd1488de8ea51575f-Abstract-Conference.html`).
- MAC-style attention is recorded as a compositional-attention baseline from an older method family.

### Recent Method Scan Beyond DAT/Abstractor

The following methods were scanned or run as additional external comparisons. The table records target datasets, relevance, comparability notes, and current status.

| Method | Year | Best-fit TRUA dataset(s) | Relevance | Comparability note | Current status |
|---|---:|---|---|---|---|
| AAI: Attention-Aware Intervention | 2026 | ProofWriter, PrOntoQA-OOD | Modulates selected attention heads for logical reasoning and reports no external solver or interactive pipeline. | Instruction-LLM setting rather than encoder-only DeBERTa. | ProofWriter run completed with Qwen3-32B. |
| Symbolic-Aided CoT | 2025 | ProofWriter, PrOntoQA-OOD | Non-iterative logical prompting baseline on ProofWriter/PrOntoQA. | Prompting/LLM setting; uses symbolic structure in prompt. | Not separately run; AAI uses a related prompting setup. |
| LoGiPT | NAACL Findings 2024 | ProofWriter, PrOntoQA-OOD | Trains LMs to internalize solver-like deduction; Hugging Face checkpoints exist for ProofWriter/PrOntoQA. | Decoder LM, not same-backbone; trained from solver-derived instructions. | ProofWriter raw adapter and Logic-LM-style runs completed. |
| Coconut | 2024 | PrOntoQA-OOD | Latent/continuous thought baseline; official code supports PrOntoQA and does not expose gold proof at test time. | Decoder LLM + CoT/latent curriculum, not same-backbone. | Converted PrOntoQA-OOD validation run completed. |
| CODI | EMNLP 2025 | PrOntoQA-OOD | Compresses CoT into continuous latent states via self-distillation; directly relevant to internal reasoning without explicit test-time CoT. | Main reported tasks are broader reasoning benchmarks; our task-adapted PrOntoQA runs use CODI's official training/distillation branch on GPT-2 and Llama-3.2-1B-Instruct. Compare as latent-reasoning references rather than same-backbone encoder baselines. | GPT-2 and Llama1B settings completed. |
| PCT: Probabilistic Constraint Training | EACL Findings 2024 | RuleTaker-style | End-to-end Transformer training objective that uses logical/probabilistic constraints during training but not inference, close in spirit to train-time supervision. | Official benchmark is RuleTaker-Pro/probabilistic rules, not exactly our RuleTaker raw-QDep split. | Good RuleTaker-family training-objective baseline if software zip is easy to adapt. |
| QK-score / Query-Key Alignment | EMNLP 2025 | PrOntoQA-OOD | Uses query-key alignment inside Transformer attention heads to diagnose logical consistency; reports robustness under distractors and depth. | More diagnostic than a trained classifier; not a direct accuracy baseline. | Not run; literature/diagnostic candidate only. |

### Recent External Baseline Queue

The four new methods requested after the 2026-05-29 literature scan are tracked in a separate queue. AAI, LoGiPT, Coconut, and CODI now all have completed results; CODI is reported using the task-adapted PrOntoQA train/distill setting at both GPT-2 and Llama1B scales.

- **Queue supervisor:** `scripts/run_recent_external_baselines_supervisor.py`.
- **Prepared data view:** `scripts/prepare_recent_external_data.py`.
- **Run root:** `/vepfs/tsra_outputs/recent_external_baselines/latest`.
- **Converted PrOntoQA-OOD data:** `/vepfs/tsra_outputs/recent_external_data/prontoqa_ood/`.
- **Policy:** tasks are launched only when their code/checkpoint/adapter preflight passes and enough GPUs are free; otherwise they stay in `.pending`, not `.failed`.

Queued order:

| Order | Method | Dataset | Status / gate |
|---:|---|---|---|
| 1 | AAI | ProofWriter | Completed. Official code cloned to `external_baselines/AAI`; Qwen3-32B is available through symlink `/vepfs/tsra_models/hf/Qwen3-32B -> /tos/lxh/models/qwen3_32` (`62G`, 17 safetensor shards). After adding a local fallback for missing `lightning.seed_everything`, the ProofWriter run finished: no-intervention acc `0.8350` (`501/600`), attention-aware intervention acc `0.8267` (`496/600`). |
| 2 | LoGiPT | ProofWriter | Completed. Official checkpoint `jzfeng/LoGiPT-CodeLlama-13b-Instruct-hf-proofwriter` downloaded via `HF_ENDPOINT=https://hf-mirror.com` to `/vepfs/tsra_models/hf/logipt/LoGiPT-CodeLlama-13b-Instruct-hf-proofwriter`. Option 1 TRUA raw-depth adapter result: macro depth accuracy `0.6515`, depth-3 `0.6466`, depth-5 `0.6564`. Option 2 reported-style Logic-LM ProofWriter test result: acc `0.4517` (`271/600`), parsed ratio `0.9350`. |
| 3 | Coconut | PrOntoQA-OOD | Completed. Official code cloned to `external_baselines/Coconut_official`; TRUA PrOntoQA-OOD converted to Coconut `question/answer/steps` JSON and config. Result: validation acc `1.0000`, CoT match `0.0000`. |
| 4 | CODI | PrOntoQA-OOD | Completed. CODI's official PrOntoQA training/distillation branch was run with GPT-2+LoRA and Llama-3.2-1B-Instruct+LoRA on converted train depth 1/2 and evaluated on test depth 3/4. GPT-2 result file `/vepfs/tsra_outputs/recent_external_baselines/latest/results/codi_prontoqa_train_distill_test.json` reports overall `0.8144`, depth-3 `1.0000`, depth-4 `0.7217`. Llama1B result file `/vepfs/tsra_outputs/recent_external_baselines/latest/results/codi_prontoqa_llama1b_train_distill_test.json` reports overall `0.8730`, depth-3 `1.0000`, depth-4 `0.8094`. |

### Report File Policy

- **Final report file:** `/root/TRUA/EXPERIMENT_REPORT.md`.
- **Setup notes file:** `/root/TRUA/README_EXPERIMENTS.md`.
- Do not use old local copies such as `EXPERIMENT_REPORT.remote.md` or `TRUA_FINAL_EXPERIMENT_REPORT.md`; those were local intermediate artifacts and are not present in the cleaned remote repo.

## 0.2 Removed Mid-Run Notes

The previous 2026-05-21 mid-run notes have been removed from the current report body because they described jobs that were still running at that time. The latest status in Section 0 supersedes those intermediate observations.

## 0.3 Cross-Dataset Applicability of External Baselines

The external baselines are not uniformly plug-and-play across all four TRUA datasets. Their official code uses dataset-specific input representations and supervision formats.

| Method | Official / Natural Dataset | CLUTRR | ProofWriter | RuleTaker | PrOntoQA-OOD | Recorded Use |
|---|---|---:|---:|---:|---:|---|
| EdgeTransformer | CLUTRR, CFQ, COGS | yes, completed | no direct support | no direct support | no direct support | CLUTRR structured graph-edge reference. |
| RAT | CLUTRR relation-aware baseline | yes, completed | no direct support | no direct support | no direct support | CLUTRR relation-aware Transformer baseline. |
| CREST | shortcut-learning mitigation with counterfactual reasoning | local same-input adaptation completed | not run | not run | not run | CLUTRR counterfactual training baseline; official public code not found. |
| FaiRR | ProofWriter | possible only with graph/proof conversion | yes, completed | not official in current repo | not direct | ProofWriter full end-to-end baseline; RuleTaker would require local engineering rather than official reproduction. |
| GFaiR | RuleTaker variants, Hard RuleTaker, RuleTaker-E, NL satisfiability | no | not official | yes, completed | not direct | RuleTaker-family baseline; selector2 and full official pipeline results are available. |
| NLProofS | EntailmentBank, ProofWriter/RuleTaker-style proof generation | no | possible | yes, completed | not direct | RuleTaker proof-generation baseline; final answer/proof metrics are available. |
| IBR | RuleTaker depth-5 / ParaRules-style iterative reasoning | no | not direct | yes, completed | not direct | RuleTaker proof-reasoning baseline; depth-5 test result is available. |
| Abstractor/RCA | synthetic relational reasoning tasks | local raw-text adapter only | local raw-text adapter only | local raw-text adapter only | local raw-text adapter only | Local same-input diagnostic adapter. |
| DAT | relational/dual-attention architecture | local raw-text adapter only | local raw-text adapter only | local raw-text adapter only | local raw-text adapter only | Local adapter results are recorded above. |

Applicability notes:

- EdgeTransformer/RAT rows use structured CLUTRR graph-edge input.
- FaiRR/GFaiR/IBR/NLProofS rows are official proof-system or pipeline settings.
- DAT and Abstractor/RCA rows use local same-input raw-text adapters via `scripts/fair_attention_prop.py`.
- In the processed PrOntoQA-OOD split, binary label accuracy is saturated; trace/proof-step metrics are recorded separately.

## 2. Dataset Status

### CLUTRR

- **Role:** main entity-relation path reasoning dataset.
- **Status:** already present in the repository.
- **Required split:** `data/data_089907f8`.
- **Train split:** `1.2,1.3_train.csv`, containing 2-hop and 3-hop examples.
- **Test split:** `1.2_test.csv` through `1.10_test.csv`.
- **Follow-up split:** `data/data_db9b8f04`, with `1.2,1.3,1.4_train.csv` for 2/3/4-hop training. The merged CLUTRR backbone sweep is complete under `/vepfs/tsra_outputs/clutrr_backbone_sweep/latest`.
- **Depth/hop definition:** length of the query-subject to query-object entity path.
- **Trace definition:** entity/relation path from query subject to query object.
- **Evaluation:** overall accuracy, short-hop accuracy, long-hop accuracy, per-hop accuracy.
- **Current status:** ready; same-backbone CLUTRR reruns are complete for the audited `data_089907f8` and `data_db9b8f04` splits, and external CLUTRR baselines are available.

### ProofWriter

- **Role:** proof-chain / proof-tree textual reasoning dataset.
- **Official source:** `https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip`.
- **Dev-machine path:** `/root/TRUA/data/proofwriter/raw/proofwriter-dataset-V2020.12.3`.
- **Local staging path used:** `/private/tmp/tsra_data/proofwriter-dataset-V2020.12.3.zip`.
- **TOS target for reproducibility:** `tos://c20250504/wy/data/proofwriter/proofwriter-dataset-V2020.12.3.zip`.
- **Format:** official OWA/CWA JSONL files, depth-0 through depth-5, with proof/proof-depth annotations.
- **Shallow/deep split:** train depth 0/1/2; test depth-3 and depth-5.
- **Trace definition:** proof tree linearized into sentence-level proof sequence supervision.
- **Current status:** ready; TRUA-Prop and FaiRR component experiments have been run.

### RuleTaker

- **Role:** natural-language facts/rules + query deductive reasoning dataset.
- **Status:** already present in the repository.
- **Native path:** `/root/TRUA/data/rule-reasoning-dataset-V2020.2.5.0/original`.
- **GFaiR data path:** `/root/TRUA/external_baselines/GFaiR/data/ruletaker_3ext_sat`.
- **Available settings:** depth-0, depth-1, depth-2, depth-3, depth-3ext, depth-5, hard RuleTaker variants.
- **Trace definition:** fact/rule/proposition proof-step sequence from provided proof metadata.
- **Current split used:** GFaiR RuleTaker-3ext-sat train/dev/test with `*_withmidprove.pkl`.
- **Strict raw-depth follow-up:** a new `ruletaker_raw` loader filters by question-level `QDep`, not only by directory-level `depth-*`. The completed check trains on QDep `1,2` from raw `depth-1/depth-2` train files and evaluates QDep `1,2,3,4,5` from raw depth `1,2,3,5` dev/test files.
- **Current status:** ready; TRUA-Prop and GFaiR component experiments have been run.

### PrOntoQA-OOD

- **Role:** OOD systematic/compositional generalization dataset.
- **Official source/code:** `https://github.com/asaparov/prontoqa`.
- **Raw path:** `/root/TRUA/data/prontoqa_ood/raw/prontoqa`.
- **Generated OOD data path:** `/root/TRUA/data/prontoqa_ood/processed/generated_ood_data`.
- **Official model-output path:** `/root/TRUA/data/prontoqa_ood/processed/model_outputs_ood/flan-t5/latest`.
- **Split used:** train on shallower ProofsOnly examples; test on deeper/OOD generated examples such as `4hop_OOD_Composed_random_noadj`.
- **Trace/proof definition:** proposition-level proof sequence.
- **Metric note:** many generated files are proof-generation style positive examples, so binary classification accuracy is saturated. Proof-step/trace metrics and official OOD reference results are recorded separately.
- **Current status:** ready; TRUA-Prop coverage run and official FLAN-T5 output analysis have been run.

## 3. Existing TRUA Results

The repository already contains native CLUTRR TRUA code and logs. The most relevant existing result uses the required `data_089907f8` split.

### CLUTRR Existing TRUA Logs

| Model/log | Overall | Short-hop | Long-hop >=6 | Notes |
|---|---:|---:|---:|---|
| TRUA-DeBERTa `rollback_a90_default_deberta_5ep.log` | 0.6370 | 0.8052 | 0.4332 | best existing CLUTRR TRUA result found |
| TRUA-RoBERTa seed0 | 0.5218 | 0.7175 | 0.3503 | existing run |
| TRUA-RoBERTa seed1 | 0.5401 | 0.7565 | 0.3690 | existing run |
| TRUA-RoBERTa seed123 | 0.4904 | 0.7208 | 0.2995 | existing run |
| TRUA-RoBERTa 10ep config | 0.5497 | 0.7143 | 0.4037 | existing run |

Notes:

- The listed logs are existing repository results on CLUTRR `data_089907f8`.
- The DeBERTa TRUA row records long-hop accuracy `0.4332` with 2/3-hop training.
- Later sections contain rerun/audit tables that are recorded separately from these earlier log entries.

## 4. External Baselines

### Edge Transformer on CLUTRR

- **Paper/code family:** Edge Transformer for CLUTRR.
- **Official code:** `https://github.com/bergen/EdgeTransformer`.
- **Clone path:** `/root/TRUA/external_baselines/EdgeTransformer`.
- **Dataset:** CLUTRR `data_089907f8`.
- **Status:** reproduced on the corrected split.
- **Output path:** `/vepfs/tsra_outputs/official_external/latest_edge_rat_clutrr_089907f8`.

| Model | Overall | Short-hop | Long-hop >=6 | Notes |
|---|---:|---:|---:|---|
| Edge Transformer | 0.8100 | 0.9762 | 0.6847 | structured graph-edge input |

Per-hop accuracy:

| Hop | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Accuracy | 1.0000 | 0.9524 | 1.0000 | 0.9138 | 0.8318 | 0.7986 | 0.6333 | 0.5882 | 0.5714 |

Notes:

- Edge Transformer is recorded as a CLUTRR structured graph-edge baseline.
- It uses structured CLUTRR graph/edge information at test time.
- It is not a same-input raw-text baseline.

### CREST-style Counterfactual Baseline on CLUTRR

- **Paper:** `CREST: A causal framework for mitigating shortcut learning in language models through counterfactual reasoning`, DOI `10.1016/j.ipm.2025.104418`.
- **Official code status:** no public official code was found in exact-title, DOI, PII, author, and GitHub API searches as of 2026-06-26.
- **Implementation status:** local same-input adaptation in `scripts/crest_clutrr_baseline.py`.
- **Dataset:** CLUTRR `data_089907f8`, the primary TRUA 2/3-hop train and 2-10-hop test split.
- **Backbone:** DeBERTa-v3 from `/vepfs/tsra_models/hf/deberta-v3-base`.
- **Run root:** `/vepfs/tsra_outputs/crest_clutrr/crest_clutrr_20260626_002032`.
- **Aggregated artifact:** `docs/aggregated_results/CREST_CLUTRR_COUNTERFACTUAL.md/json`.
- **Training signal:** final-label CE plus entity-renaming consistency and query-reversal counterfactual labels derived from the CLUTRR relation schema.
- **Test-time input:** raw story + query only; no gold trace/path/proof and no graph-edge input.

| Model | Seeds | Best Overall | Best Short | Best Long >=6 | Final Overall | Final Long >=6 |
|---|---:|---:|---:|---:|---:|---:|
| CREST-style DeBERTa-v3 | 3 | `0.6428 +/- 0.0013` | `0.9394 +/- 0.0107` | `0.5216 +/- 0.0059` | `0.6082 +/- 0.0277` | `0.4690 +/- 0.0406` |

Per-hop best accuracy:

| Hop | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Accuracy | `0.9649 +/- 0.0402` | `0.9302 +/- 0.0055` | `0.7702 +/- 0.0299` | `0.7050 +/- 0.0327` | `0.5826 +/- 0.0235` | `0.5394 +/- 0.0145` | `0.4289 +/- 0.0204` | `0.5658 +/- 0.0463` | `0.5182 +/- 0.0540` |

Comparison notes:

- This adapted counterfactual baseline is much stronger than the true vanilla DeBERTa-v3 classifier on the same split (`0.3610 +/- 0.0465` overall; `0.2426 +/- 0.0750` long-hop).
- Its best long-hop value is numerically close to TRUA-DeBERTa-v3 (`0.5169 +/- 0.0622`), while its best overall value is lower than TRUA-DeBERTa-v3 (`0.6786 +/- 0.0368`).
- The run is a local same-input adaptation, not an official CREST reproduction. The query-reversal labels use task-specific CLUTRR relation-schema knowledge during training.

### Shortcut-Reasoning Diagnostic Pilot on CLUTRR

- **Diagnostic paper:** `Discovering Highly Influential Shortcut Reasoning: An Automated Template-Free Approach`, Findings EMNLP 2023, DOI `10.18653/v1/2023.findings-emnlp.424`.
- **Official code:** `https://github.com/homoscribens/shortcut_reasoning.git`.
- **Local run type:** pilot token-occlusion approximation of the paper's IG/input-reduction diagnostic; not an official reproduction.
- **Run root:** `/vepfs/tsra_outputs/clutrr_shortcut_diagnostic/clutrr_shortcut_diag_20260626_011004`.
- **Aggregated artifact:** `docs/aggregated_results/CLUTRR_SHORTCUT_DIAGNOSTIC_PILOT.md/json`.
- **IID/OOD definition:** IID = CLUTRR short-hop test examples, 2/3-hop; OOD = CLUTRR long-hop test examples, >=6-hop.
- **Backbone / training:** DeBERTa-v3, seed `0`, 3-epoch pilot for vanilla, CREST-style, and TRUA.

| Variant | IID short acc | OOD long acc | Supported token patterns | Shortcut count | Notes |
|---|---:|---:|---:|---:|---|
| vanilla | `0.9580` | `0.1565` | 4 | 0 | Strong short/long gap, but token-level shortcut patterns are sparse. |
| CREST-style | `0.9580` | `0.4836` | 8 | 0 | Higher OOD long accuracy in this pilot; no qualifying granular token shortcut under current thresholds. |
| TRUA | `0.7435` | `0.3075` | 61 | 0 | 3-epoch diagnostic run only; not comparable to final 10-epoch TRUA rows. |

Pilot interpretation:

- This diagnostic did not identify robust granular token-level shortcuts on CLUTRR under the current token-occlusion settings.
- The result is consistent with CLUTRR's main failure mode being chain-length / relation-composition generalization rather than a stable single-token trigger.
- This pilot should not be used as a main-paper result unless replaced by a more CLUTRR-specific relation-pattern diagnostic or by a full official IG/input-reduction reproduction.

### Dual Attention Transformer on CLUTRR

- **Paper:** `Disentangling and Integrating Relational and Sensory Information in Transformer Architectures` (2024/2025).
- **Official code:** `https://github.com/Awni00/dual-attention`.
- **Clone path:** `/root/TRUA/external_baselines/dual-attention`.
- **Dataset chosen:** CLUTRR, because the method targets relational reasoning and CLUTRR is the closest entity-relation path task among the four datasets.
- **Implementation:** official PyTorch `DualAttention` module with a local CLUTRR raw-text adapter.
- **Run type:** DeBERTa encoder unfrozen, full `data_089907f8` train split, 3 epochs.
- **Output:** `/root/TRUA/outputs/external_baselines/dual_attention_clutrr_unfrozen_3ep.json`.

| Model | Train | Epochs | Overall | Short-hop | Long-hop >=6 | Training loss |
|---|---:|---:|---:|---:|---:|---|
| Dual Attention adapted | 10094 | 3 | 0.2548 | 0.9580 | 0.1424 | 2.0561 -> 0.1743 |

Per-hop accuracy:

| Hop | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Accuracy | 0.9737 | 0.9524 | 0.2000 | 0.1494 | 0.1308 | 0.1319 | 0.1467 | 0.1345 | 0.1681 |

Notes:

- This is a local raw-text CLUTRR adapter run, not a full official reproduction.
- Short-hop accuracy is `0.9580`.
- Long-hop accuracy is `0.1424`.
- Training loss changed from `2.0561` to `0.1743`.

### Abstractors and Relational Cross-Attention on CLUTRR

- **Paper:** `Abstractors and Relational Cross-Attention: An Inductive Bias for Explicit Relational Reasoning in Transformers`, ICLR 2024.
- **Official code:** `https://github.com/Awni00/abstractor`.
- **Project page:** `https://awni.xyz/abstractor/`.
- **Dataset chosen:** CLUTRR, because it is the closest entity-relation path reasoning dataset.
- **Implementation:** local PyTorch relational cross-attention adapter for CLUTRR raw text. The official repo is TensorFlow/Keras and does not provide a CLUTRR pipeline.
- **Run type:** DeBERTa encoder unfrozen, full `data_089907f8` train split, 3 epochs.
- **Output:** `/root/TRUA/outputs/external_baselines/abstractor_rca_clutrr_unfrozen_3ep.json`.

| Model | Train | Epochs | Overall | Short-hop | Long-hop >=6 | Training loss |
|---|---:|---:|---:|---:|---:|---|
| Abstractor/RCA adapted | 10094 | 3 | 0.1571 | 0.4336 | 0.1095 | 2.3970 -> 1.0967 |

Per-hop accuracy:

| Hop | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Accuracy | 0.6316 | 0.3619 | 0.1737 | 0.0862 | 0.1028 | 0.1319 | 0.0933 | 0.1008 | 0.1176 |

Notes:

- This is a local raw-text CLUTRR adapter run, not a full official reproduction.
- Short-hop accuracy is `0.4336`.
- Long-hop accuracy is `0.1095`.
- Training loss changed from `2.3970` to `1.0967`.

### MAC-style Compositional Attention on CLUTRR

- **Related method family:** MAC / compositional attention reasoning.
- **MAC paper:** `https://arxiv.org/abs/1803.03067`.
- **CLUTRR baseline repo:** `https://github.com/koustuvsinha/clutrr-baselines`.
- **Reason for inclusion:** the official CLUTRR baseline repository includes a MAC config, and MAC is a classic attention-based compositional reasoning architecture.
- **Official blocker:** the CLUTRR baseline repo depends on old unavailable packages (`addict`, `comet_ml`, `torch_geometric`, `pytorch_pretrained_bert`).
- **Implementation:** lightweight local MAC-style query-conditioned attention runner.
- **Script:** `/root/TRUA/scripts/mac_attention_clutrr.py`.
- **Output:** `/root/TRUA/outputs/external_baselines/mac_attention_clutrr_20ep.json`.

| Model | Train | Epochs | Overall | Short-hop | Long-hop >=6 | Training loss |
|---|---:|---:|---:|---:|---:|---|
| MAC-style compositional attention adapted | 10094 | 20 | 0.2173 | 0.6573 | 0.1283 | 2.6716 -> 0.3291 |

Per-hop accuracy:

| Hop | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Accuracy | 0.6316 | 0.6667 | 0.2474 | 0.1494 | 0.1121 | 0.1528 | 0.1067 | 0.1429 | 0.1261 |

Notes:

- This is a lightweight local MAC-style runner.
- Short-hop accuracy is `0.6573`.
- Long-hop accuracy is `0.1283`.
- Training loss changed from `2.6716` to `0.3291`.

### GFaiR on RuleTaker

- **Paper:** GFaiR, LREC-COLING 2024.
- **Official code:** `https://github.com/spirit-moon-fly/GFaiR`.
- **Paper link:** `https://aclanthology.org/2024.lrec-main.1436.pdf`.
- **Clone path:** `/root/TRUA/external_baselines/GFaiR`.
- **Dataset chosen:** RuleTaker, because GFaiR is built around RuleTaker-3ext-sat, depth-5, and hard RuleTaker.
- **Official checkpoint issue:** the repo hard-coded local `../../model/xlnet` and `../../model/T5` paths. This was patched to use `/vepfs/tsra_models/hf/xlnet-large-cased` and `/vepfs/tsra_models/hf/t5-large`.
- **Current status:** selector2 and full official RuleTaker inference completed.
- **Selector2 output:** `/vepfs/tsra_outputs/official_external/latest_gfair_selector2_test_retry/test_result_recording.txt`.
- **Full inference output:** `/vepfs/tsra_outputs/official_external/gfair_full_20260521_085403/full_inference_ruletaker_3ext_retry_after_reboot_bs8/test_result_recording.txt`.

| Model | Train | Eval | Result | Notes |
|---|---:|---:|---|---|
| GFaiR selector2 official XLNet | official | test | top1 `0.984560`; top2 `0.997896`; invalid `0.000597` | official component |
| GFaiR full official pipeline | official | RuleTaker test | proof_acc_total `0.908629`; faithful_total `0.992208` | official pipeline |
| GFaiR Selector2 adapted DeBERTa | 1000 | dev/test 500 | dev top1_acc 0.914; test top1_acc 0.886 | component-level post-selector result |

Notes:

- Official GFaiR selector2 and full-pipeline results are recorded separately from the adapted DeBERTa selector result.
- The adapted DeBERTa selector row is a local component-level result.
- GFaiR is a RuleTaker proof-reasoning pipeline setting, not a same-backbone encoder-only setting.

### FaiRR on ProofWriter

- **Paper:** `FaiRR: Faithful and Robust Deductive Reasoning over Natural Language`, ACL 2022.
- **Official code:** `https://github.com/INK-USC/FaiRR`.
- **Paper link:** `https://aclanthology.org/2022.acl-long.77/`.
- **Clone path:** `/root/TRUA/external_baselines/FaiRR`.
- **Dataset chosen:** ProofWriter, because FaiRR decomposes natural-language reasoning into rule selection, fact selection, and reasoning.
- **Current status:** official end-to-end ProofWriter run completed; older component-level runs are retained as diagnostics.

| Component | Train | Eval | Result | Output |
|---|---:|---:|---|---|
| FaiRR end-to-end | official | test | answer acc `98.403099`; proof acc `97.174721` | `/vepfs/tsra_outputs/official_external/latest_fairr_e2e_retry` |
| FaiRR fact-selector adapted DeBERTa | 2000 | 1000 | dev top1_acc 0.981; test top1_acc 0.988; test token_acc 0.9956 | `outputs/external_baselines/fairr_fact_deberta_2k.json` |

Notes:

- Official FaiRR end-to-end results and adapted fact-selector component results are recorded separately.
- The official rule-selector path was also verified during the run.

### PrOntoQA-OOD Official Reference

- **Official code/data:** `https://github.com/asaparov/prontoqa`.
- **Paper:** `https://arxiv.org/abs/2305.15269`.
- **Analyzer target:** `4hop_OOD_Composed_random_noadj.json`.

| Reference | Setting | Strict proof correctness | Relaxed/non-atomic correctness | Notes |
|---|---|---:|---:|---|
| Official FLAN-T5 output analysis | 4-hop OOD composed random no-adj | 0.01 | 0.35 | official output reference |

Notes:

- The row above is an official PrOntoQA-OOD reference output analysis.
- PrOntoQA-OOD task format differs from entity-path CLUTRR and proof-pipeline settings.

## 5. TRUA Coverage Beyond CLUTRR

In addition to the native CLUTRR TRUA pipeline, the repository now has a shared TRUA-Prop runner for ProofWriter, RuleTaker, and PrOntoQA-OOD:

- **Script:** `/root/TRUA/scripts/transformer_trua_prop.py`.
- **Backbones run formally:** DeBERTa, RoBERTa, and BERT.
- **Design:** same encoder for baseline and TRUA; trace labels supervise sentence-selection logits during training only.
- **Test-time policy:** raw context/query only; no gold trace/path/proof is provided at inference.

Early small-limit TRUA-Prop runs have been superseded by formal 10-epoch runs. The tables below are the current results to cite.

### ProofWriter Formal TRUA-Prop

| Backbone | Model | Seed | Depth-3 Acc | Depth-5 Acc | Depth-3 Trace@1 | Depth-5 Trace@1 | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| BERT | baseline | 0 | 0.9629 | 0.8899 | 0.1795 | 0.1392 | formal 10ep |
| BERT | baseline | 1 | 0.9577 | 0.8728 | 0.2106 | 0.1512 | formal 10ep |
| BERT | TRUA | 0 | 0.9635 | 0.8952 | 0.8330 | 0.7834 | formal 10ep |
| BERT | TRUA | 1 | 0.9591 | 0.8873 | 0.8298 | 0.7848 | formal 10ep |
| RoBERTa | baseline | 0 | 0.9406 | 0.8507 | 0.2112 | 0.1876 | formal 10ep |
| RoBERTa | baseline | 1 | 0.7288 | 0.7160 | 0.2027 | 0.1521 | seed1 lower than seed0 |
| RoBERTa | TRUA | 0 | 0.9322 | 0.8386 | 0.8487 | 0.7966 | formal 10ep |
| RoBERTa | TRUA | 1 | 0.9565 | 0.8698 | 0.8423 | 0.7928 | formal 10ep |
| DeBERTa | baseline | 0 | 0.9473 | 0.8855 | 0.3145 | 0.1937 | formal 10ep |
| DeBERTa | baseline | 1 | 0.9228 | 0.8422 | 0.2342 | 0.1642 | seed-1 rerun existed; previously omitted from top summary |
| DeBERTa | TRUA | 0 | 0.9479 | 0.8550 | 0.7875 | 0.7433 | formal 10ep |
| DeBERTa | TRUA | 1 | 0.9454 | 0.8495 | 0.8365 | 0.7854 | seed-1 rerun |

### RuleTaker Formal TRUA-Prop

| Backbone | Model | Seed | Test Acc | Dev Acc | Dev Trace@1 | Notes |
|---|---|---:|---:|---:|---:|---|
| BERT | baseline | 0 | 0.9608 | 0.9560 | 0.1553 | formal 10ep |
| BERT | baseline | 1 | 0.9638 | 0.9605 | 0.1681 | formal 10ep |
| BERT | TRUA | 0 | 0.9637 | 0.9582 | 0.8386 | formal 10ep |
| BERT | TRUA | 1 | 0.9630 | 0.9609 | 0.8387 | formal 10ep |
| RoBERTa | baseline | 0 | 0.9564 | 0.9518 | 0.1151 | formal 10ep |
| RoBERTa | baseline | 1 | 0.9587 | 0.9542 | 0.1726 | formal 10ep |
| RoBERTa | TRUA | 0 | 0.9618 | 0.9581 | 0.8356 | formal 10ep |
| RoBERTa | TRUA | 1 | 0.9604 | 0.9582 | 0.8419 | formal 10ep |
| DeBERTa | baseline | 0 | 0.7215 | 0.7154 | 0.1732 | formal 10ep |
| DeBERTa | baseline | 1 | 0.7215 | 0.7154 | 0.1627 | seed-1 rerun |
| DeBERTa | TRUA | 0 | 0.9646 | 0.9599 | 0.2444 | formal 10ep |
| DeBERTa | TRUA | 1 | 0.9676 | 0.9637 | 0.8424 | seed-1 rerun |

### PrOntoQA-OOD Formal TRUA-Prop

| Backbone | Model | Seed | OOD Label Acc | Trace@1 | Notes |
|---|---|---:|---:|---:|---|
| BERT | baseline | 0 | 1.0000 | 0.1767 | label acc saturated |
| BERT | baseline | 1 | 1.0000 | 0.2000 | label acc saturated |
| BERT | TRUA | 0 | 1.0000 | 0.5100 | trace@1 above paired baseline |
| BERT | TRUA | 1 | 1.0000 | 0.4300 | trace@1 above paired baseline |
| RoBERTa | baseline | 0 | 1.0000 | 0.1733 | label acc saturated |
| RoBERTa | baseline | 1 | 1.0000 | 0.2067 | label acc saturated |
| RoBERTa | TRUA | 0 | 1.0000 | 0.3867 | trace@1 above paired baseline |
| RoBERTa | TRUA | 1 | 1.0000 | 0.5167 | trace@1 above paired baseline |
| DeBERTa | baseline | 0 | 1.0000 | 0.2300 | label acc saturated |
| DeBERTa | baseline | 1 | 1.0000 | 0.2933 | label acc saturated |
| DeBERTa | TRUA | 0 | 1.0000 | 0.6233 | trace@1 above paired baseline |
| DeBERTa | TRUA | 1 | 1.0000 | 0.3667 | trace@1 above paired baseline |

Notes:

- ProofWriter BERT TRUA has higher depth-5 accuracy than BERT baseline in the listed seeds and higher trace@1.
- RuleTaker DeBERTa TRUA has higher mean accuracy than DeBERTa baseline in the GFaiR split and raw-QDep table.
- PrOntoQA-OOD label accuracy is `1.0000` in all listed same-backbone rows; trace@1 values are recorded separately.

## 6. Consolidated Results Tables

### CLUTRR Main Comparison

Same-backbone rerun, primary `data_089907f8` split. Here `label-only` is a TRUA-architecture final-label ablation, not a plain backbone classifier:

| Backbone | Variant | Seeds | Best Overall | Best Short | Best Long >=6 | Final Overall | Final Long >=6 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BERT | label-only | 3 | 0.3685 +/- 0.0149 | 0.5996 +/- 0.0150 | 0.2727 +/- 0.0278 | 0.3505 +/- 0.0277 | 0.2433 +/- 0.0508 |
| BERT | tsra | 3 | 0.4282 +/- 0.0315 | 0.6266 +/- 0.0117 | 0.3066 +/- 0.0615 | 0.3880 +/- 0.0214 | 0.2540 +/- 0.0377 |
| RoBERTa | label-only | 3 | 0.4887 +/- 0.0482 | 0.6580 +/- 0.0327 | 0.3779 +/- 0.0449 | 0.4593 +/- 0.0547 | 0.3235 +/- 0.0819 |
| RoBERTa | tsra | 3 | 0.5512 +/- 0.0176 | 0.7327 +/- 0.0312 | 0.3930 +/- 0.0334 | 0.5137 +/- 0.0219 | 0.3432 +/- 0.0376 |
| RoBERTa | no-consistency | 3 | 0.5785 +/- 0.0154 | 0.7587 +/- 0.0179 | 0.4127 +/- 0.0041 | 0.5398 +/- 0.0394 | 0.3824 +/- 0.0490 |
| DeBERTa | label-only | 3 | 0.4706 +/- 0.0298 | 0.6602 +/- 0.0276 | 0.3414 +/- 0.0293 | 0.4171 +/- 0.0146 | 0.2719 +/- 0.0257 |
| DeBERTa | tsra | 3 | 0.6262 +/- 0.0214 | 0.8117 +/- 0.0181 | 0.4198 +/- 0.0335 | 0.5777 +/- 0.0083 | 0.3654 +/- 0.0147 |
| DeBERTa | no-consistency | 3 | 0.6222 +/- 0.0023 | 0.8214 +/- 0.0056 | 0.4109 +/- 0.0270 | 0.6073 +/- 0.0123 | 0.3779 +/- 0.0082 |
| DeBERTa-v3 | label-only | 3 | 0.6617 +/- 0.0419 | 0.7521 +/- 0.0414 | 0.5294 +/- 0.0520 | 0.6227 +/- 0.0551 | 0.4893 +/- 0.0468 |
| DeBERTa-v3 | tsra | 3 | 0.6786 +/- 0.0368 | 0.8095 +/- 0.0199 | 0.5169 +/- 0.0622 | 0.6693 +/- 0.0303 | 0.4974 +/- 0.0578 |
| ModernBERT | label-only | 3 | 0.4424 +/- 0.0114 | 0.6786 +/- 0.0203 | 0.2986 +/- 0.0137 | 0.3965 +/- 0.0294 | 0.2424 +/- 0.0310 |
| ModernBERT | tsra | 3 | 0.5529 +/- 0.0234 | 0.7370 +/- 0.0234 | 0.3859 +/- 0.0147 | 0.5433 +/- 0.0201 | 0.3788 +/- 0.0319 |

Same-backbone rerun, `data_db9b8f04` 2/3/4-hop train split. Here `label-only` has the same TRUA-architecture ablation meaning:

| Backbone | Variant | Seeds | Best Overall | Best Short | Best Long >=6 | Final Overall | Final Long >=6 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BERT | label-only | 3 | 0.4942 +/- 0.0132 | 0.6919 +/- 0.0106 | 0.3986 +/- 0.0203 | 0.4863 +/- 0.0173 | 0.3735 +/- 0.0182 |
| BERT | tsra | 3 | 0.5223 +/- 0.0130 | 0.6869 +/- 0.0175 | 0.4255 +/- 0.0095 | 0.4930 +/- 0.0361 | 0.3727 +/- 0.0541 |
| RoBERTa | label-only | 3 | 0.6292 +/- 0.0242 | 0.7798 +/- 0.0155 | 0.5444 +/- 0.0450 | 0.5722 +/- 0.0086 | 0.4849 +/- 0.0176 |
| RoBERTa | tsra | 3 | 0.7099 +/- 0.0150 | 0.8414 +/- 0.0076 | 0.6139 +/- 0.0192 | 0.6924 +/- 0.0251 | 0.5812 +/- 0.0320 |
| DeBERTa | label-only | 3 | 0.5862 +/- 0.0173 | 0.7555 +/- 0.0076 | 0.4925 +/- 0.0314 | 0.5471 +/- 0.0182 | 0.4338 +/- 0.0300 |
| DeBERTa | tsra | 3 | 0.7455 +/- 0.0167 | 0.8596 +/- 0.0177 | 0.6508 +/- 0.0181 | 0.7074 +/- 0.0373 | 0.6131 +/- 0.0553 |
| DeBERTa-v3 | label-only | 3 | 0.7449 +/- 0.0209 | 0.8081 +/- 0.0155 | 0.7010 +/- 0.0262 | 0.7115 +/- 0.0339 | 0.6340 +/- 0.0510 |
| DeBERTa-v3 | tsra | 3 | 0.7741 +/- 0.0127 | 0.8263 +/- 0.0076 | 0.7370 +/- 0.0228 | 0.6667 +/- 0.0289 | 0.5176 +/- 0.0605 |
| ModernBERT | label-only | 3 | 0.6546 +/- 0.0353 | 0.7939 +/- 0.0320 | 0.5737 +/- 0.0406 | 0.6349 +/- 0.0187 | 0.5486 +/- 0.0167 |
| ModernBERT | tsra | 3 | 0.7068 +/- 0.0289 | 0.8253 +/- 0.0206 | 0.6273 +/- 0.0442 | 0.6778 +/- 0.0482 | 0.5611 +/- 0.0804 |

External/reference baselines:

True vanilla classifier audit on primary `data_089907f8`:

| Backbone | Input/model setting | Best Overall | Best Long >=6 | Final Overall | Final Long >=6 |
|---|---|---:|---:|---:|---:|
| BERT | story+query, encoder+linear classifier | 0.2874 +/- 0.0103 | 0.1946 +/- 0.0254 | 0.2542 +/- 0.0232 | 0.1445 +/- 0.0323 |
| RoBERTa | story+query, encoder+linear classifier | 0.3249 +/- 0.0068 | 0.2223 +/- 0.0054 | 0.2880 +/- 0.0123 | 0.1732 +/- 0.0086 |
| DeBERTa | story+query, encoder+linear classifier | 0.3438 +/- 0.0186 | 0.2415 +/- 0.0122 | 0.2821 +/- 0.0229 | 0.1685 +/- 0.0265 |
| DeBERTa-v3 | story+query, encoder+linear classifier | 0.3610 +/- 0.0465 | 0.2426 +/- 0.0750 | 0.3310 +/- 0.0487 | 0.2222 +/- 0.0625 |
| ModernBERT | story+query, encoder+linear classifier | 0.2958 +/- 0.0263 | 0.1873 +/- 0.0235 | 0.2822 +/- 0.0201 | 0.1690 +/- 0.0071 |

Same-input counterfactual training baseline on primary `data_089907f8`:

| Model | Input/model setting | Best Overall | Best Long >=6 | Final Overall | Final Long >=6 |
|---|---|---:|---:|---:|---:|
| CREST-style DeBERTa-v3 | story+query, encoder+linear classifier, relation-schema counterfactual augmentation | 0.6428 +/- 0.0013 | 0.5216 +/- 0.0059 | 0.6082 +/- 0.0277 | 0.4690 +/- 0.0406 |

CLUTRR per-hop representative rerun with explicit hop-2 through hop-10 logging:

- **Run root:** `/vepfs/tsra_outputs/clutrr_perhop_representative/latest`.
- **Artifacts:** `docs/aggregated_results/CLUTRR_PERHOP_REPRESENTATIVE.md` and `docs/aggregated_results/CLUTRR_PERHOP_REPRESENTATIVE.json`.
- **Code paths:** vanilla rows use `clutrr.cli.baseline`; TRUA rows use `clutrr.cli.train`.
- **Status:** `24/24 done, 0 failed`.

| Split | Backbone | Variant | Seeds | Best Overall | Best Long >=6 | Per-hop table |
|---|---|---|---:|---:|---:|---|
| `data_089907f8` | RoBERTa | vanilla | 3 | `0.3246 +/- 0.0136` | `0.2238 +/- 0.0113` | hop 2-10 in per-hop artifact |
| `data_089907f8` | RoBERTa | tsra | 3 | `0.5567 +/- 0.0311` | `0.3904 +/- 0.0602` | hop 2-10 in per-hop artifact |
| `data_089907f8` | DeBERTa | vanilla | 3 | `0.3438 +/- 0.0186` | `0.2415 +/- 0.0122` | hop 2-10 in per-hop artifact |
| `data_089907f8` | DeBERTa | tsra | 3 | `0.6262 +/- 0.0214` | `0.4198 +/- 0.0335` | hop 2-10 in per-hop artifact |
| `data_089907f8` | DeBERTa-v3 | vanilla | 3 | `0.3610 +/- 0.0465` | `0.2426 +/- 0.0750` | hop 2-10 in per-hop artifact |
| `data_089907f8` | DeBERTa-v3 | tsra | 3 | `0.6786 +/- 0.0368` | `0.5169 +/- 0.0622` | hop 2-10 in per-hop artifact |
| `data_db9b8f04` | DeBERTa-v3 | vanilla | 3 | `0.4237 +/- 0.0241` | `0.2792 +/- 0.0336` | hop 2-10 in per-hop artifact |
| `data_db9b8f04` | DeBERTa-v3 | tsra | 3 | `0.7741 +/- 0.0127` | `0.7370 +/- 0.0228` | hop 2-10 in core-final artifact |

CLUTRR API LLM raw-prompt references:

- **Run interface:** `clutrr.cli.eval_gemini_openai`, OpenAI-compatible chat API, `--raw_only`, raw story+query prompt, single relation-label answer.
- **Evaluation split:** public CLUTRR `data_089907f8` 2--10-hop test set, `1146` examples.
- **Artifact names from the previous run:** `/root/TRUA/outputs/gemini_eval/gpt52_clutrr_full_rawacc_hop2to10_dirfix_v1.summary.json` and `/root/TRUA/outputs/gemini_eval/gemini31pro_clutrr_full_rawacc_hop2to10_dirfix_v1.summary.json`. The current tree no longer keeps these ignored output files, so the values below are recorded from retained session output and checked against the test-hop totals.
- **Comparability note:** these are large pretrained API models and may have been exposed to public CLUTRR-like data during pretraining. They should be used as scale references, not as same-input fine-tuned baselines.

| Model | Setting | Overall | Short 2-3 | Long >=6 | H2 | H3 | H4 | H5 | H6 | H7 | H8 | H9 | H10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.2 API | raw-label prompt, no task fine-tuning | `0.6981` (`800/1146`) | `0.6713` (`96/143`) | `0.6385` (`408/639`) | `1.0000` | `0.5524` | `0.8632` | `0.7586` | `0.8505` | `0.7500` | `0.5867` | `0.5042` | `0.5126` |
| Gemini 3.1 Pro API | raw-label prompt, no task fine-tuning | `0.8010` (`918/1146`) | `0.6853` (`98/143`) | `0.7825` (`500/639`) | `1.0000` | `0.5714` | `0.9053` | `0.8391` | `0.8598` | `0.8333` | `0.7333` | `0.7563` | `0.7395` |

| Model | Input setting | Overall | Short-hop | Long-hop >=6 | Status label |
|---|---|---:|---:|---:|---|
| Edge Transformer | structured graph edges | 0.8100 | 0.9762 | 0.6847 | structured/reference baseline |
| RAT | structured relation-aware baseline | 0.5755 | 0.9762 | 0.3483 | structured/reference baseline |
| CREST-style counterfactual | raw text, DeBERTa-v3, relation-schema counterfactual training | 0.6428 | 0.9394 | 0.5216 | same-input adapted baseline |
| Dual Attention adapted | raw text, DeBERTa unfrozen | 0.2548 | 0.9580 | 0.1424 | external adapted diagnostic |
| Abstractor/RCA adapted | raw text, DeBERTa unfrozen | 0.1571 | 0.4336 | 0.1095 | external adapted diagnostic |
| MAC-style attention adapted | raw text, local MAC-style model | 0.2173 | 0.6573 | 0.1283 | attention baseline diagnostic |

CLUTRR table notes:

- The current table is the audited 10-epoch, 3-seed TRUA-architecture ablation rerun.
- The true vanilla classifier audit uses a different model/input setting from the TRUA-architecture `label-only` rows.
- Edge Transformer has the highest overall and long-hop values among rows shown here and uses structured graph-edge input.
- On the 2/3/4-hop split, TRUA rows have higher `Best Long >=6` than label-only rows for BERT, RoBERTa, DeBERTa, DeBERTa-v3, and ModernBERT.

### ProofWriter

| Model | Train depth | Test depth / task | Result | Status label |
|---|---|---|---|---|
| BERT baseline | 0/1/2 | depth-5 | acc `0.8899/0.8728` | same-backbone baseline |
| BERT+TRUA | 0/1/2 | depth-5 | acc `0.8952/0.8873`; trace@1 `0.7834/0.7848` | same-backbone TRUA |
| RoBERTa baseline | 0/1/2 | depth-5 | acc `0.8507/0.7160` | seed1 lower than seed0 |
| RoBERTa+TRUA | 0/1/2 | depth-5 | acc `0.8386/0.8698`; trace@1 `0.7966/0.7928` | trace@1 recorded |
| DeBERTa baseline | 0/1/2 | depth-5 | acc `0.8855/0.8422`; trace@1 `0.1937/0.1642` | seed0/1 baseline |
| DeBERTa+TRUA | 0/1/2 | depth-5 | acc `0.8550/0.8495`; trace@1 `0.7433/0.7854` | trace@1 recorded |
| DeBERTa DAT / Dual Attention | 0/1/2 | depth-3 / depth-5 | acc `0.6630 +/- 0.3436` / `0.6390 +/- 0.3111`; trace@1 `0.2054 +/- 0.0204` / `0.1740 +/- 0.0634` | same-input architecture baseline |
| DeBERTa Abstractor/RCA | 0/1/2 | depth-3 / depth-5 | acc `0.8761 +/- 0.1275` / `0.8221 +/- 0.0930`; trace@1 `0.2153 +/- 0.0077` / `0.1803 +/- 0.0495` | same-input architecture baseline |
| FaiRR end-to-end | official | test | answer acc `98.403`; proof acc `97.175` | official external baseline |

### RuleTaker

| Model | Setting | Result | Status label |
|---|---|---|---|
| BERT baseline | GFaiR data, test | acc `0.9608/0.9638` | same-backbone baseline |
| BERT+TRUA | GFaiR data, test | acc `0.9637/0.9630` | paired TRUA row |
| RoBERTa baseline | GFaiR data, test | acc `0.9564/0.9587` | same-backbone baseline |
| RoBERTa+TRUA | GFaiR data, test | acc `0.9618/0.9604` | paired TRUA row |
| DeBERTa baseline | GFaiR data, test | acc `0.7215/0.7215` | paired baseline row |
| DeBERTa+TRUA | GFaiR data, test | acc `0.9646/0.9676` | paired TRUA row |
| DeBERTa DAT / Dual Attention | raw-QDep 1/2 -> 1-5 | overall `0.5583 +/- 0.0605`; QDep5 `0.6121 +/- 0.1909`; trace@1 `0.1624 +/- 0.0038` | same-input architecture baseline |
| DeBERTa Abstractor/RCA | raw-QDep 1/2 -> 1-5 | overall `0.5603 +/- 0.0639`; QDep5 `0.6203 +/- 0.2051`; trace@1 `0.2082 +/- 0.0588` | same-input architecture baseline |
| GFaiR selector2 official XLNet | RuleTaker | top1 `0.9846`; top2 `0.9979` | official external component |
| GFaiR full official pipeline | RuleTaker | proof_acc_total `0.9086`; faithful_total `0.9922` | official external pipeline |
| IBR | RuleTaker depth-5 | full `0.9372`; proof `0.9374` | official external baseline |
| NLProofS | RuleTaker depth-3ext | answer overall `0.6796`; proof overall `0.9187` | completed external proof-generation baseline |

### PrOntoQA-OOD

| Model/reference | Setting | Result | Status label |
|---|---|---|---|
| Official FLAN-T5 output analysis | 4-hop OOD composed | strict proof 0.01; relaxed proof 0.35 | reference baseline |
| BERT baseline | OOD generated split | acc `1.000`; trace@1 `0.1767/0.2000` | label acc saturated |
| BERT+TRUA | OOD generated split | acc `1.000`; trace@1 `0.5100/0.4300` | trace@1 above paired baseline |
| RoBERTa baseline | OOD generated split | acc `1.000`; trace@1 `0.1733/0.2067` | label acc saturated |
| RoBERTa+TRUA | OOD generated split | acc `1.000`; trace@1 `0.3867/0.5167` | trace@1 above paired baseline |
| DeBERTa baseline | OOD generated split | acc `1.000`; trace@1 `0.2300/0.2933` | label acc saturated |
| DeBERTa+TRUA | OOD generated split | acc `1.000`; trace@1 `0.6233/0.3667` | trace@1 above paired baseline |
| DeBERTa DAT / Dual Attention | OOD generated split | acc `1.0000 +/- 0.0000`; trace@1 `0.2400 +/- 0.0850` | same-input architecture baseline |
| DeBERTa Abstractor/RCA | OOD generated split | acc `1.0000 +/- 0.0000`; trace@1 `0.2044 +/- 0.0158` | same-input architecture baseline |
| Coconut official | converted PrOntoQA-OOD validation | acc `1.0000`; CoT match `0.0000` | latent-reasoning reference; metric differs from classifier rows |
| CODI-GPT2 official train/distill | converted PrOntoQA-OOD train 1/2 -> test 3/4 | exact final-statement acc `0.8144`; depth-3 `1.0000`; depth-4 `0.7217` | task-adapted small-model CODI reference; GPT-2+LoRA, no gold proof/trace at test time |
| CODI-Llama1B official train/distill | converted PrOntoQA-OOD train 1/2 -> test 3/4 | exact final-statement acc `0.8730`; depth-3 `1.0000`; depth-4 `0.8094` | task-adapted CODI reference; Llama-3.2-1B-Instruct+LoRA, no gold proof/trace at test time |
| LoGiPT official checkpoint | ProofWriter depth-3/depth-5 raw adapter | macro `0.6515`; depth-3 `0.6466`; depth-5 `0.6564` | no gold proof/trace at test time |
| LoGiPT official checkpoint | Logic-LM ProofWriter test | acc `0.4517`; parsed `0.9350`; `271/600` | option-2 reported-style run with Logic-LM data/prompts/metric |
| AAI Qwen3-32B no intervention | ProofWriter LogicCoT test | acc `0.8350`; `501/600` | LLM attention-reference baseline |
| AAI Qwen3-32B attention intervention | ProofWriter LogicCoT test | acc `0.8267`; `496/600` | no-intervention row is `0.8350` on the same file |

## 7. Consolidated Objective Notes

This section records observed metrics, configuration scope, and comparability notes.

### CLUTRR Attention / Relational Runs

DAT adapted CLUTRR run:
- encoder/backbone: DeBERTa;
- encoder status: unfrozen;
- train/test setting: raw-text CLUTRR adaptation;
- short-hop accuracy: `0.9580`;
- long-hop accuracy: `0.1424`;
- training loss changed from `2.0561` to `0.1743`.

MAC-style attention adapted CLUTRR run:
- short-hop accuracy: `0.6573`;
- long-hop accuracy: `0.1283`.

Existing TRUA-DeBERTa CLUTRR row:
- overall accuracy: `0.6370`;
- short-hop accuracy: `0.8052`;
- long-hop accuracy: `0.4332`.

Comparability notes:
- Edge Transformer and RAT use structured CLUTRR graph/edge input.
- DAT, Abstractor/RCA, and MAC CLUTRR rows are local raw-text adaptations, not necessarily official reproductions of the original papers' preferred settings.
- CLUTRR `label-only` rows in this report are TRUA-architecture label-only ablations, not plain encoder classifier baselines.

### Non-CLUTRR Same-Input Attention Adapters

Common setting:
- script: `scripts/fair_attention_prop.py`;
- datasets: ProofWriter, RuleTaker raw-QDep, PrOntoQA-OOD;
- backbone: DeBERTa-base;
- epochs: `10`;
- seeds: `0/1/42`;
- test-time input: raw text + query; no gold trace/proof/graph; no external symbolic solver.

Recorded values:
- ProofWriter Abstractor/RCA depth-5 acc: `0.8221 +/- 0.0930`;
- ProofWriter Abstractor/RCA depth-5 trace@1: `0.1803 +/- 0.0495`;
- ProofWriter DAT depth-5 acc: `0.6390 +/- 0.3111`;
- ProofWriter DAT depth-5 trace@1: `0.1740 +/- 0.0634`;
- RuleTaker raw-QDep DAT overall: `0.5583 +/- 0.0605`;
- RuleTaker raw-QDep Abstractor/RCA overall: `0.5603 +/- 0.0639`;
- PrOntoQA-OOD DAT label acc: `1.0000 +/- 0.0000`;
- PrOntoQA-OOD DAT trace@1: `0.2400 +/- 0.0850`;
- PrOntoQA-OOD Abstractor/RCA label acc: `1.0000 +/- 0.0000`;
- PrOntoQA-OOD Abstractor/RCA trace@1: `0.2044 +/- 0.0158`.

### Latent / LLM Reasoning Baselines

Coconut PrOntoQA-OOD:
- code path: `external_baselines/Coconut_official`;
- data: converted TRUA PrOntoQA-OOD validation view;
- label accuracy: `80/80 = 1.0000`;
- CoT match: `0/80 = 0.0000`.

CODI PrOntoQA-OOD:
- code path: `external_baselines/CODI`;
- data: converted train depth 1/2 -> test depth 3/4;
- GPT-2 result file: `/vepfs/tsra_outputs/recent_external_baselines/latest/results/codi_prontoqa_train_distill_test.json`;
- CODI-GPT2 overall: `0.8144`;
- CODI-GPT2 depth-3: `1.0000`;
- CODI-GPT2 depth-4: `0.7217`;
- Llama1B result file: `/vepfs/tsra_outputs/recent_external_baselines/latest/results/codi_prontoqa_llama1b_train_distill_test.json`;
- CODI-Llama1B overall: `0.8730`;
- CODI-Llama1B depth-3: `1.0000`;
- CODI-Llama1B depth-4: `0.8094`;
- Llama1B base model path: `/vepfs/tsra_models/hf/Llama-3.2-1B-Instruct`, symlinked from `/tos/hgf/models/Llama/Llama-3.2-1B-Instruct`.

LoGiPT:
- raw ProofWriter depth adapter macro accuracy: `0.6515`;
- raw adapter depth-3 accuracy: `0.6466`;
- raw adapter depth-5 accuracy: `0.6564`;
- raw adapter parsed ratios: depth-3 `0.9783`, depth-5 `0.9959`;
- Logic-LM reported-style ProofWriter test accuracy: `0.4517`;
- Logic-LM reported-style parsed ratio: `0.9350`;
- Logic-LM reported-style count: `271/600`.

AAI:
- model: Qwen3-32B;
- model symlink: `/vepfs/tsra_models/hf/Qwen3-32B -> /tos/lxh/models/qwen3_32`;
- test file: `external_baselines/AAI/data/ProofWriter/test.logiccotkb_prompting.json`;
- no-intervention accuracy: `0.8350` (`501/600`);
- attention-aware intervention accuracy: `0.8267` (`496/600`).

CLUTRR API LLM references:
- GPT-5.2 API on public `data_089907f8` test, raw-label prompt, no task fine-tuning: overall `0.6981`, short `0.6713`, long `0.6385`.
- Gemini 3.1 Pro API on public `data_089907f8` test, raw-label prompt, no task fine-tuning: overall `0.8010`, short `0.6853`, long `0.7825`.
- These rows are scale-oriented references. For controlled encoder training with additional 4-hop supervision, DeBERTa-v3+TRUA on `data_db9b8f04` reaches overall `0.7741 +/- 0.0127` and long-hop `0.7370 +/- 0.0228`.

### NoRA / When No Paths Lead to Rome Pilot

Paper: `When No Paths Lead to Rome: Benchmarking Systematic Neural Relational Reasoning` (NeurIPS 2025 Datasets and Benchmarks track).

Benchmark role:
- optional post-main benchmark for testing whether TRUA-style trace supervision transfers beyond CLUTRR's single source-target path assumption;
- not part of the original four-dataset TRUA main evaluation;
- task format is multi-label relation-set prediction, evaluated with exact-match and F1 rather than single-label CLUTRR accuracy.

Official resources:
- generation code: `external_baselines/WhenNoPathsLeadToRome_gen`, cloned from `https://github.com/axd353/WhenNoPathsLeadToRome`;
- evaluation code: `external_baselines/WhenNoPathsLeadToRome_eval`, cloned from `https://github.com/erg0dic/WhenNoPathsLeadToRome`;
- dataset index: `https://huggingface.co/datasets/axd353/When-No-Paths-Lead-to-Rome`;
- NoRA-1.1 data path: `data/nora_1_1`;
- output root: `/vepfs/tsra_outputs/nora_1_1`.

Data status:
- NoRA-1.1 downloaded from Hugging Face mirror because direct Hugging Face TLS access failed on the dev machine;
- retained files: `train`, `test_d_na`, `test_bl_na`, `test_opec_na` parquet splits;
- removed the downloaded tutorial video (`VideoDemo/WorkingWithTheDataWalkThrough.mp4`) because it is not used for experiments.

NoRA-1.1 split summary:

| Split | Examples | Intended Generalization Axis | Notes |
|---|---:|---|---|
| train | 10,589 | in-distribution | ReasoningDepth <= 6, non-ambiguous |
| test_d_na | 6,944 | depth | ReasoningDepth > 6 |
| test_bl_na | 2,598 | backtracking load | BL out-of-distribution |
| test_opec_na | 29,498 | off-path reasoning | OPEC out-of-distribution |

Official reported reference, exact-match accuracy on NoRA without ambiguity:

| Model | D-na | BL-na | OPEC-na | Notes |
|---|---:|---:|---:|---|
| ET single-edge | 0.822 | 0.104 | 0.110 | Official paper Table 2 |
| ET multi-edge | 0.494 | 0.056 | 0.077 | Official paper Table 2 |
| RAT single-edge | 0.493 | 0.092 | 0.094 | Official paper Table 2 |
| RAT multi-edge | 0.768 | 0.023 | 0.017 | Official paper Table 2 |
| NBFNet BCE | 0.764 | 0.012 | 0.043 | Official paper Table 2 |
| R-GCN | 0.740 | 0.018 | 0.012 | Official paper Table 2 |

Current TRUA pilot configuration:
- script: `scripts/nora_tsra_runner.py`;
- backbone: `/vepfs/tsra_models/hf/deberta-v3-base`;
- input: symbolic story facts rendered as text plus symbolic query;
- target: multi-hot set of true query relations;
- TRUA supervision: derive trace labels from `derivation_chain` and supervise attention over story facts during training only;
- test-time input: story facts + query only; no gold derivation, no external solver, no world rules.
- important naming note: rows previously named `label-only` in this NoRA pilot are TRUA-architecture ablations with `lambda_trace=0.0`; they are not plain vanilla Transformer classifiers.

Running jobs:

| Run | Train/Test Limit | Epochs | Trace Weight | Output |
|---|---:|---:|---:|---|
| TRUA-architecture label-only quick pilot | 3,000 / 1,000 per test split | 3 | 0.0 | completed: `/vepfs/tsra_outputs/nora_1_1/debertav3_label_seed0_3k3ep.json` |
| TRUA quick pilot | 3,000 / 1,000 per test split | 3 | 0.1 | completed: `/vepfs/tsra_outputs/nora_1_1/debertav3_tsra_lam01_seed0_3k3ep.json` |
| TRUA-architecture label-only full pilot | 10,589 / full | 3 | 0.0 | completed: `/vepfs/tsra_outputs/nora_1_1/debertav3_label_seed0_full3.json` |
| TRUA full pilot | 10,589 / full | 3 | 0.1 | completed: `/vepfs/tsra_outputs/nora_1_1/debertav3_tsra_lam01_seed0_full3.json` |
| true vanilla DeBERTa-v3 full pilot | 10,589 / full | 3 | n/a | completed: `/vepfs/tsra_outputs/nora_1_1/debertav3_vanilla_seed0_full3.json` |

Full pilot result, full train / full test / 3 epochs:

| Model | test_d_na EM | test_d_na micro-F1 | test_bl_na EM | test_bl_na micro-F1 | test_opec_na EM | test_opec_na micro-F1 | Trace top-1 |
|---|---:|---:|---:|---:|---:|---:|---|
| true vanilla DeBERTa-v3 | 0.1872 | 0.1544 | 0.2637 | 0.2964 | 0.3182 | 0.2199 | n/a |
| TRUA-architecture label-only, lambda=0.0 | 0.1358 | 0.1115 | 0.0000 | 0.0000 | 0.0147 | 0.0101 | 0.1966 / 0.0808 / 0.2107 |
| TRUA, lambda=0.1 | 0.1875 | 0.1556 | 0.2637 | 0.2980 | 0.3193 | 0.2264 | 0.4179 / 0.1105 / 0.5382 |

Full-pilot interpretation:
- TRUA improves over the TRUA-architecture label-only ablation on all three NoRA-1.1 OOD splits in both exact-match and micro-F1;
- the largest gains are on `test_bl_na` and `test_opec_na`;
- compared with a true vanilla DeBERTa-v3 classifier, TRUA gives small positive gains on `test_d_na`, `test_bl_na` micro-F1, and `test_opec_na`;
- the gain over vanilla is much smaller than the gain over the TRUA-architecture trace-off ablation, so this should be treated as a preliminary stress-test result rather than a main-table claim without multi-seed confirmation.

Quick pilot result, 3,000 train / 1,000 test per split / 3 epochs:

| Model | test_d_na EM | test_d_na micro-F1 | test_bl_na EM | test_bl_na micro-F1 | test_opec_na EM | test_opec_na micro-F1 | Trace top-1 |
|---|---:|---:|---:|---:|---:|---:|---|
| TRUA-architecture label-only, lambda=0.0 | 0.1830 | 0.1504 | 0.5580 | 0.7541 | 0.1980 | 0.1265 | 0.0000 / 0.0000 / 0.0000 |
| TRUA, lambda=0.1 | 0.1830 | 0.1504 | 0.5580 | 0.7541 | 0.1980 | 0.1265 | 0.3690 / 0.0620 / 0.4750 |

Quick-pilot interpretation:
- classification metrics are unchanged between the TRUA-architecture label-only ablation and TRUA under this adapter and thresholding setup;
- TRUA improves trace-aligned top-1 evidence selection on `test_d_na` and `test_opec_na`;
- this suggests that the current adapter can learn derivation-relevant attention but does not yet convert it into better multi-label relation prediction;
- NoRA should remain a stress-test / future-extension result until the prediction head and threshold calibration are improved.

Sanity check result, 500 train / 300 test per split / 1 epoch:

| Model | test_d_na EM | test_d_na micro-F1 | test_bl_na EM | test_bl_na micro-F1 | test_opec_na EM | test_opec_na micro-F1 | Trace top-1 range |
|---|---:|---:|---:|---:|---:|---:|---|
| TRUA-architecture label-only, lambda=0.0 | 0.0033 | 0.0692 | 0.0000 | 0.0000 | 0.0033 | 0.1313 | 0.21-0.44 |
| TRUA, lambda=1.0 | 0.0000 | 0.0387 | 0.0000 | 0.0000 | 0.0000 | 0.0708 | 0.32-0.52 |

Sanity-check interpretation:
- the pipeline runs end-to-end and trace supervision increases evidence top-1 on this tiny setup;
- 500 examples and 1 epoch are not sufficient for meaningful exact-match comparison;
- exact-match should be interpreted strictly because many NoRA targets have multiple true relations.

Official baseline reproduction status:
- official ET/RAT/EpiGNN code is cloned but not yet runnable in the current TRUA environment;
- missing dependencies include `torch_geometric`, `torch_scatter`, and `lightning`;
- because these are CUDA/PyTorch-version-sensitive packages, official reproduction should be done in a separate environment before treating numbers as reproduced.

### Official Pipeline / Structured Reference Results

These rows use official or task-specific pipelines and are recorded separately from same-input raw-text architecture adapters:
- FaiRR end-to-end ProofWriter answer acc: `98.403099`;
- FaiRR end-to-end ProofWriter proof acc: `97.174721`;
- GFaiR selector2 official XLNet top1: `0.984560`;
- GFaiR selector2 official XLNet top2: `0.997896`;
- GFaiR full official pipeline proof_acc_total: `0.908629`;
- GFaiR full official pipeline faithful_total: `0.992208`;
- IBR RuleTaker depth-5 QA: `0.994153`;
- IBR RuleTaker depth-5 proof: `0.937416`;
- IBR RuleTaker depth-5 full: `0.937169`;
- NLProofS RuleTaker depth-3ext answer overall: `0.6796`;
- NLProofS RuleTaker depth-3ext proof overall: `0.9187`;
- Edge Transformer CLUTRR overall: `0.809951`;
- Edge Transformer CLUTRR long-hop 6-10: `0.684677`.

## 8. Table-Construction Inputs

This section lists result groups available for later paper-table construction. It does not decide which rows belong in the final manuscript.

- CLUTRR same-backbone rerun: `data_089907f8`, 10 epochs, 3 seeds, TRUA-architecture label-only and TRUA variants.
- CLUTRR true vanilla classifier audit: BERT, RoBERTa, DeBERTa, DeBERTa-v3, ModernBERT.
- CLUTRR `data_db9b8f04` 2/3/4-hop train check: BERT, RoBERTa, DeBERTa, DeBERTa-v3, ModernBERT.
- CLUTRR structured reference rows: Edge Transformer and RAT.
- CLUTRR local raw-text adapted rows: DAT, Abstractor/RCA, MAC-style attention.
- ProofWriter same-backbone TRUA rows: BERT, RoBERTa, DeBERTa.
- ProofWriter same-input attention adapter rows: DAT and Abstractor/RCA.
- ProofWriter official/reference rows: FaiRR, LoGiPT, AAI.
- RuleTaker same-backbone TRUA rows: GFaiR split and raw-QDep check.
- RuleTaker same-input attention adapter rows: DAT and Abstractor/RCA.
- RuleTaker official/reference rows: GFaiR, IBR, NLProofS.
- PrOntoQA-OOD same-backbone TRUA rows: BERT, RoBERTa, DeBERTa.
- PrOntoQA-OOD same-input attention adapter rows: DAT and Abstractor/RCA.
- PrOntoQA-OOD latent-reasoning/reference rows: Coconut, CODI-GPT2, CODI-Llama1B.

## 9. Remaining Data / Reporting Tasks

Recorded task list:

1. Keep `docs/aggregated_results/AGGREGATED_RESULTS_20260527.md` and `docs/aggregated_results/aggregated_results_20260527.json` as the source for aggregated mean/std and depth/hop grouped metrics.
2. When generating paper tables, preserve method-setting labels: same-backbone encoder, TRUA-architecture ablation, plain vanilla classifier, structured graph-edge reference, official proof pipeline, local raw-text adapter, and latent/LLM reasoning reference.
3. For PrOntoQA-OOD, record both label accuracy and trace/proof-step metrics because label accuracy is saturated in the processed split.
4. For results with high seed variance, include the reported standard deviation rather than only the mean.
5. Keep large checkpoints, generated logs, and raw result files outside git.

## 10. Chinese Objective Summary

四个数据集均已有 TRUA 相关结果记录。核心最终 rerun 的聚合文件为 `docs/aggregated_results/CORE_FINAL_CLUTRR_RERUN.md`、`docs/aggregated_results/CORE_FINAL_PROP_RERUN.md`，机器可读 JSON 为对应 `.json` 文件。

CLUTRR 结果包括三组：`data_089907f8` 主 split 的 TRUA-architecture label-only 与 TRUA rerun，plain vanilla classifier audit，以及 `data_db9b8f04` 2/3/4-hop train split 检查。`label-only` 表示 TRUA 架构下关闭 trace/edge/consistency loss 的最终标签训练，不表示纯 encoder classifier。plain vanilla classifier audit 中，BERT best/final overall 为 `0.2874 +/- 0.0103` / `0.2542 +/- 0.0232`，RoBERTa 为 `0.3249 +/- 0.0068` / `0.2880 +/- 0.0123`，DeBERTa 为 `0.3438 +/- 0.0186` / `0.2821 +/- 0.0229`，DeBERTa-v3 为 `0.3610 +/- 0.0465` / `0.3310 +/- 0.0487`，ModernBERT 为 `0.2958 +/- 0.0263` / `0.2822 +/- 0.0201`。另外，CLUTRR API LLM raw-prompt reference 已补入：GPT-5.2 API 在 `data_089907f8` test 上 overall `0.6981`、long `0.6385`；Gemini 3.1 Pro API overall `0.8010`、long `0.7825`。这些行是大模型规模参照，不是同输入微调基线。

ProofWriter 使用 train depth `0/1/2`，test depth `3/5`。BERT baseline/TRUA depth-5 acc 为 `0.8818 +/- 0.0086` / `0.8842 +/- 0.0037`，depth-5 trace@1 为 `0.1459 +/- 0.0061` / `0.7863 +/- 0.0029`。RoBERTa baseline/TRUA depth-5 acc 为 `0.8064 +/- 0.0783` / `0.8511 +/- 0.0064`，depth-5 trace@1 为 `0.1694 +/- 0.0178` / `0.7890 +/- 0.0086`。DeBERTa 行也已记录，且 seed variance 较高。

RuleTaker 包括 GFaiR split 和 raw strict QDep 1/2 train -> 1-5 test。GFaiR split 中 DeBERTa baseline/TRUA test acc 为 `0.7215 +/- 0.0000` / `0.9667 +/- 0.0011`。raw-QDep 中 DeBERTa baseline/TRUA overall 为 `0.5234 +/- 0.0000` / `0.7783 +/- 0.1353`，trace@1 为 `0.1913 +/- 0.0051` / `0.7444 +/- 0.2414`。

PrOntoQA-OOD processed split 中，BERT/RoBERTa/DeBERTa baseline 与 TRUA 的 OOD label acc 均为 `1.0000 +/- 0.0000`。trace@1 分别记录为：BERT baseline/TRUA `0.2133 +/- 0.0448` / `0.4667 +/- 0.0404`，RoBERTa baseline/TRUA `0.1733 +/- 0.0333` / `0.4856 +/- 0.0876`，DeBERTa baseline/TRUA `0.2511 +/- 0.0366` / `0.5378 +/- 0.1482`。

公平 raw-text attention adapters 已完成：DAT / Dual Attention 和 Abstractor/RCA 在 ProofWriter、RuleTaker raw-QDep、PrOntoQA-OOD 上运行，统一使用 DeBERTa-base、10 epochs、seeds `0/1/42`，测试阶段不提供 gold trace/proof/graph，不调用外部 symbolic solver。ProofWriter depth-5 上 Abstractor/RCA acc `0.8221 +/- 0.0930`、trace@1 `0.1803 +/- 0.0495`；DAT acc `0.6390 +/- 0.3111`、trace@1 `0.1740 +/- 0.0634`。RuleTaker raw-QDep 上 DAT/Abstractor overall 为 `0.5583 +/- 0.0605` / `0.5603 +/- 0.0639`。PrOntoQA-OOD 上 DAT/Abstractor label acc 均为 `1.0000 +/- 0.0000`，trace@1 为 `0.2400 +/- 0.0850` / `0.2044 +/- 0.0158`。

外部官方或参考 baseline 已记录：FaiRR end-to-end ProofWriter answer acc `98.403099`、proof acc `97.174721`；GFaiR selector2 official XLNet top1 `0.984560`、top2 `0.997896`，GFaiR full pipeline proof_acc_total `0.908629`、faithful_total `0.992208`；IBR RuleTaker depth-5 full `0.937169`；NLProofS RuleTaker depth-3ext answer overall `0.6796`、proof overall `0.9187`；Edge Transformer CLUTRR overall `0.809951`、long-hop 6-10 `0.684677`。

Coconut on PrOntoQA-OOD validation label accuracy 为 `80/80 = 1.0000`，CoT match 为 `0/80 = 0.0000`。CODI 使用 converted PrOntoQA-OOD train depth 1/2 -> test depth 3/4：GPT-2 result file 为 `/vepfs/tsra_outputs/recent_external_baselines/latest/results/codi_prontoqa_train_distill_test.json`，overall `0.8144`，depth-3 `1.0000`，depth-4 `0.7217`；Llama1B result file 为 `/vepfs/tsra_outputs/recent_external_baselines/latest/results/codi_prontoqa_llama1b_train_distill_test.json`，overall `0.8730`，depth-3 `1.0000`，depth-4 `0.8094`。AAI Qwen3-32B ProofWriter no-intervention acc 为 `0.8350`，attention-aware intervention acc 为 `0.8267`。LoGiPT raw ProofWriter adapter macro acc 为 `0.6515`；Logic-LM reported-style ProofWriter test acc 为 `0.4517`，parsed ratio `0.9350`。
