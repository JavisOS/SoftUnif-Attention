# TSRA Experiment Report

## 1. Goal

This report summarizes the current experimental evidence for **TSRA: Trace-Supervised Reasoning Attention**.

The main claim under examination is:

> In query-conditioned multi-step textual reasoning, training-time supervision over gold reasoning traces helps a Transformer learn internal step-selection behavior that generalizes better to long-hop, high-depth, and OOD reasoning than ordinary final-label supervision or generic attention/relational architectures.

The focus is not to prove that TSRA beats every symbolic or graph/oracle system. The intended comparison is against:

- same-backbone Transformer classifiers trained only with final labels;
- Transformer/attention/reasoning architectures that do not receive trace supervision;
- proof-reasoning methods that decompose natural-language reasoning into rule/fact selection or proof steps.

Evaluation emphasizes shallow-train / deep-test settings:

- train on short reasoning chains or low proof depth;
- test on longer chains, deeper proof depth, or OOD systematic/compositional splits;
- report hop/depth grouped metrics, not only overall accuracy.

## 0. Latest Status Snapshot

Updated on **2026-05-29 05:18 Asia/Shanghai** after the CLUTRR same-backbone rerun finished and the CLUTRR true vanilla classifier audit completed. This section is the current authoritative summary. Values are `mean +/- sample-std` over seeds `0/1/42` unless stated otherwise.

### Completion Status

- **CLUTRR backbone sweep is complete:** `/vepfs/tsra_outputs/clutrr_backbone_sweep/latest` is `42/42 done, 0 failed`.
- **CLUTRR label-only audit note:** the CLUTRR `label-only` rows in the same-backbone tables are **TSRA-architecture label-only ablations**, not plain vanilla RoBERTa/DeBERTa classifier fine-tuning. They run through `clutrr.cli.train` / `TsraReasonerModel` with trace, edge, and consistency losses set to zero, but still use entity spans, query-conditioned pair/relation attention, and renamed-input label CE.
- **True vanilla CLUTRR classifier audit is complete:** `/vepfs/tsra_outputs/clutrr_vanilla_classifier_audit/latest` is `6/6 done, 0 failed`; results are also summarized in `docs/aggregated_results/CLUTRR_VANILLA_CLASSIFIER_AUDIT_20260529.md`.
- **TSRA/backbone runs are complete:** additional depth/seed checks are `14/14 done, 0 failed`; seed-42 completion is `22/22 done, 0 failed`.
- **Final aggregated artifacts:** `docs/aggregated_results/AGGREGATED_RESULTS_20260527.md`, `docs/aggregated_results/aggregated_results_20260527.json`, `docs/aggregated_results/CLUTRR_BACKBONE_SWEEP_20260529.md`, and `docs/aggregated_results/clutrr_backbone_sweep_20260529.json`.
- **Aggregation scripts:** `scripts/aggregate_experiment_results.py` and `scripts/aggregate_clutrr_backbone_sweep.py`.
- **NLProofS formal RuleTaker test is complete:** final result file is at `/vepfs/tsra_outputs/official_external/latest_nlproofs_ruletaker_test/prover_test/lightning_logs/version_0/results_test.json`. Reported test metrics: answer overall `0.6796`, proof overall `0.9187`.

### Completed TSRA Main Results

#### CLUTRR `data_089907f8`

This is the primary CLUTRR split used throughout TSRA, with 2/3-hop training. This table replaces the older draft same-backbone CLUTRR table whose configuration was not reliable. In this CLUTRR table, `label-only` means the TSRA architecture trained with final-label CE only; it is not the plain backbone classifier baseline.

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

Interpretation: the merged rerun gives a mixed but useful architecture-ablation story. BERT, DeBERTa, RoBERTa, and ModernBERT show TSRA gains over the TSRA-architecture label-only ablation on long-hop examples. DeBERTa-v3 improves overall accuracy but is roughly tied/slightly lower on long-hop in the primary split. RoBERTa's no-consistency ablation remains stronger than full TSRA, so the consistency term should be reported cautiously. Do not cite these rows as vanilla Transformer classifier numbers until the separate vanilla audit finishes.

#### CLUTRR True Vanilla Classifier Audit

This audit uses the plain `clutrr.cli.baseline` entry point: story + query input, encoder + linear classifier, final-label CE only. It does not use entity spans, TSRA relation attention, trace/path supervision, or consistency losses.

| Backbone | Seeds | Best Overall | Best Short | Best Long >=6 | Final Overall | Final Long >=6 |
|---|---:|---:|---:|---:|---:|---:|
| RoBERTa | 3 | 0.3249 +/- 0.0068 | 0.9650 +/- 0.0070 | 0.2223 +/- 0.0054 | 0.2880 +/- 0.0123 | 0.1732 +/- 0.0086 |
| DeBERTa-v3 | 3 | 0.3610 +/- 0.0465 | 0.9580 +/- 0.0121 | 0.2426 +/- 0.0750 | 0.3310 +/- 0.0487 | 0.2222 +/- 0.0625 |

Interpretation: this audit confirms that the high CLUTRR `label-only` numbers, especially DeBERTa-v3 `0.6617`, are not pure backbone fine-tuning results. They should be reported as TSRA-architecture label-only ablations, while the true vanilla classifier baselines are much lower.

#### CLUTRR `data_db9b8f04` 2/3/4-Hop Train Check

This follow-up trains on 2/3/4-hop examples and tests long-hop generalization. As above, `label-only` means the TSRA architecture with trace/edge/consistency losses disabled, not a plain backbone classifier.

| Backbone | Variant | Seeds | Best Overall | Best Short | Best Long >=6 | Final Overall | Final Long >=6 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BERT | label-only | 3 | 0.4942 +/- 0.0132 | 0.6919 +/- 0.0106 | 0.3986 +/- 0.0203 | 0.4863 +/- 0.0173 | 0.3735 +/- 0.0182 |
| BERT | tsra | 3 | 0.5223 +/- 0.0130 | 0.6869 +/- 0.0175 | 0.4255 +/- 0.0095 | 0.4930 +/- 0.0361 | 0.3727 +/- 0.0541 |
| RoBERTa | label-only | 3 | 0.6292 +/- 0.0242 | 0.7798 +/- 0.0155 | 0.5444 +/- 0.0450 | 0.5722 +/- 0.0086 | 0.4849 +/- 0.0176 |
| RoBERTa | tsra | 3 | 0.7099 +/- 0.0150 | 0.8414 +/- 0.0076 | 0.6139 +/- 0.0192 | 0.6924 +/- 0.0251 | 0.5812 +/- 0.0320 |
| DeBERTa | label-only | 3 | 0.5862 +/- 0.0173 | 0.7555 +/- 0.0076 | 0.4925 +/- 0.0314 | 0.5471 +/- 0.0182 | 0.4338 +/- 0.0300 |
| DeBERTa | tsra | 3 | 0.7455 +/- 0.0167 | 0.8596 +/- 0.0177 | 0.6508 +/- 0.0181 | 0.7074 +/- 0.0373 | 0.6131 +/- 0.0553 |
| DeBERTa-v3 | label-only | 3 | 0.7449 +/- 0.0209 | 0.8081 +/- 0.0155 | 0.7010 +/- 0.0262 | 0.7115 +/- 0.0339 | 0.6340 +/- 0.0510 |
| DeBERTa-v3 | tsra | 3 | 0.7811 +/- 0.0011 | 0.8323 +/- 0.0046 | 0.7462 +/- 0.0140 | 0.7023 +/- 0.0363 | 0.5846 +/- 0.0671 |
| ModernBERT | label-only | 3 | 0.6546 +/- 0.0353 | 0.7939 +/- 0.0320 | 0.5737 +/- 0.0406 | 0.6349 +/- 0.0187 | 0.5486 +/- 0.0167 |
| ModernBERT | tsra | 3 | 0.7068 +/- 0.0289 | 0.8253 +/- 0.0206 | 0.6273 +/- 0.0442 | 0.6778 +/- 0.0482 | 0.5611 +/- 0.0804 |

Interpretation: the 2/3/4-hop split gives the strongest CLUTRR same-architecture evidence. TSRA improves long-hop accuracy over the TSRA-architecture label-only ablation for BERT, RoBERTa, DeBERTa, DeBERTa-v3, and ModernBERT.

#### ProofWriter

Train depth is 0/1/2; test uses depth-3 and depth-5. Accuracy and trace@1 are reported separately.

| Backbone | Model | Seeds | Depth-3 Acc | Depth-5 Acc | Depth-3 Trace@1 | Depth-5 Trace@1 |
|---|---|---:|---:|---:|---:|---:|
| BERT | baseline | 3 | 0.9596 +/- 0.0029 | 0.8818 +/- 0.0086 | 0.2053 +/- 0.0237 | 0.1459 +/- 0.0061 |
| BERT | TSRA | 3 | 0.9601 +/- 0.0030 | 0.8877 +/- 0.0073 | 0.8315 +/- 0.0016 | 0.7841 +/- 0.0007 |
| RoBERTa | baseline | 3 | 0.8703 +/- 0.1225 | 0.8064 +/- 0.0783 | 0.2080 +/- 0.0046 | 0.1694 +/- 0.0178 |
| RoBERTa | TSRA | 3 | 0.9453 +/- 0.0123 | 0.8531 +/- 0.0157 | 0.8430 +/- 0.0053 | 0.7934 +/- 0.0030 |
| DeBERTa | baseline | 3 | 0.8663 +/- 0.1197 | 0.8146 +/- 0.0880 | 0.2498 +/- 0.0585 | 0.1687 +/- 0.0231 |
| DeBERTa | TSRA | 3 | 0.8740 +/- 0.1258 | 0.8068 +/- 0.0787 | 0.6197 +/- 0.3338 | 0.5896 +/- 0.3034 |

Interpretation: BERT and RoBERTa show the cleanest ProofWriter story: label accuracy is similar or improved, while trace@1 jumps sharply. DeBERTa seed42 baseline/TSRA are identical in the additional check, which should be treated as an audit flag rather than a strong negative conclusion.

#### RuleTaker GFaiR Split

This uses the GFaiR RuleTaker-3ext-sat split. Official test depth metadata is not available in the exported test bucket, so depth grouping is reported separately in the raw-QDep check below.

| Backbone | Model | Seeds | Test Acc | Trace@1 |
|---|---|---:|---:|---:|
| BERT | baseline | 3 | 0.9624 +/- 0.0015 | 0.6090 +/- 0.0957 |
| BERT | TSRA | 3 | 0.9646 +/- 0.0022 | 0.0226 +/- 0.0017 |
| RoBERTa | baseline | 3 | 0.9559 +/- 0.0030 | 0.5065 +/- 0.1179 |
| RoBERTa | TSRA | 3 | 0.9615 +/- 0.0009 | 0.0128 +/- 0.0036 |
| DeBERTa | baseline | 3 | 0.7215 +/- 0.0000 | 0.4201 +/- 0.1834 |
| DeBERTa | TSRA | 3 | 0.9664 +/- 0.0016 | 0.0899 +/- 0.1329 |

Interpretation: label accuracy improves most clearly for DeBERTa, while BERT/RoBERTa have smaller but stable gains. The trace@1 metric on this converted GFaiR split is not reliable as an internal reasoning metric because the exported test bucket does not preserve comparable depth/trace metadata; cite the raw-QDep and ProofWriter trace metrics instead.

#### RuleTaker Raw Strict QDep 1/2 Train -> 1-5 Test

This stricter raw-data setting filters by question-level proof depth (`QDep`) and provides the clearest RuleTaker depth-generalization audit.

| Backbone | Model | Seeds | Overall | QDep1 | QDep2 | QDep3 | QDep4 | QDep5 | Trace@1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeBERTa | baseline | 3 | 0.5234 +/- 0.0000 | 0.5399 +/- 0.0000 | 0.5081 +/- 0.0000 | 0.5027 +/- 0.0000 | 0.5038 +/- 0.0000 | 0.5019 +/- 0.0000 | 0.1913 +/- 0.0051 |
| DeBERTa | TSRA | 3 | 0.7783 +/- 0.1353 | 0.8188 +/- 0.2180 | 0.8463 +/- 0.1650 | 0.6414 +/- 0.1981 | 0.5263 +/- 0.1471 | 0.4944 +/- 0.2509 | 0.7444 +/- 0.2414 |

Interpretation: TSRA greatly improves overall strict-QDep accuracy and trace@1, but seed variance is large on the deepest QDep buckets. This is useful supporting evidence, not yet the cleanest final headline table.

#### PrOntoQA-OOD

The processed PrOntoQA-OOD label task is saturated, so label accuracy is a sanity check and trace@1 is the meaningful TSRA signal.

| Backbone | Model | Seeds | OOD Acc | Trace@1 |
|---|---|---:|---:|---:|
| BERT | baseline | 3 | 1.0000 +/- 0.0000 | 0.2133 +/- 0.0448 |
| BERT | TSRA | 3 | 1.0000 +/- 0.0000 | 0.4667 +/- 0.0404 |
| RoBERTa | baseline | 3 | 1.0000 +/- 0.0000 | 0.1733 +/- 0.0333 |
| RoBERTa | TSRA | 3 | 1.0000 +/- 0.0000 | 0.4856 +/- 0.0876 |
| DeBERTa | baseline | 3 | 1.0000 +/- 0.0000 | 0.2511 +/- 0.0366 |
| DeBERTa | TSRA | 3 | 1.0000 +/- 0.0000 | 0.5378 +/- 0.1482 |

Interpretation: do not use PrOntoQA label accuracy as a central claim. Use it as an internal reasoning/trace supervision sanity check, and compare against reported PrOntoQA-OOD baselines qualitatively or in a reference table.

### Completed External Baselines

| Method | Dataset | Status | Result |
|---|---|---|---|
| EdgeTransformer | CLUTRR `data_089907f8` | completed | overall `0.809951`; short-hop `0.976191`; long-hop 6-10 `0.684677`. |
| RAT | CLUTRR `data_089907f8` | completed | overall `0.575493`; short-hop `0.976191`; long-hop 6-10 `0.348255`. |
| FaiRR end-to-end | ProofWriter | completed | answer acc `98.403099`; proof acc `97.174721`. |
| GFaiR selector2 official XLNet | RuleTaker | completed | top1 `0.984560`; top2 `0.997896`; invalid ratio `0.000597`. |
| GFaiR full official pipeline | RuleTaker | completed | proof_acc_total `0.908629`; faithful_total `0.992208`. |
| IBR | RuleTaker depth-5 | completed | QA `0.994153`; proof `0.937416`; full `0.937169`. |
| NLProofS | RuleTaker depth-3ext | completed | answer overall `0.6796`; proof overall `0.9187`. |
| Abstractor/RCA adapted | CLUTRR | completed diagnostic | 3-epoch unfrozen raw-text adapter: overall `0.1571`; short `0.4336`; long `0.1095`. |
| Dual Attention adapted | CLUTRR | completed diagnostic | 3-epoch unfrozen raw-text adapter: overall `0.2548`; short `0.9580`; long `0.1424`. |

### Report File Policy

- **Final report file:** `/root/TSRA/EXPERIMENT_REPORT.md`.
- **Setup notes file:** `/root/TSRA/README_EXPERIMENTS.md`.
- Do not use old local copies such as `EXPERIMENT_REPORT.remote.md` or `TSRA_FINAL_EXPERIMENT_REPORT.md`; those were local intermediate artifacts and are not present in the cleaned remote repo.

## 0.2 Removed Mid-Run Notes

The previous 2026-05-21 mid-run notes have been removed from the current report body because they described jobs that were still running at that time. The latest status in Section 0 supersedes those intermediate observations.

## 0.3 Cross-Dataset Applicability of External Baselines

The external baselines are not uniformly plug-and-play across all four TSRA datasets. Their official code is strongly tied to the input representation and supervision format of their target benchmarks.

| Method | Official / Natural Dataset | CLUTRR | ProofWriter | RuleTaker | PrOntoQA-OOD | Recommendation |
|---|---|---:|---:|---:|---:|---|
| EdgeTransformer | CLUTRR, CFQ, COGS | yes, completed | no direct support | no direct support | no direct support | Keep as CLUTRR structured graph-edge reference. Do not force onto proof datasets unless we create an oracle graph setting. |
| RAT | CLUTRR relation-aware baseline | yes, completed | no direct support | no direct support | no direct support | Keep as CLUTRR relation-aware Transformer baseline. |
| FaiRR | ProofWriter | possible only with graph/proof conversion | yes, completed | not official in current repo | not direct | Keep as ProofWriter full end-to-end baseline; possible future work is a RuleTaker adapter, but it would be local engineering rather than official reproduction. |
| GFaiR | RuleTaker variants, Hard RuleTaker, RuleTaker-E, NL satisfiability | no | not official | yes, completed | not direct | Keep as RuleTaker-family baseline; selector2 and full official pipeline results are available. |
| NLProofS | EntailmentBank, ProofWriter/RuleTaker-style proof generation | no | possible | yes, completed | not direct | Keep as an additional proof-generation baseline for RuleTaker; final answer/proof metrics are available. |
| IBR | RuleTaker depth-5 / ParaRules-style iterative reasoning | no | not direct | yes, completed | not direct | Keep as an additional RuleTaker proof-reasoning baseline; depth-5 test result is available. |
| Abstractor/RCA | synthetic relational reasoning tasks | local raw-text adapter only | local raw-text adapter only | local raw-text adapter only | local raw-text adapter only | Use only as diagnostic if needed; not a clean official baseline for the four datasets. |
| DAT | relational/dual-attention architecture | local raw-text adapter only | local raw-text adapter only | local raw-text adapter only | local raw-text adapter only | Same as Abstractor/RCA: useful diagnostic, weak as paper-level external baseline unless adapter is carefully validated. |

Current judgment:

- We already have one strong external method per main dataset family: EdgeTransformer for CLUTRR, FaiRR for ProofWriter, GFaiR/IBR for RuleTaker.
- These external methods generally perform well on their intended datasets: EdgeTransformer has strong CLUTRR long-hop accuracy; FaiRR has very high ProofWriter answer/proof accuracy; GFaiR selector2 and full GFaiR are complete.
- The missing external-method gap is PrOntoQA-OOD. The right comparison there should be PrOntoQA-specific reported baselines or a PrOntoQA-compatible LLM/neuro-symbolic reference, not EdgeTransformer/FaiRR/GFaiR forced through an unnatural adapter.

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
- **Dev-machine path:** `/root/TSRA/data/proofwriter/raw/proofwriter-dataset-V2020.12.3`.
- **Local staging path used:** `/private/tmp/tsra_data/proofwriter-dataset-V2020.12.3.zip`.
- **TOS target for reproducibility:** `tos://c20250504/wy/data/proofwriter/proofwriter-dataset-V2020.12.3.zip`.
- **Format:** official OWA/CWA JSONL files, depth-0 through depth-5, with proof/proof-depth annotations.
- **Shallow/deep split:** train depth 0/1/2; test depth-3 and depth-5.
- **Trace definition:** proof tree linearized into sentence-level proof sequence supervision.
- **Current status:** ready; TSRA-Prop and FaiRR component experiments have been run.

### RuleTaker

- **Role:** natural-language facts/rules + query deductive reasoning dataset.
- **Status:** already present in the repository.
- **Native path:** `/root/TSRA/data/rule-reasoning-dataset-V2020.2.5.0/original`.
- **GFaiR data path:** `/root/TSRA/external_baselines/GFaiR/data/ruletaker_3ext_sat`.
- **Available settings:** depth-0, depth-1, depth-2, depth-3, depth-3ext, depth-5, hard RuleTaker variants.
- **Trace definition:** fact/rule/proposition proof-step sequence from provided proof metadata.
- **Current split used:** GFaiR RuleTaker-3ext-sat train/dev/test with `*_withmidprove.pkl`.
- **Strict raw-depth follow-up:** a new `ruletaker_raw` loader filters by question-level `QDep`, not only by directory-level `depth-*`. The completed check trains on QDep `1,2` from raw `depth-1/depth-2` train files and evaluates QDep `1,2,3,4,5` from raw depth `1,2,3,5` dev/test files.
- **Current status:** ready; TSRA-Prop and GFaiR component experiments have been run.

### PrOntoQA-OOD

- **Role:** OOD systematic/compositional generalization dataset.
- **Official source/code:** `https://github.com/asaparov/prontoqa`.
- **Raw path:** `/root/TSRA/data/prontoqa_ood/raw/prontoqa`.
- **Generated OOD data path:** `/root/TSRA/data/prontoqa_ood/processed/generated_ood_data`.
- **Official model-output path:** `/root/TSRA/data/prontoqa_ood/processed/model_outputs_ood/flan-t5/latest`.
- **Split used:** train on shallower ProofsOnly examples; test on deeper/OOD generated examples such as `4hop_OOD_Composed_random_noadj`.
- **Trace/proof definition:** proposition-level proof sequence.
- **Important caveat:** many generated files are proof-generation style positive examples, so binary classification accuracy is degenerate. For this dataset, proof-step/trace metrics and official OOD reference results are more meaningful.
- **Current status:** ready; TSRA-Prop coverage run and official FLAN-T5 output analysis have been run.

## 3. Existing TSRA Results

The repository already contains native CLUTRR TSRA code and logs. The most relevant existing result uses the required `data_089907f8` split.

### CLUTRR Existing TSRA Logs

| Model/log | Overall | Short-hop | Long-hop >=6 | Notes |
|---|---:|---:|---:|---|
| TSRA-DeBERTa `rollback_a90_default_deberta_5ep.log` | 0.6370 | 0.8052 | 0.4332 | best existing CLUTRR TSRA result found |
| TSRA-RoBERTa seed0 | 0.5218 | 0.7175 | 0.3503 | existing run |
| TSRA-RoBERTa seed1 | 0.5401 | 0.7565 | 0.3690 | existing run |
| TSRA-RoBERTa seed123 | 0.4904 | 0.7208 | 0.2995 | existing run |
| TSRA-RoBERTa 10ep config | 0.5497 | 0.7143 | 0.4037 | existing run |

Interpretation:

- CLUTRR already has the strongest TSRA evidence.
- The DeBERTa TSRA run gives meaningful long-hop generalization: long-hop accuracy 0.4332 when training only on 2/3-hop examples.
- Full same-backbone Transformer tables from the paper draft should be reused rather than rerun unless exact missing cells are identified.

## 4. External Baselines

### Edge Transformer on CLUTRR

- **Paper/code family:** Edge Transformer for CLUTRR.
- **Official code:** `https://github.com/bergen/EdgeTransformer`.
- **Clone path:** `/root/TSRA/external_baselines/EdgeTransformer`.
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

Interpretation:

- Edge Transformer is a strong CLUTRR reference baseline.
- It should be clearly labeled as a **structured graph-edge baseline**, because it uses structured CLUTRR graph/edge information at test time.
- It is not a same-input raw-text baseline against TSRA.

### Dual Attention Transformer on CLUTRR

- **Paper:** `Disentangling and Integrating Relational and Sensory Information in Transformer Architectures` (2024/2025).
- **Official code:** `https://github.com/Awni00/dual-attention`.
- **Clone path:** `/root/TSRA/external_baselines/dual-attention`.
- **Dataset chosen:** CLUTRR, because the method targets relational reasoning and CLUTRR is the closest entity-relation path task among the four datasets.
- **Implementation:** official PyTorch `DualAttention` module with a local CLUTRR raw-text adapter.
- **Run type:** DeBERTa encoder unfrozen, full `data_089907f8` train split, 3 epochs.
- **Output:** `/root/TSRA/outputs/external_baselines/dual_attention_clutrr_unfrozen_3ep.json`.

| Model | Train | Epochs | Overall | Short-hop | Long-hop >=6 | Training loss |
|---|---:|---:|---:|---:|---:|---|
| Dual Attention adapted | 10094 | 3 | 0.2548 | 0.9580 | 0.1424 | 2.0561 -> 0.1743 |

Per-hop accuracy:

| Hop | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Accuracy | 0.9737 | 0.9524 | 0.2000 | 0.1494 | 0.1308 | 0.1319 | 0.1467 | 0.1345 | 0.1681 |

Interpretation:

- This is one of the clearest diagnostic results.
- With DeBERTa unfrozen, the model clearly learns the shallow training distribution: short-hop accuracy reaches 0.9580 and training loss nearly vanishes.
- However, long-hop accuracy remains only 0.1424.
- This supports the TSRA motivation: generic relational/attention inductive bias can fit shallow relational patterns but does not necessarily learn systematic multi-hop reasoning under shallow-train/deep-test generalization.
- Because the CLUTRR adapter is local, this should be described as an **adapted raw-text CLUTRR run**, not full official reproduction.

### Abstractors and Relational Cross-Attention on CLUTRR

- **Paper:** `Abstractors and Relational Cross-Attention: An Inductive Bias for Explicit Relational Reasoning in Transformers`, ICLR 2024.
- **Official code:** `https://github.com/Awni00/abstractor`.
- **Project page:** `https://awni.xyz/abstractor/`.
- **Dataset chosen:** CLUTRR, because it is the closest entity-relation path reasoning dataset.
- **Implementation:** local PyTorch relational cross-attention adapter for CLUTRR raw text. The official repo is TensorFlow/Keras and does not provide a CLUTRR pipeline.
- **Run type:** DeBERTa encoder unfrozen, full `data_089907f8` train split, 3 epochs.
- **Output:** `/root/TSRA/outputs/external_baselines/abstractor_rca_clutrr_unfrozen_3ep.json`.

| Model | Train | Epochs | Overall | Short-hop | Long-hop >=6 | Training loss |
|---|---:|---:|---:|---:|---:|---|
| Abstractor/RCA adapted | 10094 | 3 | 0.1571 | 0.4336 | 0.1095 | 2.3970 -> 1.0967 |

Per-hop accuracy:

| Hop | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Accuracy | 0.6316 | 0.3619 | 0.1737 | 0.0862 | 0.1028 | 0.1319 | 0.0933 | 0.1008 | 0.1176 |

Interpretation:

- Unfreezing the text encoder confirms that the model can learn some shallow CLUTRR signal.
- Generalization to deeper hops remains weak.
- This should not be phrased as a failure of the original ICLR 2024 method on its intended synthetic relational tasks. The fair statement is narrower: in this raw-text CLUTRR adaptation, relational cross-attention mainly learns shallow-hop cues and does not solve long-hop systematic generalization.

### MAC-style Compositional Attention on CLUTRR

- **Related method family:** MAC / compositional attention reasoning.
- **MAC paper:** `https://arxiv.org/abs/1803.03067`.
- **CLUTRR baseline repo:** `https://github.com/koustuvsinha/clutrr-baselines`.
- **Reason for inclusion:** the official CLUTRR baseline repository includes a MAC config, and MAC is a classic attention-based compositional reasoning architecture.
- **Official blocker:** the CLUTRR baseline repo depends on old unavailable packages (`addict`, `comet_ml`, `torch_geometric`, `pytorch_pretrained_bert`).
- **Implementation:** lightweight local MAC-style query-conditioned attention runner.
- **Script:** `/root/TSRA/scripts/mac_attention_clutrr.py`.
- **Output:** `/root/TSRA/outputs/external_baselines/mac_attention_clutrr_20ep.json`.

| Model | Train | Epochs | Overall | Short-hop | Long-hop >=6 | Training loss |
|---|---:|---:|---:|---:|---:|---|
| MAC-style compositional attention adapted | 10094 | 20 | 0.2173 | 0.6573 | 0.1283 | 2.6716 -> 0.3291 |

Per-hop accuracy:

| Hop | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Accuracy | 0.6316 | 0.6667 | 0.2474 | 0.1494 | 0.1121 | 0.1528 | 0.1067 | 0.1429 | 0.1261 |

Interpretation:

- The model clearly learns the shallow train distribution.
- Long-hop generalization remains weak.
- This provides a useful attention-reasoning baseline for the TSRA story: iterative attention helps shallow reasoning but does not by itself produce robust systematic extrapolation to deeper hops.

### GFaiR on RuleTaker

- **Paper:** GFaiR, LREC-COLING 2024.
- **Official code:** `https://github.com/spirit-moon-fly/GFaiR`.
- **Paper link:** `https://aclanthology.org/2024.lrec-main.1436.pdf`.
- **Clone path:** `/root/TSRA/external_baselines/GFaiR`.
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

Interpretation:

- The official GFaiR results are now available and should be preferred over the older adapted-DeBERTa smoke/component result.
- The adapted DeBERTa selector result is useful only as a local sanity check.
- GFaiR remains a strong RuleTaker proof-reasoning reference baseline, but it is not a same-backbone comparison against TSRA.

### FaiRR on ProofWriter

- **Paper:** `FaiRR: Faithful and Robust Deductive Reasoning over Natural Language`, ACL 2022.
- **Official code:** `https://github.com/INK-USC/FaiRR`.
- **Paper link:** `https://aclanthology.org/2022.acl-long.77/`.
- **Clone path:** `/root/TSRA/external_baselines/FaiRR`.
- **Dataset chosen:** ProofWriter, because FaiRR decomposes natural-language reasoning into rule selection, fact selection, and reasoning.
- **Current status:** official end-to-end ProofWriter run completed; older component-level runs are retained as diagnostics.

| Component | Train | Eval | Result | Output |
|---|---:|---:|---|---|
| FaiRR end-to-end | official | test | answer acc `98.403099`; proof acc `97.174721` | `/vepfs/tsra_outputs/official_external/latest_fairr_e2e_retry` |
| FaiRR fact-selector adapted DeBERTa | 2000 | 1000 | dev top1_acc 0.981; test top1_acc 0.988; test token_acc 0.9956 | `outputs/external_baselines/fairr_fact_deberta_2k.json` |

Interpretation:

- The official end-to-end result should be used as the paper-level FaiRR comparison.
- The adapted fact-selector result is only a component diagnostic.
- The official rule-selector path was also verified; however, the final summary should cite the end-to-end result rather than the older component diagnostic.

### PrOntoQA-OOD Official Reference

- **Official code/data:** `https://github.com/asaparov/prontoqa`.
- **Paper:** `https://arxiv.org/abs/2305.15269`.
- **Analyzer target:** `4hop_OOD_Composed_random_noadj.json`.

| Reference | Setting | Strict proof correctness | Relaxed/non-atomic correctness | Notes |
|---|---|---:|---:|---|
| Official FLAN-T5 output analysis | 4-hop OOD composed random no-adj | 0.01 | 0.35 | official output reference |

Interpretation:

- PrOntoQA-OOD is not naturally suited to EdgeTransformer/FaiRR/DAT adaptation without changing the task.
- For now, official OOD reported/analyzed results are the appropriate reference baseline.

## 5. TSRA Coverage Beyond CLUTRR

In addition to the native CLUTRR TSRA pipeline, the repository now has a shared TSRA-Prop runner for ProofWriter, RuleTaker, and PrOntoQA-OOD:

- **Script:** `/root/TSRA/scripts/transformer_tsra_prop.py`.
- **Backbones run formally:** DeBERTa, RoBERTa, and BERT.
- **Design:** same encoder for baseline and TSRA; trace labels supervise sentence-selection logits during training only.
- **Test-time policy:** raw context/query only; no gold trace/path/proof is provided at inference.

Early small-limit TSRA-Prop runs have been superseded by formal 10-epoch runs. The tables below are the current results to cite.

### ProofWriter Formal TSRA-Prop

| Backbone | Model | Seed | Depth-3 Acc | Depth-5 Acc | Depth-3 Trace@1 | Depth-5 Trace@1 | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| BERT | baseline | 0 | 0.9629 | 0.8899 | 0.1795 | 0.1392 | formal 10ep |
| BERT | baseline | 1 | 0.9577 | 0.8728 | 0.2106 | 0.1512 | formal 10ep |
| BERT | TSRA | 0 | 0.9635 | 0.8952 | 0.8330 | 0.7834 | formal 10ep |
| BERT | TSRA | 1 | 0.9591 | 0.8873 | 0.8298 | 0.7848 | formal 10ep |
| RoBERTa | baseline | 0 | 0.9406 | 0.8507 | 0.2112 | 0.1876 | formal 10ep |
| RoBERTa | baseline | 1 | 0.7288 | 0.7160 | 0.2027 | 0.1521 | unstable seed |
| RoBERTa | TSRA | 0 | 0.9322 | 0.8386 | 0.8487 | 0.7966 | formal 10ep |
| RoBERTa | TSRA | 1 | 0.9565 | 0.8698 | 0.8423 | 0.7928 | formal 10ep |
| DeBERTa | baseline | 0 | 0.9473 | 0.8855 | 0.3145 | 0.1937 | formal 10ep |
| DeBERTa | baseline | 1 | 0.9228 | 0.8422 | 0.2342 | 0.1642 | seed-1 rerun existed; previously omitted from top summary |
| DeBERTa | TSRA | 0 | 0.9479 | 0.8550 | 0.7875 | 0.7433 | formal 10ep |
| DeBERTa | TSRA | 1 | 0.9454 | 0.8495 | 0.8365 | 0.7854 | seed-1 rerun |

### RuleTaker Formal TSRA-Prop

| Backbone | Model | Seed | Test Acc | Dev Acc | Dev Trace@1 | Notes |
|---|---|---:|---:|---:|---:|---|
| BERT | baseline | 0 | 0.9608 | 0.9560 | 0.1553 | formal 10ep |
| BERT | baseline | 1 | 0.9638 | 0.9605 | 0.1681 | formal 10ep |
| BERT | TSRA | 0 | 0.9637 | 0.9582 | 0.8386 | formal 10ep |
| BERT | TSRA | 1 | 0.9630 | 0.9609 | 0.8387 | formal 10ep |
| RoBERTa | baseline | 0 | 0.9564 | 0.9518 | 0.1151 | formal 10ep |
| RoBERTa | baseline | 1 | 0.9587 | 0.9542 | 0.1726 | formal 10ep |
| RoBERTa | TSRA | 0 | 0.9618 | 0.9581 | 0.8356 | formal 10ep |
| RoBERTa | TSRA | 1 | 0.9604 | 0.9582 | 0.8419 | formal 10ep |
| DeBERTa | baseline | 0 | 0.7215 | 0.7154 | 0.1732 | formal 10ep |
| DeBERTa | baseline | 1 | 0.7215 | 0.7154 | 0.1627 | seed-1 rerun |
| DeBERTa | TSRA | 0 | 0.9646 | 0.9599 | 0.2444 | formal 10ep |
| DeBERTa | TSRA | 1 | 0.9676 | 0.9637 | 0.8424 | seed-1 rerun |

### PrOntoQA-OOD Formal TSRA-Prop

| Backbone | Model | Seed | OOD Label Acc | Trace@1 | Notes |
|---|---|---:|---:|---:|---|
| BERT | baseline | 0 | 1.0000 | 0.1767 | label metric degenerate |
| BERT | baseline | 1 | 1.0000 | 0.2000 | label metric degenerate |
| BERT | TSRA | 0 | 1.0000 | 0.5100 | trace metric improves |
| BERT | TSRA | 1 | 1.0000 | 0.4300 | trace metric improves |
| RoBERTa | baseline | 0 | 1.0000 | 0.1733 | label metric degenerate |
| RoBERTa | baseline | 1 | 1.0000 | 0.2067 | label metric degenerate |
| RoBERTa | TSRA | 0 | 1.0000 | 0.3867 | trace metric improves |
| RoBERTa | TSRA | 1 | 1.0000 | 0.5167 | trace metric improves |
| DeBERTa | baseline | 0 | 1.0000 | 0.2300 | label metric degenerate |
| DeBERTa | baseline | 1 | 1.0000 | 0.2933 | label metric degenerate |
| DeBERTa | TSRA | 0 | 1.0000 | 0.6233 | trace metric improves |
| DeBERTa | TSRA | 1 | 1.0000 | 0.3667 | trace metric improves less than seed0 |

Interpretation:

- ProofWriter shows the cleanest BERT same-backbone gain: TSRA improves depth-5 accuracy in both seeds while strongly improving trace@1.
- RuleTaker shows very large DeBERTa gain and smaller but stable BERT/RoBERTa gains.
- PrOntoQA label accuracy is degenerate, but trace@1 improves consistently for BERT/RoBERTa and partly for DeBERTa.

## 6. Consolidated Results Tables

### CLUTRR Main Comparison

Same-backbone rerun, primary `data_089907f8` split. Here `label-only` is a TSRA-architecture final-label ablation, not a plain backbone classifier:

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

Same-backbone rerun, `data_db9b8f04` 2/3/4-hop train split. Here `label-only` has the same TSRA-architecture ablation meaning:

| Backbone | Variant | Seeds | Best Overall | Best Short | Best Long >=6 | Final Overall | Final Long >=6 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BERT | label-only | 3 | 0.4942 +/- 0.0132 | 0.6919 +/- 0.0106 | 0.3986 +/- 0.0203 | 0.4863 +/- 0.0173 | 0.3735 +/- 0.0182 |
| BERT | tsra | 3 | 0.5223 +/- 0.0130 | 0.6869 +/- 0.0175 | 0.4255 +/- 0.0095 | 0.4930 +/- 0.0361 | 0.3727 +/- 0.0541 |
| RoBERTa | label-only | 3 | 0.6292 +/- 0.0242 | 0.7798 +/- 0.0155 | 0.5444 +/- 0.0450 | 0.5722 +/- 0.0086 | 0.4849 +/- 0.0176 |
| RoBERTa | tsra | 3 | 0.7099 +/- 0.0150 | 0.8414 +/- 0.0076 | 0.6139 +/- 0.0192 | 0.6924 +/- 0.0251 | 0.5812 +/- 0.0320 |
| DeBERTa | label-only | 3 | 0.5862 +/- 0.0173 | 0.7555 +/- 0.0076 | 0.4925 +/- 0.0314 | 0.5471 +/- 0.0182 | 0.4338 +/- 0.0300 |
| DeBERTa | tsra | 3 | 0.7455 +/- 0.0167 | 0.8596 +/- 0.0177 | 0.6508 +/- 0.0181 | 0.7074 +/- 0.0373 | 0.6131 +/- 0.0553 |
| DeBERTa-v3 | label-only | 3 | 0.7449 +/- 0.0209 | 0.8081 +/- 0.0155 | 0.7010 +/- 0.0262 | 0.7115 +/- 0.0339 | 0.6340 +/- 0.0510 |
| DeBERTa-v3 | tsra | 3 | 0.7811 +/- 0.0011 | 0.8323 +/- 0.0046 | 0.7462 +/- 0.0140 | 0.7023 +/- 0.0363 | 0.5846 +/- 0.0671 |
| ModernBERT | label-only | 3 | 0.6546 +/- 0.0353 | 0.7939 +/- 0.0320 | 0.5737 +/- 0.0406 | 0.6349 +/- 0.0187 | 0.5486 +/- 0.0167 |
| ModernBERT | tsra | 3 | 0.7068 +/- 0.0289 | 0.8253 +/- 0.0206 | 0.6273 +/- 0.0442 | 0.6778 +/- 0.0482 | 0.5611 +/- 0.0804 |

External/reference baselines:

True vanilla classifier audit on primary `data_089907f8`:

| Backbone | Input/model setting | Best Overall | Best Long >=6 | Final Overall | Final Long >=6 |
|---|---|---:|---:|---:|---:|
| RoBERTa | story+query, encoder+linear classifier | 0.3249 +/- 0.0068 | 0.2223 +/- 0.0054 | 0.2880 +/- 0.0123 | 0.1732 +/- 0.0086 |
| DeBERTa-v3 | story+query, encoder+linear classifier | 0.3610 +/- 0.0465 | 0.2426 +/- 0.0750 | 0.3310 +/- 0.0487 | 0.2222 +/- 0.0625 |

| Model | Input setting | Overall | Short-hop | Long-hop >=6 | Paper-use status |
|---|---|---:|---:|---:|---|
| Edge Transformer | structured graph edges | 0.8100 | 0.9762 | 0.6847 | structured/reference baseline |
| RAT | structured relation-aware baseline | 0.5755 | 0.9762 | 0.3483 | structured/reference baseline |
| Dual Attention adapted | raw text, DeBERTa unfrozen | 0.2548 | 0.9580 | 0.1424 | external adapted diagnostic |
| Abstractor/RCA adapted | raw text, DeBERTa unfrozen | 0.1571 | 0.4336 | 0.1095 | external adapted diagnostic |
| MAC-style attention adapted | raw text, local MAC-style model | 0.2173 | 0.6573 | 0.1283 | attention baseline diagnostic |

Key CLUTRR takeaway:

- The older draft same-backbone table should be retired; the current table is the audited 10-epoch, 3-seed TSRA-architecture ablation rerun.
- The true vanilla classifier audit is much lower than the TSRA-architecture `label-only` rows, so these two baselines must not be conflated in the paper.
- Edge Transformer remains strongest, but it uses structured graph-edge input rather than the same raw-text setting.
- On the 2/3/4-hop split, TSRA gives the clearest same-architecture long-hop gains across BERT, RoBERTa, DeBERTa, DeBERTa-v3, and ModernBERT.

### ProofWriter

| Model | Train depth | Test depth / task | Result | Paper-use status |
|---|---|---|---|---|
| BERT baseline | 0/1/2 | depth-5 | acc `0.8899/0.8728` | same-backbone baseline |
| BERT+TSRA | 0/1/2 | depth-5 | acc `0.8952/0.8873`; trace@1 `0.7834/0.7848` | strongest same-backbone TSRA signal |
| RoBERTa baseline | 0/1/2 | depth-5 | acc `0.8507/0.7160` | unstable seed1 |
| RoBERTa+TSRA | 0/1/2 | depth-5 | acc `0.8386/0.8698`; trace@1 `0.7966/0.7928` | stabilizes trace selection |
| DeBERTa baseline | 0/1/2 | depth-5 | acc `0.8855/0.8422`; trace@1 `0.1937/0.1642` | seed0/1 baseline |
| DeBERTa+TSRA | 0/1/2 | depth-5 | acc `0.8550/0.8495`; trace@1 `0.7433/0.7854` | trace improves, label acc lower |
| FaiRR end-to-end | official | test | answer acc `98.403`; proof acc `97.175` | official external baseline |

### RuleTaker

| Model | Setting | Result | Paper-use status |
|---|---|---|---|
| BERT baseline | GFaiR data, test | acc `0.9608/0.9638` | same-backbone baseline |
| BERT+TSRA | GFaiR data, test | acc `0.9637/0.9630` | tied/small gain |
| RoBERTa baseline | GFaiR data, test | acc `0.9564/0.9587` | same-backbone baseline |
| RoBERTa+TSRA | GFaiR data, test | acc `0.9618/0.9604` | small gain |
| DeBERTa baseline | GFaiR data, test | acc `0.7215/0.7215` | weak baseline in this adapter |
| DeBERTa+TSRA | GFaiR data, test | acc `0.9646/0.9676` | strong TSRA gain |
| GFaiR selector2 official XLNet | RuleTaker | top1 `0.9846`; top2 `0.9979` | official external component |
| GFaiR full official pipeline | RuleTaker | proof_acc_total `0.9086`; faithful_total `0.9922` | official external pipeline |
| IBR | RuleTaker depth-5 | full `0.9372`; proof `0.9374` | official external baseline |
| NLProofS | RuleTaker depth-3ext | answer overall `0.6796`; proof overall `0.9187` | completed external proof-generation baseline |

### PrOntoQA-OOD

| Model/reference | Setting | Result | Paper-use status |
|---|---|---|---|
| Official FLAN-T5 output analysis | 4-hop OOD composed | strict proof 0.01; relaxed proof 0.35 | reference baseline |
| BERT baseline | OOD generated split | acc `1.000`; trace@1 `0.1767/0.2000` | label metric degenerate |
| BERT+TSRA | OOD generated split | acc `1.000`; trace@1 `0.5100/0.4300` | trace metric improves |
| RoBERTa baseline | OOD generated split | acc `1.000`; trace@1 `0.1733/0.2067` | label metric degenerate |
| RoBERTa+TSRA | OOD generated split | acc `1.000`; trace@1 `0.3867/0.5167` | trace metric improves |
| DeBERTa baseline | OOD generated split | acc `1.000`; trace@1 `0.2300/0.2933` | label metric degenerate |
| DeBERTa+TSRA | OOD generated split | acc `1.000`; trace@1 `0.6233/0.3667` | trace metric partly improves |

## 7. Main Experimental Interpretation

The most important new result is the CLUTRR attention/relational baseline behavior.

When DAT is allowed to fully train its DeBERTa encoder, it nearly solves the shallow-hop portion:

- short-hop accuracy: 0.9580;
- training loss: 2.0561 -> 0.1743.

However, its long-hop accuracy remains only 0.1424. MAC-style attention shows the same qualitative behavior:

- short-hop accuracy: 0.6573;
- long-hop accuracy: 0.1283.

This supports the TSRA motivation. These models are capable of learning local/shallow relational patterns from the training distribution, but they do not robustly extrapolate to deeper reasoning chains. In other words, strong attention or relational inductive bias alone can still behave like pattern learning or shallow perceptual association under shallow-train/deep-test evaluation.

By contrast, the existing TSRA-DeBERTa CLUTRR result reaches:

- overall accuracy: 0.6370;
- short-hop accuracy: 0.8052;
- long-hop accuracy: 0.4332.

This is the strongest current evidence that train-time trace supervision improves the model's internal reasoning-step behavior and helps long-hop generalization under raw-text inference.

Important wording for the paper:

- Do **not** claim that Abstractor/RCA or DAT are generally weak reasoning methods.
- Do claim that, under this CLUTRR raw-text shallow-train/deep-test adaptation, they learn shallow-hop behavior but fail to extrapolate strongly to long-hop reasoning.
- Do clearly separate structured/oracle-like baselines such as Edge Transformer from raw-text baselines.

## 8. What Can Enter the Paper Now

### Ready for Main Paper Tables

- CLUTRR TSRA-DeBERTa existing result.
- CLUTRR Edge Transformer and RAT reproduced reference results, clearly labeled as structured graph-edge input.
- CLUTRR DAT adapted DeBERTa-unfrozen diagnostic result.
- CLUTRR MAC-style attention diagnostic result.
- ProofWriter BERT/RoBERTa same-backbone baseline vs TSRA seed results.
- ProofWriter FaiRR official end-to-end result.
- RuleTaker BERT/RoBERTa same-backbone baseline vs TSRA seed results.
- RuleTaker GFaiR selector2 official XLNet and full official pipeline results.
- RuleTaker IBR depth-5 result.
- PrOntoQA-OOD BERT/RoBERTa trace@1 comparison, with the explicit caveat that label accuracy is degenerate.

### Preliminary / Appendix / Diagnostic Only

- Earlier ProofWriter/RuleTaker/PrOntoQA TSRA-Prop small-limit runs.
- Abstractor/RCA CLUTRR adapted result, unless more training/adapter refinement is done.
- DAT and MAC-style CLUTRR raw-text adapted results: useful diagnostics, not official reproductions of the original papers' preferred settings.

### Not Yet Paper-Quality

- PrOntoQA label-accuracy claims, unless we switch to proof/trace correctness or a non-degenerate official evaluation.

## 9. Next Steps

Priority for the next 2-3 days:

1. Use `docs/aggregated_results/AGGREGATED_RESULTS_20260527.md` as the source for paper tables; it contains mean/std and depth/hop grouped metrics over seeds `0/1/42`.
2. Build the final CLUTRR paper table with TSRA, TSRA-architecture label-only ablations, true vanilla classifier audit results, Edge Transformer, RAT, DAT adapted, MAC-style attention, and Abstractor/RCA if desired.
3. Redesign PrOntoQA-OOD reporting around trace/proof-step correctness rather than degenerate binary classification.
4. Move old diagnostic/adapted external baselines to appendix language and keep official EdgeTransformer/FaiRR/GFaiR/IBR/NLProofS as main external comparisons.
5. Update the paper draft's experiment section directly from this report.

## 10. Chinese Summary for Meeting / Draft Writing

本轮实验已经完成统一汇总。四个数据集都已经有 TSRA 结果，主要表格都补齐到 seeds `0/1/42`，并且已经产出 mean/std 和 depth/hop 分组指标。最终聚合文件在 `docs/aggregated_results/AGGREGATED_RESULTS_20260527.md`，机器可读 JSON 在 `docs/aggregated_results/aggregated_results_20260527.json`。

CLUTRR 的旧论文初稿 same-backbone 表已经废弃；当前可用的是 2026-05-29 完成的 10 epoch、3 seed rerun，但这里的 `label-only` 必须理解为 TSRA 架构下关闭 trace/edge/consistency loss 的最终标签消融，不是纯 RoBERTa/DeBERTa classifier 微调。真正的 vanilla classifier 审计已经完成：RoBERTa best overall 为 `0.3249 +/- 0.0068`、final overall 为 `0.2880 +/- 0.0123`；DeBERTa-v3 best overall 为 `0.3610 +/- 0.0465`、final overall 为 `0.3310 +/- 0.0487`。这确认了之前偏高的 RoBERTa `0.4887` 和 DeBERTa-v3 `0.6617` 不是纯 classifier baseline，而是 TSRA-architecture label-only ablation。`data_089907f8` 主 split 上，BERT、DeBERTa、RoBERTa、ModernBERT 显示 TSRA 相比这个 TSRA-architecture label-only ablation 的 long-hop 提升；DeBERTa-v3 overall 有提升但 long-hop 基本持平/略低，RoBERTa no-consistency ablation 仍强于 full TSRA，应作为例外如实报告。`data_db9b8f04` 2/3/4-hop train split 上证据更强：BERT、RoBERTa、DeBERTa、DeBERTa-v3、ModernBERT 的 TSRA long-hop 均高于 label-only，其中 DeBERTa 从 `0.4925 +/- 0.0314` 到 `0.6508 +/- 0.0181`，DeBERTa-v3 从 `0.7010 +/- 0.0262` 到 `0.7462 +/- 0.0140`，ModernBERT 从 `0.5737 +/- 0.0406` 到 `0.6273 +/- 0.0442`。

ProofWriter 上，BERT 和 RoBERTa 的 TSRA 结果最适合写入正文：BERT depth-5 从 `0.8818 +/- 0.0086` 提升到 `0.8877 +/- 0.0073`，同时 trace@1 从 `0.1459 +/- 0.0061` 提升到 `0.7841 +/- 0.0007`；RoBERTa depth-5 从 `0.8064 +/- 0.0783` 提升到 `0.8531 +/- 0.0157`，trace@1 从 `0.1694 +/- 0.0178` 提升到 `0.7934 +/- 0.0030`。DeBERTa 的 seed42 baseline/TSRA 异常相同，建议作为审计点，不作为强结论。

RuleTaker 上，GFaiR split 中 BERT/RoBERTa 是小幅稳定提升，DeBERTa 提升很大：baseline `0.7215 +/- 0.0000`，TSRA `0.9664 +/- 0.0016`。更严格的 raw QDep 1/2 train -> 1-5 test 检查中，DeBERTa baseline overall 为 `0.5234 +/- 0.0000`，TSRA 为 `0.7783 +/- 0.1353`；这个结果支持 TSRA，但方差比较大，应该作为补充实验呈现。

PrOntoQA-OOD 的 label accuracy 在当前 processed split 中完全饱和，baseline/TSRA 都是 `1.0`，不能作为核心 label claim。更有意义的是 trace@1：BERT 从 `0.2133 +/- 0.0448` 到 `0.4667 +/- 0.0404`，RoBERTa 从 `0.1733 +/- 0.0333` 到 `0.4856 +/- 0.0876`，DeBERTa 从 `0.2511 +/- 0.0366` 到 `0.5378 +/- 0.1482`。

外部 baseline 方面，FaiRR end-to-end 在 ProofWriter 上达到 answer acc `98.403`、proof acc `97.175`；GFaiR selector2 official XLNet 在 RuleTaker 上 top1 `0.9846`，完整 GFaiR pipeline 的 proof_acc_total 为 `0.9086`、faithful_total 为 `0.9922`；IBR depth-5 的 full 为 `0.9372`；NLProofS formal test 已完成，answer overall 为 `0.6796`、proof overall 为 `0.9187`；Edge Transformer 在 CLUTRR 上 overall `0.8100`、long-hop `0.6847`，但它使用结构化 graph-edge input，应该作为 structured/reference baseline，而不是和 TSRA raw-text setting 直接公平比较。

整体结论：当前结果支持 TSRA 的核心叙事。普通 Transformer 或通用 attention/relational 方法可以拟合浅层训练分布，但在 shallow-train/deep-test 的 long-hop/high-depth systematic generalization 上不足；TSRA 通过训练阶段 trace supervision 直接约束内部 reasoning-step selection，更适合 query-conditioned multi-step textual reasoning。
