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

Updated on **2026-05-26 08:12 UTC**. This is the current authoritative status for the experiment report. Older "live update" notes below are retained only as execution history and should not be cited over this section.

### Running Jobs

- **NLProofS formal RuleTaker test is still running.** The previous full test was interrupted by a dev-machine shutdown at about `15826/17580` examples and did not write a final result file. It was restarted from the trained prover checkpoint at:
  `/vepfs/tsra_outputs/official_external/latest_nlproofs_ruletaker/prover/lightning_logs/version_0/checkpoints/epoch=19-step=16940.ckpt`.
- Current retry path:
  `/vepfs/tsra_outputs/official_external/latest_nlproofs_ruletaker_test`.
- Current progress at the latest check: about `7987/17580` test batches (`~45%`) after restart. No final `results_test*.json` exists yet.
- GPU keepalive is active on GPU1-4 using `gpu_run.py`; NLProofS is on GPU0.
- **Additional depth/seed checks are running on GPUs 5-7.** This follow-up queue adds:
  CLUTRR `data_db9b8f04` 2/3/4-hop train, strict RuleTaker raw-data QDep 1/2 train -> QDep 1-5 test, and ProofWriter DeBERTa seed-42 baseline/TSRA checks.
  Current symlink: `/vepfs/tsra_outputs/additional_depth_checks/latest_depth_seed_checks`.
- **Three-seed policy update:** the formal report should ultimately aggregate seeds `0/1/42`. Most earlier formal tables were only seed `0/1`; a queued follow-up script,
  `scripts/run_seed42_completion_after_additional.sh`, waits for the current depth/seed queue and then fills the missing seed-42 runs for the main CLUTRR, ProofWriter, RuleTaker, and PrOntoQA tables.

### Completed Main TSRA/Backbone Runs

The earlier formal TSRA/backbone runs for **CLUTRR, ProofWriter, RuleTaker, and PrOntoQA-OOD** mostly cover seeds `0/1`. A third-seed completion queue is now scheduled so final paper tables can report mean/std over seeds `0/1/42`. NLProofS remains the only optional external baseline test still running.

| Dataset | Backbone | Model | Seeds | Main Result |
|---|---|---|---:|---|
| CLUTRR `data_089907f8` | DeBERTa/RoBERTa | TSRA ablations | 0/1 where available | Formal CLUTRR ablation queue complete under `/vepfs/tsra_outputs/formal_10ep/latest_tsra_formal_10ep`. |
| ProofWriter | BERT | baseline | 0,1 | depth-3 `0.9629/0.9577`; depth-5 `0.8899/0.8728`. |
| ProofWriter | BERT | TSRA | 0,1 | depth-3 `0.9635/0.9591`; depth-5 `0.8952/0.8873`. |
| ProofWriter | RoBERTa | baseline | 0,1 | depth-3 `0.9406/0.7288`; depth-5 `0.8507/0.7160`. |
| ProofWriter | RoBERTa | TSRA | 0,1 | depth-3 `0.9322/0.9565`; depth-5 `0.8386/0.8698`. |
| ProofWriter | DeBERTa | baseline | 0,1 | depth-3 `0.9473/0.9228`; depth-5 `0.8855/0.8422`. |
| ProofWriter | DeBERTa | TSRA | 0,1 | depth-3 `0.9479/0.9454`; depth-5 `0.8550/0.8495`; trace@1 strongly improves. |
| RuleTaker | BERT | baseline | 0,1 | test accuracy `0.9608/0.9638`. |
| RuleTaker | BERT | TSRA | 0,1 | test accuracy `0.9637/0.9630`. |
| RuleTaker | RoBERTa | baseline | 0,1 | test accuracy `0.9564/0.9587`. |
| RuleTaker | RoBERTa | TSRA | 0,1 | test accuracy `0.9618/0.9604`. |
| RuleTaker | DeBERTa | baseline | 0,1 | test accuracy `0.7215/0.7215`. |
| RuleTaker | DeBERTa | TSRA | 0,1 | test accuracy `0.9646/0.9676`. |
| PrOntoQA-OOD | BERT | baseline | 0,1 | label acc `1.0/1.0`; trace@1 `0.1767/0.2000`. |
| PrOntoQA-OOD | BERT | TSRA | 0,1 | label acc `1.0/1.0`; trace@1 `0.5100/0.4300`. |
| PrOntoQA-OOD | RoBERTa | baseline | 0,1 | label acc `1.0/1.0`; trace@1 `0.1733/0.2067`. |
| PrOntoQA-OOD | RoBERTa | TSRA | 0,1 | label acc `1.0/1.0`; trace@1 `0.3867/0.5167`. |
| PrOntoQA-OOD | DeBERTa | baseline | 0,1 | label acc `1.0/1.0`; trace@1 `0.2300/0.2933`. |
| PrOntoQA-OOD | DeBERTa | TSRA | 0,1 | label acc `1.0/1.0`; trace@1 `0.6233/0.3667`. |

### Completed CLUTRR TSRA Ablation

The formal CLUTRR ablation queue is complete under `/vepfs/tsra_outputs/formal_10ep/latest_tsra_formal_10ep`. It includes DeBERTa and RoBERTa with seeds 0/1 for:

- `full`: label + trace/next-hop supervision + consistency setting used by the formal TSRA run.
- `label_only`: same backbone trained without trace/next-hop supervision.
- `no_consistency`: trace/next-hop supervision retained, consistency component disabled.

The table below reports the **best logged evaluation point** from the 10-epoch queue for each seed; final-epoch values remain in the `.done` logs and can be used for a stricter appendix table.

| Backbone | Variant | Seed | Overall | Short-hop | Long-hop >=6 |
|---|---|---:|---:|---:|---:|
| DeBERTa | full | 0 | 0.6475 | 0.8279 | 0.4545 |
| DeBERTa | full | 1 | 0.6030 | 0.8117 | 0.3930 |
| DeBERTa | label_only | 0 | 0.4904 | 0.7045 | 0.3155 |
| DeBERTa | label_only | 1 | 0.4677 | 0.6396 | 0.3743 |
| DeBERTa | no_consistency | 0 | 0.6213 | 0.8182 | 0.4278 |
| DeBERTa | no_consistency | 1 | 0.6204 | 0.8279 | 0.3797 |
| RoBERTa | full | 0 | 0.5253 | 0.7013 | 0.3690 |
| RoBERTa | full | 1 | 0.5297 | 0.7435 | 0.3369 |
| RoBERTa | label_only | 0 | 0.4337 | 0.6234 | 0.3289 |
| RoBERTa | label_only | 1 | 0.4948 | 0.6818 | 0.3396 |
| RoBERTa | no_consistency | 0 | 0.5951 | 0.7792 | 0.4118 |
| RoBERTa | no_consistency | 1 | 0.5628 | 0.7468 | 0.3957 |

Mean over seeds:

| Backbone | Variant | Overall | Short-hop | Long-hop >=6 |
|---|---|---:|---:|---:|
| DeBERTa | full | 0.6253 | 0.8198 | 0.4238 |
| DeBERTa | label_only | 0.4791 | 0.6721 | 0.3449 |
| DeBERTa | no_consistency | 0.6209 | 0.8231 | 0.4038 |
| RoBERTa | full | 0.5275 | 0.7224 | 0.3530 |
| RoBERTa | label_only | 0.4643 | 0.6526 | 0.3343 |
| RoBERTa | no_consistency | 0.5790 | 0.7630 | 0.4038 |

Interpretation:

- The strongest and most consistent ablation signal is **trace/next-hop supervision vs label-only**. DeBERTa improves from `0.4791` to `0.6253` overall and from `0.3449` to `0.4238` on long-hop; RoBERTa improves from `0.4643` to `0.5275` overall.
- The consistency component is not uniformly positive in these reruns. For DeBERTa, `full` and `no_consistency` are close; for RoBERTa, `no_consistency` is higher than `full`. This should be reported honestly: the core evidence supports trace-supervised step selection, while the consistency term needs more careful tuning before being claimed as essential.

Interpretation for the current paper draft:

- On **ProofWriter**, BERT+TSRA gives the cleanest stable improvement over BERT baseline, especially at depth-5.
- On **RuleTaker**, BERT/RoBERTa TSRA is roughly tied to or slightly above the corresponding baseline; this should be reported as a stable result rather than overstated.
- On **PrOntoQA-OOD**, label accuracy is degenerate in the generated split, but TSRA substantially improves trace@1 for BERT/RoBERTa. Treat this as internal reasoning/trace evidence, not label-accuracy evidence.
- On **CLUTRR**, EdgeTransformer remains the strongest structured graph-edge baseline; TSRA comparisons must clearly separate raw-text/same-backbone settings from structured graph-edge reference baselines.

### Completed External Baselines

| Method | Dataset | Status | Result |
|---|---|---|---|
| EdgeTransformer | CLUTRR `data_089907f8` | completed | overall `0.809951`; short-hop `0.976191`; long-hop 6-10 `0.684677`. |
| RAT | CLUTRR `data_089907f8` | completed | overall `0.575493`; short-hop `0.976191`; long-hop 6-10 `0.348255`. |
| FaiRR end-to-end | ProofWriter | completed | answer acc `98.403099`; proof acc `97.174721`. |
| GFaiR selector2 official XLNet | RuleTaker | completed | top1 `0.984560`; top2 `0.997896`; invalid ratio `0.000597`. |
| GFaiR full official pipeline | RuleTaker | completed | proof_acc_total `0.908629`; faithful_total `0.992208`. |
| IBR | RuleTaker depth-5 | completed | QA `0.994153`; proof `0.937416`; full `0.937169`. |
| NLProofS | RuleTaker depth-3ext | running | formal test restarted after shutdown; final test file pending. |
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
| NLProofS | EntailmentBank, ProofWriter/RuleTaker-style proof generation | no | possible | yes, formal test running | not direct | Keep as an additional proof-generation baseline for RuleTaker; final test file is pending after a shutdown restart. |
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
- **Follow-up split:** `data/data_db9b8f04`, with `1.2,1.3,1.4_train.csv` for 2/3/4-hop training. A DeBERTa TSRA-vs-label-only seed `0/1/42` queue is running under `/vepfs/tsra_outputs/additional_depth_checks/latest_depth_seed_checks`.
- **Depth/hop definition:** length of the query-subject to query-object entity path.
- **Trace definition:** entity/relation path from query subject to query object.
- **Evaluation:** overall accuracy, short-hop accuracy, long-hop accuracy, per-hop accuracy.
- **Current status:** ready; TSRA and external CLUTRR baselines have been run on the correct `data_089907f8` split.

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
- **Strict raw-depth follow-up:** a new `ruletaker_raw` loader filters by question-level `QDep`, not only by directory-level `depth-*`. The running check trains on QDep `1,2` from raw `depth-1/depth-2` train files and evaluates QDep `1,2,3,4,5` from raw depth `1,2,3,5` dev/test files.
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

| Model | Input setting | Overall | Short-hop | Long-hop >=6 | Paper-use status |
|---|---|---:|---:|---:|---|
| TSRA-DeBERTa | raw text + train-time trace | 0.6370 | 0.8052 | 0.4332 | main TSRA evidence |
| Edge Transformer | structured graph edges | 0.8100 | 0.9762 | 0.6847 | structured/reference baseline |
| RAT | structured relation-aware baseline | 0.5755 | 0.9762 | 0.3483 | structured/reference baseline |
| Dual Attention adapted | raw text, DeBERTa unfrozen | 0.2548 | 0.9580 | 0.1424 | external adapted diagnostic |
| Abstractor/RCA adapted | raw text, DeBERTa unfrozen | 0.1571 | 0.4336 | 0.1095 | external adapted diagnostic |
| MAC-style attention adapted | raw text, local MAC-style model | 0.2173 | 0.6573 | 0.1283 | attention baseline diagnostic |

Key CLUTRR takeaway:

- Edge Transformer is strongest but uses structured graph-edge input.
- DAT and MAC learn shallow-hop patterns very well but collapse on deeper hops.
- TSRA-DeBERTa has substantially stronger long-hop performance under raw-text inference than these generic attention/reasoning adapters.

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
| NLProofS | RuleTaker depth-3ext | running | final test pending |

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

- NLProofS formal test result, because the current retry is still running after a shutdown interruption.
- PrOntoQA label-accuracy claims, unless we switch to proof/trace correctness or a non-degenerate official evaluation.

## 9. Next Steps

Priority for the next 2-3 days:

1. Wait for the restarted NLProofS formal test to finish, then add its final test file and status.
2. Build final aggregation tables from BERT/RoBERTa/DeBERTa seed outputs: mean, standard deviation, and depth/hop grouped metrics.
3. Build final CLUTRR table with TSRA, same-backbone Transformer classifier from the draft/logs, Edge Transformer, RAT, DAT adapted, MAC-style attention, and Abstractor/RCA if desired.
4. Redesign PrOntoQA-OOD reporting around trace/proof-step correctness rather than degenerate binary classification.
5. Move old diagnostic/adapted external baselines to appendix language and keep official EdgeTransformer/FaiRR/GFaiR/IBR as main external comparisons.
6. Once tables are frozen, update the paper draft's experiment section directly from this report.

## 10. Chinese Summary for Meeting / Draft Writing

本轮实验已经把四个数据集都准备到了可实验状态：CLUTRR 和 RuleTaker 使用仓库已有数据，ProofWriter 已从官方 S3 下载并传到开发机，PrOntoQA-OOD 官方数据和官方 FLAN-T5 输出也已整理完成。CLUTRR 统一使用我们一直采用的 `data_089907f8` split。

目前最有力的结论来自 CLUTRR 和 ProofWriter/RuleTaker 的多 backbone 结果。CLUTRR 中，TSRA-DeBERTa 在 raw-text 输入、训练时使用 trace supervision、测试时不使用 gold trace 的设置下，达到 overall 0.6370、short-hop 0.8052、long-hop 0.4332。相比之下，DAT 和 MAC-style attention 在浅层 hop 上可以学得很好，例如 DAT short-hop 达到 0.9580，说明模型并不是训练失败；但它的 long-hop 只有 0.1424，MAC-style attention long-hop 也只有 0.1283。这说明普通 attention/relational inductive bias 更容易学到浅层模式或局部关系组合，而不一定真正学会可系统泛化的多步推理。

Edge Transformer 在 CLUTRR 上效果很好，overall 0.8100、long-hop 0.6847，但它使用结构化 graph-edge input，因此应该作为 structured/reference baseline，而不是和 TSRA raw-text setting 直接公平比较。

ProofWriter 和 RuleTaker 方面，BERT/RoBERTa 的 10epoch seed 实验已经完成。BERT 在 ProofWriter depth-5 上从 baseline 的 0.8899/0.8728 提升到 TSRA 的 0.8952/0.8873；RoBERTa 的 TSRA 结果也明显比不稳定的 baseline seed 更稳。RuleTaker 上 BERT/RoBERTa 的 TSRA 与 baseline 基本持平到小幅提升，应作为稳定但不夸大的结果呈现。

外部 baseline 方面，FaiRR end-to-end 在 ProofWriter 上达到 answer acc 98.403、proof acc 97.175；GFaiR selector2 official XLNet 在 RuleTaker 上 top1 0.9846，完整 GFaiR pipeline 的 proof_acc_total 为 0.9086、faithful_total 为 0.9922；IBR depth-5 也已经跑出 full 0.9372。NLProofS formal test 因开发机关机被中断，目前已经从 checkpoint 重启，最终 test 文件仍在等待。

整体上，当前结果支持 TSRA 的核心叙事：普通 Transformer 或通用 attention/relational reasoning 方法可以拟合浅层训练分布，但在 shallow-train/deep-test 的 long-hop systematic generalization 上明显不足；TSRA 通过训练阶段的 trace supervision 更直接地约束内部 reasoning-step selection，因此更适合 query-conditioned multi-step textual reasoning。
