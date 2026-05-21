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

## 0. Live External Baseline Update

Updated on 2026-05-21 12:30 server time.

- **GFaiR Selector2 official XLNet on RuleTaker:** test completed on 40,867 examples. Top-1 accuracy is `0.9845596691707246`; top-2 accuracy is `0.9978956125969609`; invalid ratio is `0.0005969115522889963`. Result path: `/vepfs/tsra_outputs/official_external/latest_gfair_selector2_test_retry/test_result_recording.txt`.
- **GFaiR full official pipeline:** preprocessing completed. Convertor training is still running on GPU2; reasoner training is still running on GPU4. At the latest check, reasoner was about 69% through its current training pass; convertor is still active. The driver is set to launch full RuleTaker inference on GPU5 after both modules finish, so it will not collide with the TSRA seed=1 queue on GPU7. Path: `/vepfs/tsra_outputs/official_external/latest_gfair_full`.
- **NLProofS on RuleTaker depth-3ext:** official code is now patched only for environment compatibility: local T5/RoBERTa checkpoint paths, torchmetrics API, NumPy 2 `np.Inf`, and Lightning scheduler stepping. The current retry is running with prover on GPU1 and verifier on GPU6. Prover is training with T5-large; verifier is training with RoBERTa-large and GPU6 is fully utilized. Path: `/vepfs/tsra_outputs/official_external/latest_nlproofs_ruletaker`; logs: `logs/prover_train.log`, `logs/verifier_train.log`.
- **IBR on RuleTaker depth-5:** the first retry trained for part of an epoch but failed because the installed NLTK `punkt_tab` resource is a corrupt zip. The fallback tokenizer has now been patched to catch runtime NLTK failures, and IBR has been restarted at `/vepfs/tsra_outputs/official_external/latest_ibr_depth5_retry`. It is training again, but still appears mostly CPU-bound despite `CUDA_VISIBLE_DEVICES=5`.
- **TSRA-Prop DeBERTa seed=1 retry on ProofWriter/RuleTaker/PrOntoQA:** a formal 10-epoch queue is running on GPU7 to supplement the completed seed-0 DeBERTa runs for the three non-CLUTRR datasets. It is currently on `proofwriter_deberta_baseline_seed1_10ep`. Path: `/vepfs/tsra_outputs/formal_10ep/latest_prop_deberta_seed1_gpu7_retry`.
- **FaiRR end-to-end on ProofWriter:** full rule-selector + fact-selector + reasoner inference completed with local RoBERTa-large/T5-large checkpoints. Test answer accuracy is `98.4030990600586`; proof accuracy is `97.17472076416016`. Path: `/vepfs/tsra_outputs/official_external/latest_fairr_e2e_retry`.
- **EdgeTransformer on CLUTRR `data_089907f8`:** completed. Unweighted per-hop overall accuracy is `0.809951`; short-hop 2/3 accuracy is `0.976191`; long-hop 6-10 accuracy is `0.684677`. Per-hop accuracies: 2=`1.000000`, 3=`0.952381`, 4=`1.000000`, 5=`0.913793`, 6=`0.831776`, 7=`0.798611`, 8=`0.633333`, 9=`0.588235`, 10=`0.571429`. Path: `/vepfs/tsra_outputs/official_external/latest_edge_rat_clutrr_089907f8`.
- **RAT on CLUTRR `data_089907f8`:** completed as an additional relation-aware Transformer baseline. Unweighted per-hop overall accuracy is `0.575493`; short-hop 2/3 accuracy is `0.976191`; long-hop 6-10 accuracy is `0.348255`. Per-hop accuracies: 2=`1.000000`, 3=`0.952381`, 4=`0.842105`, 5=`0.643678`, 6=`0.448598`, 7=`0.451389`, 8=`0.286667`, 9=`0.319328`, 10=`0.235294`.

## 0.1 Cross-Dataset Applicability of External Baselines

The external baselines are not uniformly plug-and-play across all four TSRA datasets. Their official code is strongly tied to the input representation and supervision format of their target benchmarks.

| Method | Official / Natural Dataset | CLUTRR | ProofWriter | RuleTaker | PrOntoQA-OOD | Recommendation |
|---|---|---:|---:|---:|---:|---|
| EdgeTransformer | CLUTRR, CFQ, COGS | yes, completed | no direct support | no direct support | no direct support | Keep as CLUTRR structured graph-edge reference. Do not force onto proof datasets unless we create an oracle graph setting. |
| RAT | CLUTRR relation-aware baseline | yes, completed | no direct support | no direct support | no direct support | Keep as CLUTRR relation-aware Transformer baseline. |
| FaiRR | ProofWriter | possible only with graph/proof conversion | yes, completed | not official in current repo | not direct | Keep as ProofWriter full end-to-end baseline; possible future work is a RuleTaker adapter, but it would be local engineering rather than official reproduction. |
| GFaiR | RuleTaker variants, Hard RuleTaker, RuleTaker-E, NL satisfiability | no | not official | yes, running full pipeline | not direct | Keep as RuleTaker-family baseline; additionally run depth-5 / hard RuleTaker once current full run finishes. |
| NLProofS | EntailmentBank, ProofWriter/RuleTaker-style proof generation | no | possible | yes, running | not direct | Keep as an additional proof-generation baseline for RuleTaker; use reported/reference results if full training is too slow. |
| IBR | RuleTaker depth-5 / ParaRules-style iterative reasoning | no | not direct | yes, running | not direct | Keep as an additional RuleTaker proof-reasoning baseline if the current run produces a valid test result. |
| Abstractor/RCA | synthetic relational reasoning tasks | local raw-text adapter only | local raw-text adapter only | local raw-text adapter only | local raw-text adapter only | Use only as diagnostic if needed; not a clean official baseline for the four datasets. |
| DAT | relational/dual-attention architecture | local raw-text adapter only | local raw-text adapter only | local raw-text adapter only | local raw-text adapter only | Same as Abstractor/RCA: useful diagnostic, weak as paper-level external baseline unless adapter is carefully validated. |

Current judgment:

- We already have one strong external method per main dataset family: EdgeTransformer for CLUTRR, FaiRR for ProofWriter, GFaiR for RuleTaker.
- These external methods generally perform well on their intended datasets: EdgeTransformer has strong CLUTRR long-hop accuracy; FaiRR has very high ProofWriter answer/proof accuracy; GFaiR selector2 is very strong and full GFaiR is still running.
- The missing external-method gap is PrOntoQA-OOD. The right comparison there should be PrOntoQA-specific reported baselines or a PrOntoQA-compatible LLM/neuro-symbolic reference, not EdgeTransformer/FaiRR/GFaiR forced through an unnatural adapter.

## 2. Dataset Status

### CLUTRR

- **Role:** main entity-relation path reasoning dataset.
- **Status:** already present in the repository.
- **Required split:** `data/data_089907f8`.
- **Train split:** `1.2,1.3_train.csv`, containing 2-hop and 3-hop examples.
- **Test split:** `1.2_test.csv` through `1.10_test.csv`.
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
- **Output log:** `/root/TSRA/outputs/external_baselines/edge_transformer/data_089907f8_50ep.log`.

| Model | Overall | Short-hop | Long-hop >=6 | Notes |
|---|---:|---:|---:|---|
| Edge Transformer | 0.7618 | 0.9650 | 0.6197 | structured graph-edge input |

Per-hop accuracy:

| Hop | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Accuracy | 1.0000 | 0.9524 | 0.9842 | 0.8736 | 0.7290 | 0.7153 | 0.5733 | 0.6387 | 0.4454 |

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
- **Full official blocker:** the repo expects local `../../model/xlnet` and `../../model/T5` checkpoints, which are not currently available on the dev machine.
- **Runnable setting:** official GFaiR RuleTaker data and post-selector objective, with local DeBERTa encoder adaptation.
- **Output:** `/root/TSRA/outputs/external_baselines/gfair_deberta_selector2_1k_unfrozen.json`.

| Model | Train | Eval | Result | Notes |
|---|---:|---:|---|---|
| GFaiR Selector2 adapted DeBERTa | 1000 | dev/test 500 | dev top1_acc 0.914; test top1_acc 0.886 | component-level post-selector result |

Interpretation:

- This is a meaningful runnable GFaiR component result.
- It should be labeled as **GFaiR post-selector adapted DeBERTa**, not full official GFaiR pipeline.
- It provides a strong RuleTaker comparison point for proof-step or selector-style reasoning.

### FaiRR on ProofWriter

- **Paper:** `FaiRR: Faithful and Robust Deductive Reasoning over Natural Language`, ACL 2022.
- **Official code:** `https://github.com/INK-USC/FaiRR`.
- **Paper link:** `https://aclanthology.org/2022.acl-long.77/`.
- **Clone path:** `/root/TSRA/external_baselines/FaiRR`.
- **Dataset chosen:** ProofWriter, because FaiRR decomposes natural-language reasoning into rule selection, fact selection, and reasoning.
- **Current status:** component-level runs completed using official processed data format and local DeBERTa adaptation.

| Component | Train | Eval | Result | Output |
|---|---:|---:|---|---|
| FaiRR fact-selector adapted DeBERTa | 2000 | 1000 | dev top1_acc 0.981; test top1_acc 0.988; test token_acc 0.9956 | `outputs/external_baselines/fairr_fact_deberta_2k.json` |

Interpretation:

- This is a meaningful ProofWriter external method comparison at the component level.
- It should be labeled as **FaiRR selector adapted DeBERTa**, not full end-to-end FaiRR, until rule selector, fact selector, and reasoner are connected.
- The official rule-selector path was also verified, but the fact-selector result above is the cleaner component result for the final summary.

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

In addition to the existing CLUTRR TSRA results, I added a shared DeBERTa-base TSRA-Prop runner:

- **Script:** `/root/TSRA/scripts/transformer_tsra_prop.py`.
- **Backbone:** `microsoft/deberta-base`.
- **Design:** same encoder for baseline and TSRA; trace labels supervise sentence-selection logits during training only.
- **Test-time policy:** raw context/query only; no gold trace/path/proof is provided at inference.

These runs are useful for coverage and diagnostics, but they should be labeled preliminary because they use small limits and simplified sentence-level proof selection.

### ProofWriter DeBERTa TSRA-Prop

| Model | Train depth | Test depth | Accuracy | Trace@1 | Notes |
|---|---|---|---:|---:|---|
| DeBERTa baseline | 0/1/2 | depth-3 | 0.720 | 0.378 | small-limit run |
| DeBERTa+TSRA | 0/1/2 | depth-3 | 0.720 | 0.288 | not improved on D3 |
| DeBERTa baseline | 0/1/2 | depth-5 | 0.720 | 0.071 | deep test |
| DeBERTa+TSRA | 0/1/2 | depth-5 | 0.720 | 0.143 | trace selection improves on D5 |

### RuleTaker DeBERTa TSRA-Prop

| Model | Split | Accuracy | Trace@1 | Notes |
|---|---|---:|---:|---|
| DeBERTa baseline | dev | 0.705 | 0.125 | small-limit run |
| DeBERTa+TSRA | dev | 0.705 | 0.325 | trace-selection improvement |
| DeBERTa baseline | test | 0.720 | 0.157 | small-limit run |
| DeBERTa+TSRA | test | 0.720 | 0.229 | trace-selection improvement |

### PrOntoQA-OOD DeBERTa TSRA-Prop

| Model | OOD accuracy | Trace@1 | Notes |
|---|---:|---:|---|
| DeBERTa baseline | 1.000 | 0.189 | binary label setup is degenerate |
| DeBERTa+TSRA | 1.000 | 0.117 | current adapter/metric not adequate |

Interpretation:

- ProofWriter and RuleTaker show that the TSRA training objective can be applied beyond CLUTRR.
- The strongest preliminary signals are RuleTaker trace selection and ProofWriter depth-5 trace selection.
- PrOntoQA requires a better non-degenerate metric and task adapter before it can support final claims.

## 6. Consolidated Results Tables

### CLUTRR Main Comparison

| Model | Input setting | Overall | Short-hop | Long-hop >=6 | Paper-use status |
|---|---|---:|---:|---:|---|
| TSRA-DeBERTa | raw text + train-time trace | 0.6370 | 0.8052 | 0.4332 | main TSRA evidence |
| Edge Transformer | structured graph edges | 0.7618 | 0.9650 | 0.6197 | structured/reference baseline |
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
| DeBERTa baseline | 0/1/2 | depth-5 | acc 0.720; trace@1 0.071 | preliminary |
| DeBERTa+TSRA | 0/1/2 | depth-5 | acc 0.720; trace@1 0.143 | preliminary TSRA signal |
| FaiRR fact-selector adapted | official processed selector data | 1000 eval | test top1_acc 0.988 | component baseline |

### RuleTaker

| Model | Setting | Result | Paper-use status |
|---|---|---|---|
| DeBERTa baseline | GFaiR data, test | acc 0.720; trace@1 0.157 | preliminary |
| DeBERTa+TSRA | GFaiR data, test | acc 0.720; trace@1 0.229 | preliminary TSRA signal |
| GFaiR Selector2 adapted DeBERTa | RuleTaker post-selector | test top1_acc 0.886 | component baseline |

### PrOntoQA-OOD

| Model/reference | Setting | Result | Paper-use status |
|---|---|---|---|
| Official FLAN-T5 output analysis | 4-hop OOD composed | strict proof 0.01; relaxed proof 0.35 | reference baseline |
| DeBERTa baseline | current TSRA-Prop adapter | acc 1.000; trace@1 0.189 | diagnostic only |
| DeBERTa+TSRA | current TSRA-Prop adapter | acc 1.000; trace@1 0.117 | diagnostic only |

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
- CLUTRR Edge Transformer reproduced reference result, clearly labeled as structured graph-edge input.
- CLUTRR DAT adapted DeBERTa-unfrozen diagnostic result.
- CLUTRR MAC-style attention diagnostic result.
- RuleTaker GFaiR Selector2 adapted DeBERTa component result.
- ProofWriter FaiRR selector adapted DeBERTa component result.

### Preliminary / Appendix / Diagnostic Only

- ProofWriter/RuleTaker DeBERTa TSRA-Prop small-limit runs.
- PrOntoQA current TSRA-Prop results, because the current binary label setup is degenerate.
- Abstractor/RCA CLUTRR adapted result, unless more training/adapter refinement is done.

### Not Yet Paper-Quality

- Full FaiRR end-to-end proof inference.
- Full GFaiR official XLNet/T5 pipeline.
- Full TSRA-Prop on ProofWriter/RuleTaker/PrOntoQA with larger training, stronger encoder, and cleaner depth metrics.
- Multi-seed aggregation.

## 9. Next Steps

Priority for the next 2-3 days:

1. Build final CLUTRR table with TSRA, same-backbone Transformer classifier from the draft, Edge Transformer, DAT adapted, MAC-style attention, and Abstractor/RCA if desired.
2. Re-run or extract exact same-backbone Transformer baseline numbers from the existing draft/logs so CLUTRR raw-text comparisons are clean.
3. Extend ProofWriter and RuleTaker TSRA-Prop from small diagnostic runs to fuller DeBERTa/RoBERTa runs with depth-grouped evaluation.
4. Decide whether GFaiR/FaiRR component results are enough for the paper narrative or whether full pipelines must be completed.
5. Redesign PrOntoQA-OOD evaluation around proof correctness or step correctness rather than degenerate binary classification.
6. Add multi-seed aggregation for final reported tables.

## 10. Chinese Summary for Meeting / Draft Writing

本轮实验已经把四个数据集都准备到了可实验状态：CLUTRR 和 RuleTaker 使用仓库已有数据，ProofWriter 已从官方 S3 下载并传到开发机，PrOntoQA-OOD 官方数据和官方 FLAN-T5 输出也已整理完成。CLUTRR 统一使用我们一直采用的 `data_089907f8` split。

目前最有力的结论来自 CLUTRR。TSRA-DeBERTa 在 raw-text 输入、训练时使用 trace supervision、测试时不使用 gold trace 的设置下，达到 overall 0.6370、short-hop 0.8052、long-hop 0.4332。相比之下，DAT 和 MAC-style attention 在浅层 hop 上可以学得很好，例如 DAT short-hop 达到 0.9580，说明模型并不是训练失败；但它的 long-hop 只有 0.1424，MAC-style attention long-hop 也只有 0.1283。这说明普通 attention/relational inductive bias 更容易学到浅层模式或局部关系组合，而不一定真正学会可系统泛化的多步推理。

Edge Transformer 在 CLUTRR 上效果很好，overall 0.7618、long-hop 0.6197，但它使用结构化 graph-edge input，因此应该作为 structured/reference baseline，而不是和 TSRA raw-text setting 直接公平比较。

ProofWriter 和 RuleTaker 方面，已经跑通 FaiRR 和 GFaiR 的关键组件级 baseline。FaiRR fact-selector adapted DeBERTa 在 ProofWriter 上 test top1_acc 达到 0.988；GFaiR Selector2 adapted DeBERTa 在 RuleTaker 上 test top1_acc 达到 0.886。这些结果可以说明我们已经开始和相关 proof-reasoning 方法对比，但需要在论文中明确它们是 component-level adapted baselines，不是完整官方 pipeline。

整体上，当前结果支持 TSRA 的核心叙事：普通 Transformer 或通用 attention/relational reasoning 方法可以拟合浅层训练分布，但在 shallow-train/deep-test 的 long-hop systematic generalization 上明显不足；TSRA 通过训练阶段的 trace supervision 更直接地约束内部 reasoning-step selection，因此更适合 query-conditioned multi-step textual reasoning。
