# Experiment Setup Notes

## Data

### CLUTRR

- Status: ready, already in repo.
- Required split: `/root/TSRA/data/data_089907f8`.
- Train: `1.2,1.3_train.csv`.
- Test: `1.2_test.csv` through `1.10_test.csv`.
- Policy: shallow train on hop 2/3; deep test emphasizes held-out hops >=6.
- TSRA config: `/root/TSRA/configs/clutrr/train_tsra.yaml`.
- EdgeTransformer copy: `/root/TSRA/external_baselines/EdgeTransformer/clutrr/data/data_089907f8`.

### RuleTaker

- Status: ready, already in repo.
- Native path: `/root/TSRA/data/rule-reasoning-dataset-V2020.2.5.0/original`.
- Current config: train `depth-1,depth-2`; eval `depth-0,depth-1,depth-2,depth-3,depth-5`.
- GFaiR data path: `/root/TSRA/external_baselines/GFaiR/data/ruletaker_3ext_sat`.
- TSRA-Prop preliminary outputs: `/root/TSRA/outputs/tsra_prop/ruletaker_gfair_*.json`.

### ProofWriter

- Official source: `https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip`.
- Local staging used: `/private/tmp/tsra_data/proofwriter-dataset-V2020.12.3.zip`.
- Dev-machine raw path: `/root/TSRA/data/proofwriter/raw/proofwriter-dataset-V2020.12.3`.
- TOS target to record for reproducibility: `tos://c20250504/wy/data/proofwriter/proofwriter-dataset-V2020.12.3.zip`.
- TSRA-Prop split: train depth 0/1/2; test depth-3 and depth-5.
- Outputs: `/root/TSRA/outputs/tsra_prop/proofwriter_*.json`.

### PrOntoQA-OOD

- Official repo: `https://github.com/asaparov/prontoqa`.
- Raw path: `/root/TSRA/data/prontoqa_ood/raw/prontoqa`.
- Generated OOD data: `/root/TSRA/data/prontoqa_ood/processed/generated_ood_data`.
- Official FLAN-T5 outputs: `/root/TSRA/data/prontoqa_ood/processed/model_outputs_ood/flan-t5/latest`.
- TSRA-Prop output: `/root/TSRA/outputs/tsra_prop/prontoqa_*.json`.

## External Baselines

### EdgeTransformer

- Official code: `https://github.com/bergen/EdgeTransformer`.
- Clone path: `/root/TSRA/external_baselines/EdgeTransformer`.
- Status: reproduced on CLUTRR `data_089907f8`.
- Compatibility: old Lightning stack plus `np.Inf=np.inf` shim.
- Result log: `/root/TSRA/outputs/external_baselines/edge_transformer/data_089907f8_50ep.log`.
- Summary: overall 0.7618; short 0.9650; long >=6 0.6197.
- Note: structured graph-edge input, so report as reference baseline.

### FaiRR

- Official code: `https://github.com/INK-USC/FaiRR`.
- Clone path: `/root/TSRA/external_baselines/FaiRR`.
- Status: ProofWriter preprocessing, rule-selector debug run, and fact-selector adapted DeBERTa run completed.
- Adaptation: `roberta-large` mapped to local `microsoft/deberta-base` because official checkpoint was unavailable and cached roberta-base weights were corrupt.
- Preprocess log: `/root/TSRA/outputs/external_baselines/fairr_process_rule_deberta.log`.
- Debug log: `/root/TSRA/outputs/external_baselines/fairr_rule_deberta_debug_nockpt.log`.
- Debug result: test acc 57.89, test macro-F1 0.457.
- Fact-selector script: `/root/TSRA/scripts/fairr_deberta_selector.py`.
- Fact-selector output: `/root/TSRA/outputs/external_baselines/fairr_fact_deberta_2k.json`.
- Fact-selector result: dev top1_acc 0.981; test top1_acc 0.988; test token_acc 0.9956.

### GFaiR

- Official code: `https://github.com/spirit-moon-fly/GFaiR`.
- Clone path: `/root/TSRA/external_baselines/GFaiR`.
- Full official blocker: missing `../../model/xlnet` and `../../model/T5`.
- Smoke script: `/root/TSRA/scripts/gfair_selector2_smoke.py`.
- Smoke output: `/root/TSRA/outputs/external_baselines/gfair_selector2_smoke.json`.
- Smoke result: 128 train examples, 32 steps, test top1_acc 0.0703.
- Adapted DeBERTa script: `/root/TSRA/scripts/gfair_deberta_selector2.py`.
- Adapted DeBERTa output: `/root/TSRA/outputs/external_baselines/gfair_deberta_selector2_1k_unfrozen.json`.
- Adapted DeBERTa result: dev top1_acc 0.914; test top1_acc 0.886.
- Note: adapted GFaiR post-selector component, not full official GFaiR pipeline.

### Abstractor / Relational Cross-Attention

- Official code: `https://github.com/Awni00/abstractor`.
- Clone path: `/root/TSRA/external_baselines/abstractor`.
- Local adapter: `/root/TSRA/scripts/abstractor_rca_clutrr.py`.
- Dataset: CLUTRR `data_089907f8`.
- Frozen local result: overall 0.054; short-hop 0.098; long-hop 0.039.
- Unfrozen 3-epoch smoke: overall 0.157; short-hop 0.434; long-hop 0.110.
- Output: `/root/TSRA/outputs/external_baselines/abstractor_rca_clutrr_unfrozen_3ep.json`.
- Note: unfrozen run learns shallow CLUTRR better, but remains an adapted raw-text smoke rather than an official reproduction.

### Dual Attention Transformer

- Official code: `https://github.com/Awni00/dual-attention`.
- Clone path: `/root/TSRA/external_baselines/dual-attention`.
- Local adapter: `/root/TSRA/scripts/dual_attention_clutrr.py`.
- Dataset: CLUTRR `data_089907f8`.
- Frozen result: overall 0.065; short-hop 0.000; long-hop 0.074.
- Unfrozen 3-epoch smoke: overall 0.255; short-hop 0.958; long-hop 0.142.
- Output: `/root/TSRA/outputs/external_baselines/dual_attention_clutrr_unfrozen_3ep.json`.
- Note: uses official PyTorch `DualAttention` module with local CLUTRR raw-text adapter; unfrozen run learns shallow train distribution but deep generalization remains weak.

### MAC-style Compositional Attention

- Related paper family: MAC / compositional attention; also appears as `mac.yaml` in the official CLUTRR baseline repo.
- Official CLUTRR baseline repo: `https://github.com/koustuvsinha/clutrr-baselines`.
- Clone path: `/root/TSRA/external_baselines/clutrr-baselines`.
- Official blocker: old missing dependencies (`addict`, `comet_ml`, `torch_geometric`, `pytorch_pretrained_bert`).
- Local adapter: `/root/TSRA/scripts/mac_attention_clutrr.py`.
- Dataset: CLUTRR `data_089907f8`.
- Result: overall 0.217; short-hop 0.657; long-hop 0.128.
- Output: `/root/TSRA/outputs/external_baselines/mac_attention_clutrr_20ep.json`.
- Note: more reliable than DAT/Abstractor smoke adapters because training loss clearly decreases, but still adapted rather than official reproduction.

## TSRA-Prop Preliminary Runs

- Script: `/root/TSRA/scripts/generic_tsra_prop.py`.
- ProofWriter: depth-3 trace@1 improves 0.1825 -> 0.2436; depth-5 not improved.
- RuleTaker: dev trace@1 improves 0.0861 -> 0.1236; test not improved.
- PrOntoQA-OOD: classification accuracy is degenerate; trace@1 did not improve.
- These are preliminary coverage runs, not final paper-quality TSRA numbers.

## Artifact Policy

- Do not commit large data, checkpoints, third-party `.git` histories, or logs.
- Checkpoints generated by debug runs should be deleted after metrics are recorded.
- Current root disk is tight (about 98% used); avoid new full checkpoints until storage is freed.
