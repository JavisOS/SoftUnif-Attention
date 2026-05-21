# Experiment Scripts

This directory contains the lightweight orchestration scripts used for the TSRA
experiment campaign. Generated logs, outputs, checkpoints, and downloaded
models should stay under `/vepfs` and are ignored by git.

Primary entry points:

- `formal_tsra_10ep_supervisor.sh`: formal 10-epoch TSRA/label-only queue.
- `transformer_tsra_prop.py`: DeBERTa/RoBERTa-style TSRA-Prop runner for
  ProofWriter, RuleTaker, and PrOntoQA.
- `run_prop_deberta_seed1_gpu7_retry.sh`: current seed-1 10-epoch retry queue
  for non-CLUTRR datasets.
- `run_edge_rat_clutrr_089907f8.sh`: EdgeTransformer/RAT CLUTRR baseline on
  the required `data_089907f8` split.
- `run_gfair_full_preprocess_train.sh`: GFaiR full RuleTaker pipeline.
- `run_nlproofs_ruletaker.sh`: NLProofS RuleTaker depth-3ext pipeline.
- `run_more_external_eval.sh`, `run_fairr_*`, `run_remaining_official_modules.sh`:
  FaiRR/GFaiR official external-baseline follow-ups.

Diagnostic/adapted baselines:

- `abstractor_rca_clutrr.py`, `dual_attention_clutrr.py`,
  `mac_attention_clutrr.py`: CLUTRR adapters for attention-style architectures.
- `fairr_deberta_selector.py`, `gfair_deberta_selector2.py`: adapted component
  checks, not the main official-paper baselines.

Obsolete smoke/5-epoch scripts were removed from the working tree to keep the
git status focused on reproducible experiment entry points.
