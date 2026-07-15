# Experiment Scripts

This directory contains the lightweight orchestration scripts used for the TRUA
experiment campaign. Generated logs, outputs, checkpoints, and downloaded
models should stay under `/vepfs` and are ignored by git.

Primary entry points:

- `run_clutrr_backbone_core_matrix.py`: BERT/RoBERTa/DeBERTa-v3 comparison
  of the encoder, objective-matched content-only unit attention, and TRUA under
  a declared checkpoint protocol.
- `run_proposition_backbone_core_matrix.py`: the corresponding ProofWriter and
  RuleTaker backbone/core matrix; the content-only control receives the same
  evidence objective as TRUA but no query route in unit selection.
- `run_clutrr_protocol_audit.py`: controlled full/held-out and
  final/validation checkpoint audit used to reconcile historical CLUTRR
  scores with the current protocol.
- `formal_trua_10ep_supervisor.sh`: formal 10-epoch TRUA/label-only queue.
- `transformer_trua_prop.py`: DeBERTa/RoBERTa-style TRUA-Prop runner for
  ProofWriter, RuleTaker, and PrOntoQA.
- `run_controlled_clutrr_baselines.py`: same-protocol encoder, content
  self-attention, MAC-style, and RCA-style CLUTRR controls.
- `run_proofwriter_query_controls.py`: three-seed ProofWriter controls that
  separate explicit goal injection, the query-anchor route, and matched
  content self-attention.
- `run_counterfactual_clutrr_control.py`: validation-selected, three-seed
  CLUTRR control for CREST-inspired rename/reverse-query augmentation. This is
  explicitly an adapted shortcut-mitigation control, not an official CREST
  reproduction.
- `run_prop_deberta_seed1_gpu7_retry.sh`: current seed-1 10-epoch retry queue
  for non-CLUTRR datasets.
- `run_additional_depth_seed_checks.sh`: follow-up depth/seed queue covering
  CLUTRR `data_db9b8f04` 2/3/4-hop train and strict RuleTaker raw QDep 1/2
  train with seeds `0/1/42`.
- `run_seed42_completion_after_additional.sh`: waits for the additional
  depth/seed queue, then fills missing seed-42 runs for the main formal
  CLUTRR/ProofWriter/RuleTaker/PrOntoQA tables.
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
