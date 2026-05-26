# Experiment Setup Notes

This file is the lightweight operational index for the TSRA experiment
workspace. The full result tables and interpretation live in
`EXPERIMENT_REPORT.md`; this file is meant to answer "where is the data, what
script runs what, and where are the artifacts?" without duplicating every table.

## Data

### CLUTRR

- Status: ready, already in repo.
- Required split: `/root/TSRA/data/data_089907f8`.
- Train: `1.2,1.3_train.csv`.
- Test: `1.2_test.csv` through `1.10_test.csv`.
- Policy: shallow train on hop 2/3; deep test emphasizes held-out hops >=6.
- Follow-up split: `/root/TSRA/data/data_db9b8f04`, with
  `1.2,1.3,1.4_train.csv` for 2/3/4-hop training.
- TSRA config: `/root/TSRA/configs/clutrr/train_tsra.yaml`.
- EdgeTransformer copy: `/root/TSRA/external_baselines/EdgeTransformer/clutrr/data/data_089907f8`.
- Main outputs: `/vepfs/tsra_outputs/formal_10ep/latest_tsra_formal_10ep` and
  `/vepfs/tsra_outputs/official_external/latest_edge_rat_clutrr_089907f8`.

### RuleTaker

- Status: ready, already in repo.
- Native path: `/root/TSRA/data/rule-reasoning-dataset-V2020.2.5.0/original`.
- Current TSRA split: shallow train on low proof depth; eval includes depth-3ext
  and depth-5 style settings where available.
- Strict raw-depth follow-up: `scripts/transformer_tsra_prop.py --dataset
  ruletaker_raw` can filter by question-level QDep via `--train-qdeps` and
  `--test-qdeps`. The current follow-up queue trains QDep 1/2 and tests QDep
  1-5.
- GFaiR data path: `/root/TSRA/external_baselines/GFaiR/data/ruletaker_3ext_sat`.
- Main TSRA outputs:
  `/vepfs/tsra_outputs/formal_10ep/latest_prop_bert_backbone_seeds/results`,
  `/vepfs/tsra_outputs/formal_10ep/latest_prop_roberta_backbone_seeds/results`,
  `/vepfs/tsra_outputs/formal_10ep/latest_tsra_formal_10ep/results`, and
  `/vepfs/tsra_outputs/formal_10ep/latest_prop_deberta_seed1_failed_rerun/results`.
- External outputs:
  `/vepfs/tsra_outputs/official_external/latest_gfair_selector2_test_retry`,
  `/vepfs/tsra_outputs/official_external/gfair_full_20260521_085403`, and
  `/vepfs/tsra_outputs/official_external/latest_ibr_depth5_retry2`.

### ProofWriter

- Official source: `https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip`.
- Local staging used: `/private/tmp/tsra_data/proofwriter-dataset-V2020.12.3.zip`.
- Dev-machine raw path: `/root/TSRA/data/proofwriter/raw/proofwriter-dataset-V2020.12.3`.
- TOS target to record for reproducibility:
  `tos://c20250504/wy/data/proofwriter/proofwriter-dataset-V2020.12.3.zip`.
- TSRA split: train depth 0/1/2; test depth-3 and depth-5.
- Main TSRA outputs:
  `/vepfs/tsra_outputs/formal_10ep/latest_prop_bert_backbone_seeds/results`,
  `/vepfs/tsra_outputs/formal_10ep/latest_prop_roberta_backbone_seeds/results`,
  `/vepfs/tsra_outputs/formal_10ep/latest_tsra_formal_10ep/results`, and
  `/vepfs/tsra_outputs/formal_10ep/latest_prop_deberta_seed1_failed_rerun/results`.
- External FaiRR output:
  `/vepfs/tsra_outputs/official_external/latest_fairr_e2e_retry`.

### PrOntoQA-OOD

- Official repo: `https://github.com/asaparov/prontoqa`.
- Raw path: `/root/TSRA/data/prontoqa_ood/raw/prontoqa`.
- Generated OOD data: `/root/TSRA/data/prontoqa_ood/processed/generated_ood_data`.
- Official FLAN-T5 outputs:
  `/root/TSRA/data/prontoqa_ood/processed/model_outputs_ood/flan-t5/latest`.
- TSRA split: generated OOD/compositional examples with depth metadata where
  available; report label accuracy and trace/proof-step selection separately.
- Main TSRA outputs:
  `/vepfs/tsra_outputs/formal_10ep/latest_prop_bert_backbone_seeds/results`,
  `/vepfs/tsra_outputs/formal_10ep/latest_prop_roberta_backbone_seeds/results`,
  `/vepfs/tsra_outputs/formal_10ep/latest_tsra_formal_10ep/results`, and
  `/vepfs/tsra_outputs/formal_10ep/latest_prop_deberta_seed1_failed_rerun/results`.

## External Baselines

### EdgeTransformer

- Official code: `https://github.com/bergen/EdgeTransformer`.
- Clone path: `/root/TSRA/external_baselines/EdgeTransformer`.
- Dataset: CLUTRR `data_089907f8`.
- Status: reproduced with EdgeTransformer and RAT reference variants.
- Current result path:
  `/vepfs/tsra_outputs/official_external/latest_edge_rat_clutrr_089907f8`.
- Current headline result: EdgeTransformer overall `0.809951`, short-hop
  `0.976191`, long-hop `0.684677`.
- Note: uses structured graph-edge input, so report as a strong reference
  baseline rather than a raw-text same-backbone comparison.

### GFaiR

- Official code: `https://github.com/spirit-moon-fly/GFaiR`.
- Clone path: `/root/TSRA/external_baselines/GFaiR`.
- Dataset: RuleTaker.
- Status: official selector2 and full pipeline were run.
- Selector2 result path:
  `/vepfs/tsra_outputs/official_external/latest_gfair_selector2_test_retry/test_result_recording.txt`.
- Full pipeline result path:
  `/vepfs/tsra_outputs/official_external/gfair_full_20260521_085403/full_inference_ruletaker_3ext_retry_after_reboot_bs8/test_result_recording.txt`.
- Current headline result: selector2 top1 `0.984560`; full pipeline proof
  accuracy `0.908629`, faithful score `0.992208`.

### FaiRR

- Official code: `https://github.com/INK-USC/FaiRR`.
- Clone path: `/root/TSRA/external_baselines/FaiRR`.
- Dataset: ProofWriter.
- Status: end-to-end official-style run completed.
- Current result path:
  `/vepfs/tsra_outputs/official_external/latest_fairr_e2e_retry`.
- Current headline result: answer accuracy `98.403099`, proof accuracy
  `97.174721`.

### IBR

- Clone path: `/root/TSRA/external_baselines/IBR`.
- Dataset: RuleTaker depth-5.
- Status: completed as an additional proof-reasoning reference baseline.
- Current result path:
  `/vepfs/tsra_outputs/official_external/latest_ibr_depth5_retry2/output/test_records.txt`.
- Current headline result: QA `0.994153`, proof `0.937416`, full `0.937169`.

### NLProofS

- Clone path: `/root/TSRA/external_baselines/NLProofS`.
- Dataset: RuleTaker / ProofWriter-style depth-3ext data.
- Status: training is complete; formal test is running from the trained
  checkpoint after a dev-machine shutdown interrupted the previous test pass.
- Current retry symlink:
  `/vepfs/tsra_outputs/official_external/latest_nlproofs_ruletaker_test`.
- Do not cite a final test score until a `results_test*.json` file appears.

### Diagnostic Attention/Transformer Baselines

- Abstractor / Relational Cross-Attention:
  `/root/TSRA/scripts/abstractor_rca_clutrr.py`.
- Dual Attention Transformer:
  `/root/TSRA/scripts/dual_attention_clutrr.py`.
- MAC-style compositional attention:
  `/root/TSRA/scripts/mac_attention_clutrr.py`.
- Dataset for these adapters: CLUTRR `data_089907f8`.
- These are adapted raw-text diagnostics, not official reproductions; use them
  to discuss shallow-vs-deep generalization behavior only with that caveat.

## Script Layout

- `scripts/README.md`: maintained index for experiment launchers.
- `scripts/formal_tsra_10ep_supervisor.sh`: formal 10-epoch CLUTRR TSRA
  ablation queue.
- `scripts/transformer_tsra_prop.py`: shared TSRA proposition/proof-step runner
  for ProofWriter, RuleTaker, and PrOntoQA.
- `scripts/run_*`: thin launchers for formal TSRA and external-baseline runs.
- `scripts/run_additional_depth_seed_checks.sh`: CLUTRR `data_db9b8f04`,
  strict RuleTaker raw-QDep, and DeBERTa seed-42 follow-up queue.
- `scripts/run_seed42_completion_after_additional.sh`: waits for the current
  follow-up queue, then fills missing seed-42 runs so main tables can aggregate
  seeds `0/1/42`.
- `external_baselines/README.md`: index of local third-party workspaces.

## Artifact Policy

- Commit code, configuration, report files, and small reproducibility notes.
- Do not commit benchmark data, checkpoints, third-party `.git` histories, run
  logs, generated tensors, local virtual environments, or platform-specific
  wheels.
- Keep large data, checkpoints, and logs under `/vepfs/tsra_outputs` or the
  ignored local dataset/baseline directories.
- Root-level scratch files should use `.tmp_*` and remain ignored.
