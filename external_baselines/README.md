# External Baselines

This directory is intentionally used as a local workspace for third-party
baseline repositories and their data/checkpoints. The contents are not meant to
be committed wholesale.

Current local baseline workspaces:

- `EdgeTransformer/`: CLUTRR Edge Transformer and RAT experiments on
  `data_089907f8`.
- `FaiRR/`: ProofWriter rule/fact selector and reasoner experiments.
- `GFaiR/`: RuleTaker selector2 and full pipeline experiments.
- `NLProofS/`: RuleTaker proof-generation baseline experiment.
- `IBR/`: RuleTaker depth-5 iterative reasoning baseline experiment.
- `abstractor/`, `dual-attention/`, `clutrr-baselines/`: diagnostic or
  exploratory Transformer/attention baselines.

Large artifacts are kept under `/vepfs/tsra_outputs` or local baseline data
directories and are ignored by git. Reproduction details, result paths, and
patch notes are documented in `EXPERIMENT_REPORT.md`.
