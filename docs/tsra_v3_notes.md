# TSRA-v3 Notes

TSRA-v3 is the cleaned mainline variant after the May 2026 diagnostic sweep.

## Mainline design

- Multi-mention entity pooling for CLUTRR entity representations.
- Query-guided entity attention with trace supervision on next-hop logits.
- Direct latent relation supervision: gold path edge labels supervise the relation logits used inside the attention layer.
- Gated prediction head: a learned scalar combines global CLS logits and subject-object entity-pair logits.
- Renaming consistency remains an auxiliary robustness objective.

## Diagnostic interpretation

The RoBERTa diagnostic sweep showed that trace supervision should be treated as structured regularization and a diagnostic signal, not as proof of faithful explicit reasoning. Final-answer correctness and exact trace alignment can diverge. The most useful code-level change was direct supervision of latent relation logits, especially when combined with gated fusion.

## Important command

    python -m clutrr.cli.train --config configs/clutrr/train_tsra_v3_roberta.yaml

## Archived local artifacts

Generated logs/checkpoints were moved out of the repository to /root/TSRA_cleanup_archive/20260519/outputs on the development machine. They are not intended for GitHub.
