# Full DeBERTa TSRA Runs: Non-CLUTRR

Started full DeBERTa runs for ProofWriter, RuleTaker, and PrOntoQA-OOD.

## Run Setup

- Backbone: `microsoft/deberta-base`
- Encoder: unfrozen
- Baseline: `lambda_trace=0.0`
- TSRA: `lambda_trace=1.0`
- Checkpoints: not saved
- Outputs: JSON + log only
- Remote output directory: `/root/TSRA/outputs/transformer_tsra_prop/full_deberta`

## Active Long Runs

| Dataset | Variant | GPU | PID | Output |
|---|---|---:|---:|---|
| ProofWriter | DeBERTa baseline | 0 | 3804 | `proofwriter_deberta_baseline_full.json` |
| ProofWriter | DeBERTa+TSRA | 1 | 3807 | `proofwriter_deberta_tsra_full.json` |
| RuleTaker | DeBERTa baseline | 2 | 3810 | `ruletaker_deberta_baseline_full.json` |
| RuleTaker | DeBERTa+TSRA | 3 | 3813 | `ruletaker_deberta_tsra_full.json` |

## Completed: PrOntoQA-OOD

| Model | Train | Test | Accuracy | Trace@1 | By depth |
|---|---:|---:|---:|---:|---|
| DeBERTa baseline | 200 | 300 | 1.000 | 0.2367 | D3 1.000, D4 1.000 |
| DeBERTa+TSRA | 200 | 300 | 1.000 | 0.5633 | D3 1.000, D4 1.000 |

Interpretation: classification accuracy remains degenerate on the current PrOntoQA adapter, but TSRA substantially improves proof-step/trace selection on the OOD set.
