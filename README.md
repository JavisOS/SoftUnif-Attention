# SoftUnif-Attention (TSRA Research Codebase)

This repository is organized with a layered scaffold (`config / data / models / training / utils`) per dataset package:

- `clutrr/`: TSRA on CLUTRR (your method).
- `ruletaker/`: RuleTaker pipeline.
- `comparison/other_paper_method/code/`: prior-paper code kept only for comparison.

## Project Layout

```text
clutrr/
  config/
    defaults.py
    relation_schema.py
  data/
    clutrr_dataset.py
    tsra_dataset.py
    tsra_collator.py
  models/
    backbones.py
    relation_attention.py
    tsra_model.py
  training/
    robustness.py
  utils/
    parsing.py
    seed.py
    distributed.py
    entity_alignment.py
    graph_reasoning.py
  cli/
    train.py
    baseline.py
  preprocess/

ruletaker/
  config/
    defaults.py
  models/
    backbones.py
  utils/
    distributed.py
  cli/
    train.py
    baseline.py

comparison/
  other_paper_method/
    code/
      analysis/
      diagnostics/
      experiments/
      gpt_baselines/
      scl/
    FILES.txt
    remove_other_method_files.sh
```

## Entrypoints

- `python -m clutrr.cli.train ...`
- `python -m clutrr.cli.baseline ...`
- `python -m ruletaker.cli.train ...`
- `python -m ruletaker.cli.baseline ...`

Example with YAML config:

- `python -m clutrr.cli.train --config configs/clutrr/train_tsra.yaml`
- `python -m clutrr.cli.baseline --config configs/clutrr/train_baseline.yaml`
- `python -m ruletaker.cli.train --config configs/ruletaker/train.yaml`
- `python -m ruletaker.cli.baseline --config configs/ruletaker/train_baseline.yaml`
- `python -m clutrr.cli.eval_gemini_openai --config configs/clutrr/eval_gemini_openai.yaml`
- CLI args still override YAML, e.g. `python -m clutrr.cli.train --config configs/clutrr/train_tsra.yaml --epochs 30`

## Notes

- `clutrr` and `ruletaker` are peer packages for two different datasets.
- No root-level compatibility launchers are required; use module entrypoints above.


## TSRA-v3 Quick Start

The current mainline TSRA variant is configured in configs/clutrr/train_tsra_v3_roberta.yaml. It uses multi-mention entity pooling, gated fusion between the global sequence head and entity-pair head, and direct supervision of latent relation logits on gold path edges.

Run a fast RoBERTa diagnostic:

    python -m clutrr.cli.train --config configs/clutrr/train_tsra_v3_roberta.yaml

Useful ablation switches are exposed directly by clutrr.cli.train:

    python -m clutrr.cli.train --config configs/clutrr/train_tsra_v3_roberta.yaml --prediction_head cls_only
    python -m clutrr.cli.train --config configs/clutrr/train_tsra_v3_roberta.yaml --prediction_head pair_only
    python -m clutrr.cli.train --config configs/clutrr/train_tsra_v3_roberta.yaml --edge_supervision_target separate
    python -m clutrr.cli.train --config configs/clutrr/train_tsra_v3_roberta.yaml --no-use_relation_conditioning

Large generated artifacts are intentionally not versioned. Experiment logs and checkpoints should live under outputs/, and local datasets/environments under data/ or .conda/.
