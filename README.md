# SoftUnif-Attention (TRUA Research Codebase)

This repository is organized with a layered scaffold (`config / data / models / training / utils`) per dataset package:

- `clutrr/`: TRUA on CLUTRR (your method).
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
    trua_dataset.py
    trua_collator.py
  models/
    backbones.py
    relation_attention.py
    trua_model.py
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

- `python -m clutrr.cli.train --config configs/clutrr/train_trua.yaml`
- `python -m clutrr.cli.baseline --config configs/clutrr/train_baseline.yaml`
- `python -m ruletaker.cli.train --config configs/ruletaker/train.yaml`
- `python -m ruletaker.cli.baseline --config configs/ruletaker/train_baseline.yaml`
- `python -m clutrr.cli.eval_gemini_openai --config configs/clutrr/eval_gemini_openai.yaml`
- CLI args still override YAML, e.g. `python -m clutrr.cli.train --config configs/clutrr/train_trua.yaml --epochs 30`

## Notes

- `clutrr` and `ruletaker` are peer packages for two different datasets.
- No root-level compatibility launchers are required; use module entrypoints above.


## CLUTRR TRUA Lines

The reproducible DeBERTa TRUA baseline is configured in `configs/clutrr/train_trua.yaml`. The sparse latent relation transition algebra experiment is configured in `configs/clutrr/train_trua_sparse.yaml`.

Run the baseline:

    python -m clutrr.cli.train --config configs/clutrr/train_trua.yaml

Run the sparse algebraic reasoner:

    python -m clutrr.cli.train --config configs/clutrr/train_trua_sparse.yaml

Useful sparse-reasoner ablation switches are exposed directly by `clutrr.cli.train`:

    python -m clutrr.cli.train --config configs/clutrr/train_trua_sparse.yaml --no-use_path_algebra
    python -m clutrr.cli.train --config configs/clutrr/train_trua_sparse.yaml --lambda_alg 0 --lambda_eq 0
    python -m clutrr.cli.train --config configs/clutrr/train_trua_sparse.yaml --relation_score_mode mlp
    python -m clutrr.cli.train --config configs/clutrr/train_trua_sparse.yaml --sparse_top_k 0

Large generated artifacts are intentionally not versioned. Experiment logs and checkpoints should live under outputs/, and local datasets/environments under data/ or .conda/.
