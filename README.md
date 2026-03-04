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

comparison/
  other_paper_method/
    code/
      analysis/
      cli/
      diagnostics/
      experiments/
      gpt_baselines/
      scl/
    FILES.txt
    remove_other_method_files.sh
```

## Entrypoints

- `python -m clutrr.cli.train ...`
- `python -m ruletaker.cli.train ...`
- `python -m comparison.other_paper_method.code.cli.baseline ...` (for prior-paper comparison)

## Notes

- `clutrr` and `ruletaker` are peer packages for two different datasets.
- No root-level compatibility launchers are required; use module entrypoints above.
