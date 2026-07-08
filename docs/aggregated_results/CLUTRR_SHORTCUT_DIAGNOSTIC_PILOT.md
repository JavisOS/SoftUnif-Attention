# CLUTRR Shortcut-Reasoning Diagnostic Pilot

Run root: `/vepfs/tsra_outputs/clutrr_shortcut_diagnostic/clutrr_shortcut_diag_20260626_011004`

Method: pilot token-occlusion approximation of Haraguchi et al. (2023) shortcut-reasoning diagnostics.
This is not an official IG/input-reduction reproduction. IID is CLUTRR short-hop test examples (2/3-hop); OOD is long-hop test examples (>=6-hop).

| Variant | Seed | Epochs | IID short acc | OOD long acc | Supported patterns | Shortcut count | Shortcut rate | Avg shortcut delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| crest | 0 | 3 | 0.9580 | 0.4836 | 8 | 0 | 0.0000 | 0.0000 |
| tsra | 0 | 3 | 0.7435 | 0.3075 | 61 | 0 | 0.0000 | 0.0000 |
| vanilla | 0 | 3 | 0.9580 | 0.1565 | 4 | 0 | 0.0000 | 0.0000 |

## Top Patterns: crest

| Pattern | Label | IID support | OOD support | IID acc | OOD pred rate | OOD acc | Delta | Shortcut |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `lisa` | daughter | 9 | 37 | 0.6667 | 0.0000 | 0.2973 | -0.1863 | False |
| `lisa` | granddaughter | 9 | 37 | 0.2222 | 0.0541 | 0.2973 | -0.1863 | False |
| `grandson` | grandson | 32 | 16 | 0.6875 | 0.0000 | 0.3750 | -0.1086 | False |
| `son` | grandson | 53 | 508 | 0.0189 | 0.0276 | 0.4744 | -0.0092 | False |
| `mother` | granddaughter | 20 | 234 | 0.2000 | 0.0043 | 0.5000 | 0.0164 | False |
| `father` | grandson | 32 | 220 | 0.1562 | 0.0227 | 0.5091 | 0.0255 | False |
| `father` | daughter | 32 | 220 | 0.0938 | 0.0364 | 0.5091 | 0.0255 | False |
| `grandfather` | grandson | 20 | 24 | 0.2000 | 0.0417 | 0.5417 | 0.0581 | False |

## Top Patterns: tsra

| Pattern | Label | IID support | OOD support | IID acc | OOD pred rate | OOD acc | Delta | Shortcut |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `mark` | aunt | 16 | 10 | 0.0625 | 0.0000 | 0.1000 | -0.2075 | False |
| `husband` | daughter | 56 | 57 | 0.0893 | 0.1404 | 0.1579 | -0.1496 | False |
| `husband` | daughter-in-law | 56 | 57 | 0.0179 | 0.0000 | 0.1579 | -0.1496 | False |
| `grandson` | grandson | 41 | 6 | 0.5366 | 0.3333 | 0.1667 | -0.1408 | False |
| `grandson` | mother | 41 | 6 | 0.0488 | 0.0000 | 0.1667 | -0.1408 | False |
| `grandson` | grandmother | 41 | 6 | 0.0488 | 0.0000 | 0.1667 | -0.1408 | False |
| `kevin` | grandfather | 9 | 23 | 0.2222 | 0.0435 | 0.1739 | -0.1336 | False |
| `anthony` | grandson | 11 | 21 | 0.0909 | 0.0952 | 0.2381 | -0.0694 | False |
| `grandmother` | mother | 37 | 61 | 0.1081 | 0.0492 | 0.2459 | -0.0616 | False |
| `grandmother` | grandmother | 37 | 61 | 0.0811 | 0.0492 | 0.2459 | -0.0616 | False |
| `grandmother` | grandson | 37 | 61 | 0.0541 | 0.0656 | 0.2459 | -0.0616 | False |
| `tommy` | daughter | 5 | 18 | 0.2000 | 0.1111 | 0.2778 | -0.0297 | False |

## Top Patterns: vanilla

| Pattern | Label | IID support | OOD support | IID acc | OOD pred rate | OOD acc | Delta | Shortcut |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `lisa` | daughter | 9 | 37 | 0.6667 | 0.0270 | 0.1081 | -0.0484 | False |
| `lisa` | granddaughter | 9 | 37 | 0.2222 | 0.0000 | 0.1081 | -0.0484 | False |
| `james` | father | 9 | 28 | 0.2222 | 0.1429 | 0.1786 | 0.0221 | False |
| `valerie` | father | 7 | 17 | 1.0000 | 0.0000 | 0.5882 | 0.4317 | False |
