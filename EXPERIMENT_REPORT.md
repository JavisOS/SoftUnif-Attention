# TRUA Final Experiment Report

## 1. Status and Scope

This report is the authoritative experimental record for the current TRUA paper. It supersedes earlier exploratory summaries, backbone sweeps, external-reference runs, and test-selected tables.

- Final run root: `/vepfs/trua_outputs/paper_evidence_alignment/main_20260710`
- Final experiment revision: `64f8ae0a7b0757bd6c05d1ac1c1402c10c622eba`
- Audited-evidence commit: `0926c6188dc60931f40613b746d82ba7f57e1c71`
- Runs: 51/51 complete, 0 failed
- Groups: 17/17 complete
- Seeds: 0, 1, 42
- Schedule: 10 epochs for every final run
- Selection: validation-only checkpoint selection; each test split is evaluated once after selection
- Audit: PASS, with one revision across all 51 result artifacts

The final protocol evaluates two adapter families:

1. An entity-path adapter on CLUTRR, with ordered next-unit transitions and optional edge labels.
2. A proposition adapter on ProofWriter, RuleTaker, and PrOntoQA-OOD, with unordered reference evidence sets.

The maintained implementation uses a shared query-guided unit-attention core. Task adapters define units, grounding, supervision targets, and prediction heads.

## 2. Protocol Audit

The final audit checks the task manifest, expected seeds, code revision, ten-epoch validation histories, selected checkpoints, model and objective configurations, split sizes, input limits, official CLUTRR task-length totals, and reference-evidence denominators.

- CLUTRR: 9,083 train, 1,011 validation, 1,146 test; official task lengths 2/3 at train and 2--10 at test.
- CLUTRR test totals by official length: 38, 105, 190, 174, 107, 144, 150, 119, and 119 for lengths 2 through 10.
- ProofWriter OWA: 30,000 train and 5,000 development examples from the standard depth-0/1/2 directories; 5,000 test examples each from the depth-3 and depth-5 directory configurations.
- RuleTaker raw: 30,000 train and 5,000 development examples at QDep 1/2; 5,000 test examples at QDep 1--5.
- PrOntoQA-OOD: 180 train, 20 validation, and 300 OOD `ProofsOnly` test examples.
- Reference-evidence test denominators: 3,627 and 3,877 for ProofWriter depth-3/depth-5; 5,000 for RuleTaker; 300 for PrOntoQA-OOD.
- No selected proposition example exceeds the 512-token context-query limit, 96-token sentence limit, or 64-token query limit.
- No duplicate identifiers occur within the audited ProofWriter or RuleTaker splits, and no identifiers overlap across their train, development, and test partitions.

## 3. CLUTRR Controlled Objective Comparison

All four rows below use DeBERTa-base, the same unit-attention architecture, original plus consistently entity-renamed final-label views, the same validation split, and the same test evaluation. Only the displayed path-based objectives change. Values are mean +/- sample standard deviation.

| Objective | Overall | Short 2--3 | Long 6--10 | Transition@1 |
| --- | ---: | ---: | ---: | ---: |
| Answer-only unit attention | 0.4229 +/- 0.0328 | 0.8042 +/- 0.0070 | 0.3177 +/- 0.0405 | 0.1968 +/- 0.0104 |
| Transition regularization only | 0.5218 +/- 0.0334 | 0.8228 +/- 0.0225 | 0.4100 +/- 0.0316 | 0.5114 +/- 0.0121 |
| Edge supervision only | 0.5576 +/- 0.0091 | 0.8485 +/- 0.0359 | 0.4236 +/- 0.0176 | 0.1737 +/- 0.0137 |
| TRUA, transition + edge | 0.6012 +/- 0.0245 | 0.8135 +/- 0.0146 | 0.4799 +/- 0.0268 | 0.5076 +/- 0.0049 |

Paired TRUA-minus-answer-only differences are positive for every seed: +0.170/+0.190/+0.175 overall, +0.166/+0.174/+0.147 on long tasks, and +0.308/+0.317/+0.307 for Transition@1. Transition-only training raises long-task accuracy by 0.092 and Transition@1 by 0.315 in the three-seed mean. Edge-only training raises answer accuracy without improving Transition@1. Their combination gives the highest mean answer accuracy at every unseen official length from 4 through 10.

The conventional vanilla encoder reaches 0.2830 +/- 0.0088 overall and 0.1544 +/- 0.0248 on long tasks. This row is descriptive only: unlike the controlled unit-attention rows, it does not train on the consistently renamed final-label view.

### Architecture Diagnostics

| Variant | Overall | Long 6--10 | Transition@1 |
| --- | ---: | ---: | ---: |
| TRUA | 0.6012 +/- 0.0245 | 0.4799 +/- 0.0268 | 0.5076 +/- 0.0049 |
| No explicit goal term | 0.6117 +/- 0.0193 | 0.5018 +/- 0.0268 | 0.5214 +/- 0.0056 |
| No aggregation branch | 0.5905 +/- 0.0207 | 0.4721 +/- 0.0304 | 0.5052 +/- 0.0287 |
| No selection message/objective | 0.6108 +/- 0.0402 | 0.5076 +/- 0.0495 | 0.1427 +/- 0.0056 |
| No relation conditioning | 0.5960 +/- 0.0361 | 0.4710 +/- 0.0478 | 0.4946 +/- 0.0103 |

These rows do not establish that each branch is independently necessary. The entity adapter jointly encodes context and query, so removing only the explicit goal term is a weak goal-information ablation. The no-selection result shows that the global and endpoint answer heads can compensate even when the learned selection scores cease to align with the reference path. A consistency weight of 5 is excluded because it lowers validation accuracy for every seed.

## 4. Proposition-Unit Diagnostics

Evidence@1 is the fraction of examples with a nonempty reference evidence set whose highest-scoring candidate sentence belongs to that set. It is directly optimized in the TRUA rows and is not a causal-faithfulness measure.

### ProofWriter

| Variant | Depth-3 accuracy | Depth-3 Evidence@1 | Depth-5 accuracy | Depth-5 Evidence@1 |
| --- | ---: | ---: | ---: | ---: |
| No evidence regularization | 0.9427 +/- 0.0033 | 0.1936 +/- 0.0145 | 0.9091 +/- 0.0020 | 0.1528 +/- 0.0351 |
| No query in unit selection | 0.9423 +/- 0.0067 | 0.3831 +/- 0.0091 | 0.9093 +/- 0.0037 | 0.4304 +/- 0.0119 |
| TRUA | 0.9359 +/- 0.0125 | 0.8698 +/- 0.0129 | 0.8997 +/- 0.0198 | 0.8814 +/- 0.0078 |

Evidence regularization sharply improves reference-evidence selection but does not improve final-answer accuracy on these configurations. Removing the query while keeping the evidence objective reduces Evidence@1 by 0.487 on depth-3 and 0.451 on depth-5, with comparable answer accuracy. This is the cleanest final diagnostic for query-guided selection.

The directory names are dataset configurations, not exact per-example proof depths. Each ProofWriter directory contains questions with multiple individual proof depths.

### RuleTaker

| Variant | Test accuracy | Evidence@1 |
| --- | ---: | ---: |
| No evidence regularization | 0.5575 +/- 0.0553 | 0.2907 +/- 0.0138 |
| TRUA | 0.8303 +/- 0.1292 | 0.8947 +/- 0.0049 |

| Variant | QDep1 | QDep2 | QDep3 | QDep4 | QDep5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| No evidence regularization | 0.5420 +/- 0.0216 | 0.5744 +/- 0.0749 | 0.5619 +/- 0.1029 | 0.6039 +/- 0.0858 | 0.5849 +/- 0.1951 |
| TRUA | 0.8983 +/- 0.1523 | 0.8768 +/- 0.1764 | 0.6807 +/- 0.0505 | 0.5399 +/- 0.0399 | 0.5031 +/- 0.0655 |

TRUA improves overall accuracy and Evidence@1, but its answer accuracy has high seed variance. It does not improve QDep4 or QDep5 in the three-seed mean, and QDep5 remains near chance. RuleTaker therefore supports evidence alignment and some shallow-to-intermediate transfer, not a claim that deep proof generalization is solved.

### PrOntoQA-OOD

| Variant | OOD accuracy | Evidence@1 |
| --- | ---: | ---: |
| No evidence regularization | 1.0000 +/- 0.0000 | 0.2867 +/- 0.0982 |
| TRUA | 1.0000 +/- 0.0000 | 0.4644 +/- 0.0222 |

The selected `ProofsOnly` files contain only entailed queries. Accuracy is consequently a ceiling check, not an answer-discrimination result. The useful signal is the Evidence@1 change.

## 5. Claim Boundaries

The final evidence supports the following claims:

- Query-guided unit selection defines a goal-dependent intermediate distribution.
- On CLUTRR, transition regularization improves reference next-unit selection and length extrapolation relative to the controlled answer-only unit-attention model.
- CLUTRR edge supervision and transition regularization have complementary effects on answer accuracy and transition alignment.
- On proposition tasks, evidence regularization improves reference-evidence selection; removing the query substantially weakens that selection on ProofWriter.
- Final-answer accuracy and intermediate-selection quality can diverge.

The final evidence does not support the following stronger claims:

- universal or statistically significant improvement;
- state-of-the-art performance against all large-language-model, graph, proof, or neural-symbolic systems;
- causal faithfulness or explainability of attention scores;
- uniform gains at every seen and unseen depth;
- lower compute cost than chain-of-thought methods;
- independent necessity of every aggregation or relation-conditioning component.

All comparisons use three seeds and report descriptive sample standard deviation. Transition@1 and Evidence@1 are directly optimized when their annotations are available. Transition@1 also credits one adapter-derived shortest path and may miss alternative valid derivations.

## 6. Reproducibility Artifacts

Tracked final artifacts:

- `scripts/run_final_paper_evidence_alignment.py`
- `scripts/aggregate_paper_evidence_alignment.py`
- `scripts/audit_final_paper_evidence_alignment.py`
- `docs/aggregated_results/PAPER_EVIDENCE_ALIGNMENT_20260710.json`
- `docs/aggregated_results/PAPER_EVIDENCE_ALIGNMENT_20260710.md`
- `docs/aggregated_results/PAPER_EVIDENCE_ALIGNMENT_AUDIT_20260710.json`
- `docs/aggregated_results/PAPER_EVIDENCE_ALIGNMENT_AUDIT_20260710.md`
- `docs/aggregated_results/PAPER_EVIDENCE_ALIGNMENT_MANIFEST_20260710.json`
- `docs/aggregated_results/PAPER_EVIDENCE_ALIGNMENT_STATUS_20260710.json`

Raw artifacts retained outside Git:

- Results: `/vepfs/trua_outputs/paper_evidence_alignment/main_20260710/results`
- Logs: `/vepfs/trua_outputs/paper_evidence_alignment/main_20260710/logs`
- Aggregate: `/vepfs/trua_outputs/paper_evidence_alignment/main_20260710/aggregated.json`
- Audit: `/vepfs/trua_outputs/paper_evidence_alignment/main_20260710/audit_summary.json`

Reproduction commands:

```bash
python scripts/run_final_paper_evidence_alignment.py \
  --run-root /vepfs/trua_outputs/paper_evidence_alignment/main_20260710 \
  --gpus 0,1,2,3 --poll-seconds 20

python scripts/aggregate_paper_evidence_alignment.py \
  /vepfs/trua_outputs/paper_evidence_alignment/main_20260710

python scripts/audit_final_paper_evidence_alignment.py \
  /vepfs/trua_outputs/paper_evidence_alignment/main_20260710
```

## 7. Superseded Material

Older files in `docs/aggregated_results`, `docs/figures`, historical branches, and `/vepfs/tsra_outputs` remain provenance records. They may use earlier names, test-selected checkpoints, different dataset variants, different backbones, or exploratory protocols. They are not sources for the current paper tables. The artifacts listed in Section 6 are the sole final evidence set.
