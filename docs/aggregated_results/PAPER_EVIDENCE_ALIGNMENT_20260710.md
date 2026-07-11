# Validation-Selected TRUA Evidence Alignment Results

All checkpoints are selected on a fixed training-set validation partition. The test set is evaluated once.

## CLUTRR

| Group | Seeds | Selected epoch | Overall | Short | Long >=6 | Transition@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clutrr_deberta_answer_only | 3 | 7.0000 +/- 1.0000 | 0.4229 +/- 0.0328 | 0.8042 +/- 0.0070 | 0.3177 +/- 0.0405 | 0.1968 +/- 0.0104 |
| clutrr_deberta_edge_only | 3 | 7.3333 +/- 2.3094 | 0.5576 +/- 0.0091 | 0.8485 +/- 0.0359 | 0.4236 +/- 0.0176 | 0.1737 +/- 0.0137 |
| clutrr_deberta_no_aggregation | 3 | 6.0000 +/- 3.0000 | 0.5905 +/- 0.0207 | 0.8275 +/- 0.0040 | 0.4721 +/- 0.0304 | 0.5052 +/- 0.0287 |
| clutrr_deberta_no_goal | 3 | 6.3333 +/- 2.5166 | 0.6117 +/- 0.0193 | 0.8135 +/- 0.0081 | 0.5018 +/- 0.0268 | 0.5214 +/- 0.0056 |
| clutrr_deberta_no_relation | 3 | 6.0000 +/- 1.0000 | 0.5960 +/- 0.0361 | 0.8322 +/- 0.0252 | 0.4710 +/- 0.0478 | 0.4946 +/- 0.0103 |
| clutrr_deberta_no_step | 3 | 8.3333 +/- 1.1547 | 0.6108 +/- 0.0402 | 0.8228 +/- 0.0323 | 0.5076 +/- 0.0495 | 0.1427 +/- 0.0056 |
| clutrr_deberta_transition_only | 3 | 9.3333 +/- 0.5774 | 0.5218 +/- 0.0334 | 0.8228 +/- 0.0225 | 0.4100 +/- 0.0316 | 0.5114 +/- 0.0121 |
| clutrr_deberta_trua | 3 | 8.3333 +/- 0.5774 | 0.6012 +/- 0.0245 | 0.8135 +/- 0.0146 | 0.4799 +/- 0.0268 | 0.5076 +/- 0.0049 |
| clutrr_deberta_vanilla | 3 | 9.3333 +/- 0.5774 | 0.2830 +/- 0.0088 | 0.9557 +/- 0.0107 | 0.1544 +/- 0.0248 | -- |
| clutrr_deberta_with_consistency | 3 | 8.3333 +/- 2.0817 | 0.5797 +/- 0.0202 | 0.7949 +/- 0.0476 | 0.4497 +/- 0.0235 | 0.5189 +/- 0.0182 |

## Proposition Tasks

| Group | Split | Seeds | Selected epoch | Accuracy | Evidence@1 |
| --- | --- | ---: | ---: | ---: | ---: |
| prontoqa_bert_no_evidence | ood | 3 | 1.6667 +/- 1.1547 | 1.0000 +/- 0.0000 | 0.2867 +/- 0.0982 |
| prontoqa_bert_trua | ood | 3 | 2.6667 +/- 2.0817 | 1.0000 +/- 0.0000 | 0.4644 +/- 0.0222 |
| proofwriter_bert_no_evidence | depth-3 | 3 | 8.3333 +/- 0.5774 | 0.9427 +/- 0.0033 | 0.1936 +/- 0.0145 |
| proofwriter_bert_no_evidence | depth-5 | 3 | 8.3333 +/- 0.5774 | 0.9091 +/- 0.0020 | 0.1528 +/- 0.0351 |
| proofwriter_bert_no_goal | depth-3 | 3 | 9.3333 +/- 0.5774 | 0.9423 +/- 0.0067 | 0.3831 +/- 0.0091 |
| proofwriter_bert_no_goal | depth-5 | 3 | 9.3333 +/- 0.5774 | 0.9093 +/- 0.0037 | 0.4304 +/- 0.0119 |
| proofwriter_bert_trua | depth-3 | 3 | 9.3333 +/- 1.1547 | 0.9359 +/- 0.0125 | 0.8698 +/- 0.0129 |
| proofwriter_bert_trua | depth-5 | 3 | 9.3333 +/- 1.1547 | 0.8997 +/- 0.0198 | 0.8814 +/- 0.0078 |
| ruletaker_raw_deberta_no_evidence | test | 3 | 4.3333 +/- 3.0551 | 0.5575 +/- 0.0553 | 0.2907 +/- 0.0138 |
| ruletaker_raw_deberta_trua | test | 3 | 9.6667 +/- 0.5774 | 0.8303 +/- 0.1292 | 0.8947 +/- 0.0049 |
