# TRUA-v2 Bottleneck Exploration Notes

This note records isolated exploratory results only. It is not part of the main experiment report.

## Hypothesis

The current TRUA-Prop implementation can improve trace-aligned attention without forcing the final answer to depend on the trace-supervised reasoning state. TRUA-v2 bottleneck tests whether final prediction should pass primarily through a reasoning head, while the full-text CLS head is reduced to a small residual.

## Implementation

Script:

- `experiments/trua_algorithm_explore/prop_bottleneck_trua.py`

Key outputs:

- `fused_accuracy`: final logits, `reason_logits + residual * text_logits`;
- `reason_accuracy`: reasoning head alone;
- `text_accuracy`: full-text bypass head alone;
- trace metrics: first-glimpse top-1, any-glimpse hit, and gold-trace attention mass.

## Small-Scale Setting

- train limit: 5,000
- test limit: 2,000
- epochs: 3
- seed: 0
- encoder: BERT base unless otherwise noted
- train-time trace supervision only; test-time input does not include gold trace/proof

## Results

### ProofWriter

| Variant | Backbone | Text residual | Bypass dropout | Depth-3 fused | Depth-3 reason | Depth-5 fused | Depth-5 reason | Depth-5 trace any |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| prior single evidence reader | BERT | n/a | n/a | 0.7650 | n/a | 0.7520 | n/a | 0.6259 |
| prior multi-hop reader | BERT | n/a | n/a | 0.7600 | n/a | 0.7390 | n/a | 0.6194 |
| TRUA-v2 soft bottleneck | BERT | 0.2 | 0.5 | 0.7690 | 0.7740 | 0.7855 | 0.7850 | 0.5557 |
| TRUA-v2 hard bottleneck | BERT | 0.0 | 1.0 | 0.7630 | 0.7630 | 0.7535 | 0.7535 | 0.5803 |
| TRUA-v2 soft bottleneck | DeBERTa | 0.2 | 0.5 | 0.7260 | 0.7260 | 0.7240 | 0.7240 | 0.2764 |

Observation:

- BERT soft bottleneck is the most promising result: ProofWriter depth-5 improves from `0.7520` to `0.7855`, and reason-only accuracy is essentially the same as fused accuracy.
- Hard bottleneck removes the benefit, suggesting that a small text residual helps stabilize answer prediction.
- DeBERTa soft bottleneck does not reproduce the BERT signal in this tiny setting; trace losses remain high and trace metrics are low.

### RuleTaker Raw QDep

| Variant | Text residual | Bypass dropout | Reason CE weight | Test fused | Test reason | Test text | Test trace any |
|---|---:|---:|---:|---:|---:|---:|---:|
| prior single evidence reader | n/a | n/a | n/a | 0.5780 | n/a | n/a | 0.7922 |
| prior multi-hop reader | n/a | n/a | n/a | 0.5920 | n/a | n/a | 0.8095 |
| TRUA-v2 soft bottleneck | 0.2 | 0.5 | 1.0 | 0.5615 | 0.5445 | 0.5090 | 0.8204 |
| TRUA-v2 hard bottleneck | 0.0 | 1.0 | 1.0 | 0.5365 | 0.5365 | 0.4790 | 0.8079 |
| TRUA-v2 mild bottleneck | 0.5 | 0.2 | 0.5 | 0.5805 | 0.5280 | 0.5795 | 0.8009 |

Observation:

- Strong bottlenecks hurt RuleTaker.
- Mild bottleneck recovers to roughly the prior single-reader level but does not exceed the prior multi-hop reader.
- In this small setup, RuleTaker still needs either more data/epochs or an ordered rule-transition module rather than a stronger generic evidence bottleneck.

## Interim Algorithm Judgment

Promising:

- Soft bottleneck is worth testing further on ProofWriter with larger training limits, more seeds, and possibly BERT/RoBERTa first.
- The useful configuration is not a hard oracle-like trace bottleneck, but a causal bottleneck with a small text residual.

Not yet promising:

- Hard bottleneck.
- Directly porting the same bottleneck strength to RuleTaker.
- DeBERTa with the current bottleneck hyperparameters in a 5k/3ep pilot.

Next recommended experiment:

1. ProofWriter BERT soft bottleneck, full train or 50k train, 10 epochs, seeds 0/1/42.
2. ProofWriter RoBERTa soft bottleneck, same setting if BERT remains positive.
3. RuleTaker should move toward ordered rule-state transition instead of just bottlenecking the sentence reader.
