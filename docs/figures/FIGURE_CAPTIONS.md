# TRUA Figure Captions

These figure assets were generated from the current TRUA repository state on 2026-05-29. Use the PDF/SVG versions for paper layout and the PNG versions for quick preview.

## Figure 2: CLUTRR Trace-Aligned Attention Heatmap

File: `fig2_clutrr_trace_attention_heatmap.{png,pdf,svg}`

Representative CLUTRR `data_089907f8` 7-hop sample showing step-wise candidate-edge selection. The gold path is a real sample path. The heatmap is a trace-aligned reasoning-attention visualization, not a raw Transformer self-attention head.

## Figure 3: ProofWriter Proof-Step Attention Graph

File: `fig3_proof_step_attention_graph.{png,pdf,svg}`

Representative ProofWriter depth-5 proof item. The graph structure comes from the official ProofWriter proof tree; the inset trace@1 values are the current 3-seed aggregated BERT depth-5 results: baseline 0.146 vs TRUA 0.784.

## Figure 4: Trace Faithfulness and Depth Curves

File: `fig4_trace_faithfulness_depth_curves.{png,pdf,svg}`

Quantitative plot from current aggregated 3-seed artifacts: ProofWriter depth-5 trace@1, PrOntoQA-OOD trace@1, ProofWriter answer accuracy versus trace alignment, and CLUTRR `data_089907f8` true vanilla encoder versus TRUA long-hop accuracy. The CLUTRR panel uses the plain encoder classifier audit, not the TRUA label-only ablation.

## Figure 5: Long-Hop Failure Case

File: `fig5_long_hop_failure_case.{png,pdf,svg}`

Qualitative schematic on a real CLUTRR `data_089907f8` 10-hop example. It contrasts endpoint/local shortcut attention with TRUA-style transition-regularized path coverage. Use wording such as "qualitative visualization" or "schematic failure mode" unless raw per-example attention dumps are later added.

## External Baseline Overview

File: `fig_external_baseline_overview.{png,pdf,svg}`

Four-panel summary of completed external baseline comparisons on CLUTRR, ProofWriter, RuleTaker, and PrOntoQA-OOD. Colors distinguish same-input/adapted baselines, TRUA, external methods, and official/reference systems. Some panels combine different task interfaces or metrics, so use the panel footnotes when citing the figure.

## CLUTRR External Comparison

File: `fig_clutrr_external_comparison.{png,pdf,svg}`

Long-hop accuracy on CLUTRR `data_089907f8`. The figure contrasts raw-text adapted attention methods, a true vanilla DeBERTa-v3 encoder classifier audit, TRUA, RAT, and EdgeTransformer. RAT and EdgeTransformer use structured graph-style inputs and should be described as reference/structured baselines.

## ProofWriter External Comparison

File: `fig_proofwriter_external_comparison.{png,pdf,svg}`

Depth-5 answer accuracy comparison using completed ProofWriter results. DAT and Abstractor/RCA are same-input DeBERTa adapters; LoGiPT and AAI are LM/LLM references; FaiRR is its official end-to-end proof pipeline.

## RuleTaker External Comparison

File: `fig_ruletaker_external_comparison.{png,pdf,svg}`

RuleTaker-family method comparison. DAT and Abstractor/RCA use the raw-QDep same-input setting; NLProofS, GFaiR, and IBR are official/reference proof-reasoning systems; TRUA uses the GFaiR split. Treat the figure as a method-reference overview rather than a single controlled leaderboard.

## PrOntoQA-OOD External Comparison

File: `fig_prontoqa_external_comparison.{png,pdf,svg}`

PrOntoQA-OOD comparison showing trace@1 for same-input methods and final-statement accuracy for latent/reference methods. The processed discriminative label task is saturated, so trace@1 is the more informative TRUA-side diagnostic.
