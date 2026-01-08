import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math

# Use Agg backend for non-interactive plotting
plt.switch_backend('Agg')

# Import from the existing baseline script to ensure compatibility
try:
    from baseline_roberta_analysis import CLUTRRBaseline, clutrr_loader
except ImportError:
    print("Error: Could not import from baseline_roberta_analysis.py. Make sure it is in the same directory.")
    exit(1)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ==========================================
# Helpers
# ==========================================
def plot_metric_vs_hop(hops, values, correct, metric_name, filename):
    ensure_dir("diagnostic_plots")
    
    # Bucketing
    data = {} # hop -> {'c': [], 'w': []}
    
    for h, v, c in zip(hops, values, correct):
        if h not in data: data[h] = {'c': [], 'w': []}
        if c: data[h]['c'].append(v)
        else: data[h]['w'].append(v)
        
    sorted_hops = sorted(data.keys())
    avg_c = []
    avg_w = []
    
    for h in sorted_hops:
        c_vals = data[h]['c']
        w_vals = data[h]['w']
        avg_c.append(np.mean(c_vals) if c_vals else np.nan)
        avg_w.append(np.mean(w_vals) if w_vals else np.nan)
        
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_hops, avg_c, 'g-o', label='Correct')
    plt.plot(sorted_hops, avg_w, 'r--x', label='Wrong')
    plt.xlabel('Hops')
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs Complexity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"diagnostic_plots/{filename}")
    print(f"Saved diagnostic_plots/{filename}")
    plt.close()

# ==========================================
# Experiment 1: Control Pad Confusion (Refined Sink Analysis)
# ==========================================
def analyze_attention_sinks_refined(model, test_loader, device):
    print("Running Experiment 1: Refined Attention Sink Analysis (Excluding Pad)...")
    model.eval()
    
    # Define Sinks: <s>, </s>. 
    tokenizer = model.tokenizer
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    
    sink_ids = {bos_id, eos_id}
    
    results = {
        'hop': [],
        'sink_ratio': [],
        'correct': []
    }
    
    # To identify high-sink heads later
    # Store (sum_sink_ratio, count) for each (layer, head)
    num_layers = model.roberta.config.num_hidden_layers
    num_heads = model.roberta.config.num_attention_heads
    head_sink_accum = np.zeros((num_layers, num_heads))
    head_count = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Exp 1 Analysis"):
            (x_data, y_target, hops) = batch
            y_target = y_target.to(device)
            out = model(x_data)
            
            # Attentions: Tuple of 12 layers, each (B, H, S, S)
            last_attn = out["attentions"][-1] # (B, H, S, S)
            preds = torch.argmax(out["logits"], dim=1)
            correct_mask = (preds == y_target).cpu().numpy()
            input_ids = out["input_ids"] # (B, S)
            
            B, H, S, _ = last_attn.shape
            
            for i in range(B):
                ids = input_ids[i]
                is_pad = (ids == pad_id)
                is_sink = torch.tensor([uid.item() in sink_ids for uid in ids], device=device)
                
                # Analyze attention of [CLS] token (query_idx=0)
                cls_attn = last_attn[i, :, 0, :] # (H, S)
                
                total_mass = cls_attn[:, ~is_pad].sum(dim=1) # (H,)
                sink_mass = cls_attn[:, is_sink].sum(dim=1)   # (H,)
                
                ratio_per_head = sink_mass / (total_mass + 1e-9)
                avg_sink_ratio = ratio_per_head.mean().item()
                
                results['hop'].append(hops[i].item())
                results['sink_ratio'].append(avg_sink_ratio)
                results['correct'].append(correct_mask[i])
                
            # Accumulate sink stats for ALL layers/heads for Exp 4
            head_count += B
            for l_idx, layer_attn in enumerate(out["attentions"]):
                 for b_idx in range(B):
                     ids = input_ids[b_idx]
                     is_sink = torch.tensor([uid.item() in sink_ids for uid in ids], device=device)
                     
                     cls_attn = layer_attn[b_idx, :, 0, :]
                     sink_sum = cls_attn[:, is_sink].sum(dim=1) # (H,)
                     head_sink_accum[l_idx] += sink_sum.cpu().numpy()

    # Normalize head stats
    head_sink_avg = head_sink_accum / head_count
    
    # Identify Top Sink Heads (e.g., > 0.9 ratio)
    high_sink_heads = []
    print("\nHigh Sink Heads (Sink Ratio > 0.9 on CLS):")
    for l in range(num_layers):
        for h in range(num_heads):
            # Using 0.9 as threshold for "High Sink Head"
            if head_sink_avg[l, h] > 0.9:
                high_sink_heads.append((l, h))
                print(f"Layer {l} Head {h}: {head_sink_avg[l, h]:.4f}")

    # Plotting Sink Ratio vs Hop
    plot_metric_vs_hop(results['hop'], results['sink_ratio'], results['correct'], 
                       "Sink Ratio (No Pad)", "exp1_refined_sink.png")
    
    return high_sink_heads, results

# ==========================================
# Experiment 2: Entropy Normalization
# ==========================================
def analyze_entropy_normalized(model, test_loader, device):
    print("Running Experiment 2: Normalized Entropy Analysis...")
    model.eval()
    
    results = {
        'hop': [],
        'norm_entropy': [],
        'correct': []
    }
    
    pad_id = model.tokenizer.pad_token_id
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Exp 2 Analysis"):
            (x_data, y_target, hops) = batch
            y_target = y_target.to(device)
            out = model(x_data)
            
            # Use Last Layer
            attentions = out["attentions"][-1] # (B, H, S, S)
            input_ids = out["input_ids"]
            preds = torch.argmax(out["logits"], dim=1)
            correct_mask = (preds == y_target).cpu().numpy()
            
            # Analyze Entropy of SUBJECT attention
            queries = x_data[1] # list of (sub, obj)
            
            for i in range(len(queries)):
                ids = input_ids[i].tolist()
                valid_len = len([x for x in ids if x != pad_id])
                
                # Identify Subject Token Indices
                tokens = model.tokenizer.convert_ids_to_tokens(ids)
                sub_str = queries[i][0].replace(' ', '').lower()
                
                # Heuristic: Find tokens that match subject
                sub_indices = [ix for ix, t in enumerate(tokens) if sub_str in t.lower().replace('ġ', '')]
                
                if not sub_indices:
                    sub_idx = 0 # Fallback to CLS
                else:
                    sub_idx = sub_indices[0] # Take first token of subject
                    
                attn_row = attentions[i, :, sub_idx, :]
                
                attn_row = attn_row + 1e-12
                # Renormalize
                attn_row = attn_row / attn_row.sum(dim=-1, keepdim=True)
                
                entropy = -torch.sum(attn_row * torch.log(attn_row), dim=-1) # (H,)
                avg_entropy = entropy.mean().item()
                
                norm_entropy = avg_entropy / (math.log(valid_len) + 1e-9)
                
                results['hop'].append(hops[i].item())
                results['norm_entropy'].append(norm_entropy)
                results['correct'].append(correct_mask[i])

    plot_metric_vs_hop(results['hop'], results['norm_entropy'], results['correct'], 
                       "Normalized Entropy", "exp2_norm_entropy.png")
    
    return results

# ==========================================
# Experiment 3: Correct vs Incorrect Bucketing (Hop >= 6)
# ==========================================
def analyze_correct_vs_incorrect_deep(sink_results, entropy_results, min_hop=6):
    print(f"Running Experiment 3: Correct/Incorrect Bucketing (Hop >= {min_hop})...")
    
    hops_sink = sink_results['hop']
    
    metrics = {
        'Sink Ratio': sink_results['sink_ratio'],
        'Norm Entropy': entropy_results['norm_entropy']
    }
    
    data_points = []
    
    # Assume aligned lists
    for i, h in enumerate(hops_sink):
        if h < min_hop: continue
        is_corr = sink_results['correct'][i]
        label = "Correct" if is_corr else "Wrong"
        
        data_points.append({
            'Correctness': label,
            'Sink Ratio': metrics['Sink Ratio'][i],
            'Norm Entropy': metrics['Norm Entropy'][i]
        })
        
    if not data_points:
        print("No samples found with hop >= 6.")
        return

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, metric in enumerate(['Sink Ratio', 'Norm Entropy']):
        vals = [d[metric] for d in data_points]
        cats = [d['Correctness'] for d in data_points]
        
        sns.boxplot(x=cats, y=vals, ax=axes[idx], palette="Set2")
        axes[idx].set_title(f"{metric} (Hop >= {min_hop})")
        
    plt.tight_layout()
    file_path = "diagnostic_plots/exp3_correct_vs_wrong.png"
    ensure_dir("diagnostic_plots")
    plt.savefig(file_path)
    print(f"Saved {file_path}")


import types
import torch.nn as nn
try:
    from transformers.cache_utils import Cache, EncoderDecoderCache
except ImportError:
    # Fallback for older transformers or if simpler import needed
    Cache = object
    EncoderDecoderCache = object

# ==========================================
# Experiment 4: Intervention (Head Pruning)
# ==========================================
def custom_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        past_key_values=None,
        output_attentions: bool = False,
        cache_position: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        
        # NOTE: Simplified version of RobertaSelfAttention.forward
        # Assumes standard usage (no past_key_values usually for this analysis)
        
        batch_size, seq_length, _ = hidden_states.shape
        query_layer = self.query(hidden_states)
        
        # Handle recent transformers view/reshape
        query_layer = query_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        
        # Simplified key/value path (ignoring cross-attn/caching for CLUTRR analysis speed)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        key_layer = key_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value_layer = value_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        
        # Attention Scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        # Rel Pos Embeddings (if any)
        if getattr(self, "position_embedding_type", "absolute") in ["relative_key", "relative_key_query"]:
            # Omitted for brevity, assuming absolute for RoBERTa-base usually.
            # But standard RoBERTa is absolute. If using pre-trained roberta-base, it's absolute.
            pass
            
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # === INTERVENTION ===
        if hasattr(self, 'heads_to_mask_cls') and self.heads_to_mask_cls:
            # Mask Key=CLS (Index 0) for specific heads
            indices = self.heads_to_mask_cls
            # attention_scores shape: (B, H, Q, K)
            # Set [:, indices, :, 0] to -inf
            # Check device
            neg_inf = torch.tensor(float('-inf'), device=attention_scores.device)
            attention_scores[:, indices, :, 0] = neg_inf
        # ====================

        # Softmax
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_shape)

        return (context_layer, attention_probs) if output_attentions else (context_layer,)

def experiment_intervention_pruning(model, test_loader, device, prune_heads):
    print("Running Experiment 4: Intervention (CLS-Masking on High-Sink Heads)...")
    
    if not prune_heads:
        print("No high-sink heads (>0.9) to prune found. Skipping intervention.")
        return
        
    # Group heads by layer
    heads_map = {}
    for (l, h) in prune_heads:
        if l not in heads_map: heads_map[l] = []
        heads_map[l].append(h)
        
    print(f"Applying CLS-masking to {len(prune_heads)} heads across {len(heads_map)} layers.")
    
    # 1. Patch the model
    original_forwards = {}
    
    for l_idx, heads in heads_map.items():
        layer_module = model.roberta.encoder.layer[l_idx].attention.self
        
        # Save original
        original_forwards[l_idx] = layer_module.forward
        
        # Set attributes
        layer_module.heads_to_mask_cls = heads
        
        # Bind new forward
        layer_module.forward = types.MethodType(custom_forward, layer_module)
        
    # Evaluate
    model.eval()
    
    metrics = {'base': {'corr_all': 0, 'corr_long': 0, 'tot_all': 0, 'tot_long': 0},
               'masked': {'corr_all': 0, 'corr_long': 0, 'tot_all': 0, 'tot_long': 0}}
    
    # Run evaluation
    # To compare properly, we should run BASE without patch, and MASKED with patch.
    # But patching is stateful.
    # We will do: Run batch through BASE (unpatch/repatch? No, just run model once if we could toggle)
    # Easiest: Run full eval with Patch. We already have BASE numbers from previous runs or we can assume them.
    # But for exact comparison on same batch, we need to toggle.
    
    # Let's toggle per batch for safety.
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Intervention Eval"):
            (x_data, y_target, hops) = batch
            y_target = y_target.to(device)
            
            # Construct Inputs
            (contexts, queries, context_splits, _) = x_data
            batch_input_texts = []
            for i, (start, end) in enumerate(context_splits):
                story_text = ". ".join(contexts[start:end])
                query_text = f"{queries[i][0]} and {queries[i][1]}"
                batch_input_texts.append((story_text, query_text))
            
            inputs = model.tokenizer(
                batch_input_texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(device)
            
            # --- Per Batch Toggle ---
            
            # 1. Unpatch (Restore original)
            for l_idx in heads_map:
                model.roberta.encoder.layer[l_idx].attention.self.forward = original_forwards[l_idx]
            
            outputs_base = model.roberta(**inputs)
            preds_base = torch.argmax(model.classifier(outputs_base.last_hidden_state[:, 0, :]), dim=1)
            
            # 2. Patch (Apply custom)
            for l_idx, heads in heads_map.items():
                layer_module = model.roberta.encoder.layer[l_idx].attention.self
                layer_module.forward = types.MethodType(custom_forward, layer_module)
                
            outputs_masked = model.roberta(**inputs)
            preds_masked = torch.argmax(model.classifier(outputs_masked.last_hidden_state[:, 0, :]), dim=1)
            
            # Stats
            total = len(y_target)
            long_mask = (hops >= 6)
            
            metrics['base']['tot_all'] += total
            metrics['base']['corr_all'] += (preds_base == y_target).sum().item()
            metrics['masked']['tot_all'] += total
            metrics['masked']['corr_all'] += (preds_masked == y_target).sum().item()
            
            if long_mask.any():
                n_long = long_mask.sum().item()
                metrics['base']['tot_long'] += n_long
                metrics['base']['corr_long'] += (preds_base[long_mask] == y_target[long_mask]).sum().item()
                metrics['masked']['tot_long'] += n_long
                metrics['masked']['corr_long'] += (preds_masked[long_mask] == y_target[long_mask]).sum().item()

    # Restore originals finally
    for l_idx in heads_map:
        model.roberta.encoder.layer[l_idx].attention.self.forward = original_forwards[l_idx]

    # Report
    print(f"\nIntervention Results (CLS Masking):")
    for key in ['base', 'masked']:
        acc = metrics[key]['corr_all'] / metrics[key]['tot_all']
        acc_long = metrics[key]['corr_long'] / (metrics[key]['tot_long'] + 1e-9)
        print(f"[{key.upper()}] Global: {acc:.4f} | Long Hop(>=6): {acc_long:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    model = CLUTRRBaseline(device)
    path = "best_model_run0.pth"
    if not os.path.exists(path):
        print(f"Checkpoint {path} not found. Skipping.")
        return
        
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    
    # Load Data (Full Test)
    # Note: Use path relative to current script
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/clutrr")
    
    print(f"Loading data from {data_root}...")
    _, test_loader = clutrr_loader(root=data_root, dataset="data_089907f8", batch_size=32, training_data_percentage=100)
    
    # --- Run Experiments ---
    
    # Exp 1 & Identify Heads
    sink_heads, sink_results = analyze_attention_sinks_refined(model, test_loader, device)
    
    # Exp 2
    entropy_results = analyze_entropy_normalized(model, test_loader, device)
    
    # Exp 3
    analyze_correct_vs_incorrect_deep(sink_results, entropy_results, min_hop=6)
    
    # Exp 4
    experiment_intervention_pruning(model, test_loader, device, sink_heads)

if __name__ == "__main__":
    main()
