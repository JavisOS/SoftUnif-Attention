import os
import csv
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import re

# Use Agg backend
plt.switch_backend('Agg')

# Reuse baseline model
try:
    from baseline_roberta_analysis import CLUTRRBaseline, relation_id_map, id_to_relation
except ImportError:
    print("Error imports. Run from root.")
    exit(1)

# Composition Rules Config (Same as before)
COMPO_RULES = {
    ('father', 'father'): 'grandfather',
    ('mother', 'mother'): 'grandmother',
    ('father', 'mother'): 'grandmother',
    ('mother', 'father'): 'grandfather',
    ('sister', 'father'): 'aunt',
    ('sister', 'mother'): 'aunt',
    ('brother', 'father'): 'uncle',
    ('brother', 'mother'): 'uncle',
    ('husband', 'daughter'): 'daughter', 
    ('wife', 'son'): 'son',
}

class PathDataset(torch.utils.data.Dataset):
    def __init__(self, root, dataset, split, data_limit=1000):
        self.dataset_dir = os.path.join(root, f"{dataset}/")
        self.raw_data = [] 
        if os.path.exists(self.dataset_dir):
            file_names = [os.path.join(self.dataset_dir, d) for d in os.listdir(self.dataset_dir) if f"_{split}.csv" in d]
            count = 0
            for f in file_names:
                rows = list(csv.reader(open(f)))[1:]
                for r in rows:
                    if count >= data_limit: break
                    self.raw_data.append(r)
                    count += 1
                if count >= data_limit: break

    def __len__(self):
        return len(self.raw_data)

    def extract_names_ordered(self, story_str):
        names = re.findall(r"\[(.*?)\]", story_str)
        seen = set()
        unique_names = []
        for n in names:
            if n not in seen:
                unique_names.append(n)
                seen.add(n)
        return unique_names

    def __getitem__(self, i):
        row = self.raw_data[i]
        story = row[2]
        query = eval(row[3])
        try:
            edges = eval(row[11])
            types = eval(row[12])
        except:
            return None 

        unique_names = self.extract_names_ordered(story)
        name_to_id = {n: idx for idx, n in enumerate(unique_names)}
        
        sub_name, obj_name = query
        if sub_name not in name_to_id or obj_name not in name_to_id:
            return None
            
        start_node = name_to_id[sub_name]
        end_node = name_to_id[obj_name]
        
        G = nx.Graph()
        for (u, v), t in zip(edges, types):
            G.add_edge(u, v, relation=t)
            
        try:
            path_nodes = nx.shortest_path(G, source=start_node, target=end_node)
        except nx.NetworkXNoPath:
            return None
            
        path_names = [unique_names[idx] for idx in path_nodes]
        context_list = [s.strip().lower() for s in story.split(".") if s.strip() != ""]
        
        return {
            "story": story,
            "context_list": context_list,
            "query": query,
            "path_names": path_names,
            "all_names": unique_names,
            "hops": len(path_nodes) - 1
        }

def collate_path(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return batch

def batch_list(lst, sz):
    for i in range(0, len(lst), sz):
        yield lst[i:i+sz]

def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)

# ==========================================
# Experiment 1: Layer-wise Next-Hop Entity Probe
# ==========================================
def run_exp1_next_hop(model, train_samples, test_samples, device):
    print("Running Exp 1: Layer-wise Next-Hop Entity Probe (Span-based)...")
    
    # We probe 3 layers: Early, Middle, Late
    # RoBERTa base has 12 layers. 0-11.
    layers_to_probe = [2, 6, 11]
    
    # Feature Extraction
    # We need: For each sample -> For each Layer -> Map {name: embedding}
    
    def extract_features(samples):
        # Result: list of dicts. Each dict: {layer_idx: {name: emb}}
        all_feats = []
        
        for chunk in tqdm(batch_list(samples, 16), desc="Extracting Exp1 Feats"):
            batch_texts = []
            batch_raw_stmts = []
            for s in chunk:
                 st = ". ".join(s['context_list'])
                 q = f"{s['query'][0]} and {s['query'][1]}"
                 batch_texts.append((st, q))
                 batch_raw_stmts.append(st)
                 
            inputs = model.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                return_offsets_mapping=True # New: Get character offsets
            ).to(device)
            
            with torch.no_grad():
                out = model.roberta(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True
                    # Do not pass return_offsets_mapping to model forward
                )
            
            # Access offsets
            offset_mappings = inputs.offset_mapping.cpu().numpy()

            for b_i in range(len(chunk)):
                s = chunk[b_i]
                st = batch_raw_stmts[b_i]
                sample_feats = {}
                
                # 1. Identify char spans for all names in story `st`
                char_intervals = {} # name -> list of (start, end)
                for name in s['all_names']:
                    target = f"[{name.lower()}]"
                    start_search = 0
                    intervals = []
                    while True:
                        idx = st.find(target, start_search)
                        if idx == -1: break
                        intervals.append((idx, idx + len(target)))
                        start_search = idx + 1
                    char_intervals[name] = intervals
                
                # 2. Map tokens to names via offsets
                name_token_indices = {} # name -> list of token indices
                seq_ids = inputs.sequence_ids(b_i) # Identifies segment (0=story, 1=query)
                offsets = offset_mappings[b_i]
                
                for t_idx, (start_char, end_char) in enumerate(offsets):
                    # Filter for Story segment (seq_id=0) and ignore special tokens
                    if seq_ids[t_idx] != 0: continue
                    if start_char == end_char: continue # Skip zero-width tokens if any
                    
                    # Check overlap with any name interval
                    for name, intervals in char_intervals.items():
                        for (i_start, i_end) in intervals:
                            # Strict containment: token is inside the bracketed name
                            if start_char >= i_start and end_char <= i_end:
                                if name not in name_token_indices:
                                    name_token_indices[name] = []
                                name_token_indices[name].append(t_idx)

                for l_idx in layers_to_probe:
                    # Reminder: hidden_states[0] = embeddings, [1] = layer 1 output...
                    h_layer = out.hidden_states[l_idx + 1][b_i] # (S, H)
                    
                    layer_name_map = {}
                    for name, t_indices in name_token_indices.items():
                        # Mean pool over all tokens of this entity (all occurrences)
                        unique_indices = list(set(t_indices))
                        if not unique_indices: continue
                        
                        emb = h_layer[unique_indices, :].mean(dim=0).cpu()
                        layer_name_map[name] = emb
                    
                    sample_feats[l_idx] = layer_name_map
                
                all_feats.append(sample_feats)
        return all_feats

    train_feats = extract_features(train_samples)
    test_feats = extract_features(test_samples)
    
    # Train Probes (One per Layer)
    probes = {} # layer -> Linear
    layer_accs = {l: {} for l in layers_to_probe} # layer -> {hop: acc}
    
    for l in layers_to_probe:
        print(f"Training Probe for Layer {l}...")
        probe = nn.Linear(768, 768, bias=False).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        
        # Train Loop
        probe.train()
        for epoch in range(5):
            total_loss = 0
            count = 0
            for i, feat_dict in enumerate(train_feats):
                # Sample logic
                s = train_samples[i]
                path = s['path_names'] # e0, e1, e2 ...
                layer_map = feat_dict[l]
                
                # Generate pairs (current, next)
                # If e_k is not in map (parsing fail), skip
                valid_pairs = []
                for k in range(len(path) - 1):
                    curr_name = path[k]
                    next_name = path[k+1]
                    if curr_name in layer_map and next_name in layer_map:
                        valid_pairs.append((layer_map[curr_name], layer_map[next_name]))
                
                if not valid_pairs: continue
                
                # Batch processing per sample? Or accumulate?
                # Do per sample for simplicity
                
                # Candidates (all names in map)
                cands = list(layer_map.keys())
                cand_embs = torch.stack([layer_map[c] for c in cands]).to(device)
                
                curr_embs = torch.stack([vp[0] for vp in valid_pairs]).to(device)
                
                # Predict
                preds = probe(curr_embs) # (Hops, 768)
                
                # Verify against ALL candidates
                # Score matrix: (Hops, NumCands)
                scores = torch.matmul(preds, cand_embs.T)
                
                # Targets
                target_indices = []
                for k in range(len(path)-1):
                    if path[k] in layer_map and path[k+1] in layer_map:
                        t_name = path[k+1]
                        target_indices.append(cands.index(t_name))
                
                targets_t = torch.tensor(target_indices).to(device)
                
                loss = nn.functional.cross_entropy(scores, targets_t)
                loss.backward()
                total_loss += loss.item()
                count += 1
            
            opt.step()
            opt.zero_grad()
            # print(f"  Ep {epoch} Loss: {total_loss/count:.4f}")
        
        probes[l] = probe
    
    # Evaluate
    print("Evaluating Next-Hop Probes...")
    from collections import defaultdict
    results = {l: defaultdict(list) for l in layers_to_probe}
    
    # Sanity Counters
    # Tracks how many links (curr->next) we ATTEMPTED to probe vs how many VALID (both found)
    probe_stats = {h: {'total': 0, 'valid': 0} for h in range(1, 20)} # Increase range for safety
    
    for l in layers_to_probe:
        probe = probes[l]
        probe.eval()
        
        with torch.no_grad():
            for i, feat_dict in enumerate(test_feats):
                s = test_samples[i]
                path = s['path_names']
                layer_map = feat_dict[l]
                
                cands = list(layer_map.keys())
                if not cands: continue
                
                for k in range(len(path) - 1):
                    hop_idx = k + 1 # 1-based
                    curr_name = path[k]
                    next_name = path[k+1]
                    
                    # Track Stats (only on first layer loop to avoid double counting)
                    if l == layers_to_probe[0]:
                        probe_stats[hop_idx]['total'] += 1
                        if curr_name in layer_map and next_name in layer_map:
                            probe_stats[hop_idx]['valid'] += 1

                    # Only evaluate if spans exist (DROP otherwise)
                    if curr_name in layer_map and next_name in layer_map:
                        curr_emb = layer_map[curr_name].to(device)
                        cand_embs = torch.stack([layer_map[c] for c in cands]).to(device)
                        
                        proj = probe(curr_emb)
                        scores = torch.matmul(cand_embs, proj)
                        
                        # Rank
                        best_idx = torch.argmax(scores).item()
                        is_correct = (cands[best_idx] == next_name)
                        results[l][hop_idx].append(1 if is_correct else 0)

    # Sanity Report
    print(f"\n[Probe Sanity Report]")
    print(f"Policy: If spans missing -> DROP sample (No Fallback)")
    print("Hop | Total Samples | Valid Samples (Both Spans Found) | Valid %")
    print("----|---------------|----------------------------------|--------")
    for h in range(1, 10):
        tot = probe_stats[h]['total']
        val = probe_stats[h]['valid']
        ratio = (val/tot*100) if tot > 0 else 0
        print(f"{h:3d} | {tot:13d} | {val:32d} | {ratio:6.2f}%")

    # Plotting
    ensure_dir("diagnostic_plots")
    
    print("\n[Probe Accuracy Results]")
    print(f"{'Hop':<4} | {'Layer 2':<10} | {'Layer 6':<10} | {'Layer 11':<10}")
    print("-" * 40)
    
    layer_accs_for_print = {h: {} for h in range(1, 10)}
    
    plt.figure(figsize=(10, 6))
    for l in layers_to_probe:
        hops = sorted(results[l].keys())
        accs = []
        for h in hops:
            if len(results[l][h]) < 5: # Filter low sample size
                accs.append(np.nan)
                layer_accs_for_print[h][l] = "N/A"
            else:
                val = np.mean(results[l][h])
                accs.append(val)
                layer_accs_for_print[h][l] = f"{val:.4f}"
                
        # Filter NaNs
        valid_points = [(h, a) for h,a in zip(hops, accs) if not np.isnan(a)]
        if valid_points:
            plt.plot([p[0] for p in valid_points], [p[1] for p in valid_points], '-o', label=f"Layer {l}")
            
    for h in range(1, 8): # Print up to hop 7
        l2 = layer_accs_for_print[h].get(2, "N/A")
        l6 = layer_accs_for_print[h].get(6, "N/A")
        l11 = layer_accs_for_print[h].get(11, "N/A")
        print(f"{h:<4} | {l2:<10} | {l6:<10} | {l11:<10}")

    plt.title("Exp 1: Next-Hop Prediction Acc (Span-based)")
    plt.xlabel("Hop Index")
    plt.ylabel("Accuracy (Recall@1)")
    plt.legend()
    plt.grid(True)
    plt.savefig("diagnostic_plots/exp1_nexthop_layerwise.png")
    print("Saved exp1_nexthop_layerwise.png")


# ==========================================
# Experiment 3: Clean Counterfactual Swap
# ==========================================
def run_exp3_robustness(model, test_samples, device):
    print("Running Exp 3: Counterfactual Swaps (Bijective Renaming)...")
    
    swapped_correct = 0
    total = 0
    
    # Bijective Replacement Map
    def apply_bijective_map(text, mapping):
        # We need to replace all Occurrences of keys in mapping with values
        # simultaneously or safely.
        # Safe way: Regex sub with callback.
        
        # Build regex for all names
        # Sort by length descending to match longest first
        sorted_names = sorted(mapping.keys(), key=len, reverse=True)
        escaped_names = [re.escape(n) for n in sorted_names]
        pattern = re.compile(r'\b(' + '|'.join(escaped_names) + r')\b')
        
        def replacement(match):
            return mapping[match.group(0)]
            
        return pattern.sub(replacement, text)

    for s in tqdm(test_samples):
        st_clean = ". ".join(s['context_list'])
        sub, obj = s['query']
        
        # Base Prediction
        inp1 = model.tokenizer([(st_clean, f"{sub} and {obj}")], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            pred1 = torch.argmax(model.classifier(model.roberta(**inp1).last_hidden_state[:, 0, :]), dim=1).item()
        
        # Generate Permutation
        names = s['all_names'] # Unique names in story
        if len(names) < 2: 
            # trivial, skip swap
            total += 1
            swapped_correct += 1 # Consistent trivially
            continue
            
        # Shuffle names to create mapping
        shuffled = list(names)
        random.shuffle(shuffled)
        mapping = {old: new for old, new in zip(names, shuffled)}
        
        # Apply to Story
        st_swapped = apply_bijective_map(st_clean, mapping)
        
        # Apply to Query
        # sub -> mapping[sub]
        # obj -> mapping[obj]
        if sub not in mapping or obj not in mapping:
            # Should not happen if all_names is correct
            continue
            
        new_sub = mapping[sub]
        new_obj = mapping[obj]
        
        q_swapped = f"{new_sub} and {new_obj}"
        
        inp2 = model.tokenizer([(st_swapped, q_swapped)], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            pred2 = torch.argmax(model.classifier(model.roberta(**inp2).last_hidden_state[:, 0, :]), dim=1).item()
            
        if pred1 == pred2:
            swapped_correct += 1
        total += 1
        
    consistency = swapped_correct / total if total > 0 else 0
    print(f"Exp 3 Consistency (Bijective): {consistency:.4f}")
    with open("diagnostic_plots/exp3_consistency_bijective.txt", "w") as f:
        f.write(f"Consistency: {consistency:.4f}")

# ==========================================
# Main Runner
# ==========================================
def run_diagnostics_v2(model, dataset_root):
    # Load Data
    print("Loading Path Dataset (High Limit)...")
    # Increased limit to 2000 to catch long hops
    ds = PathDataset(dataset_root, "data_089907f8", "test", data_limit=2000)
    samples = collate_path([ds[i] for i in range(len(ds))])
    if not samples:
        print("No samples found.")
        return

    random.shuffle(samples)
    train_sz = int(0.8 * len(samples))
    train_samples = samples[:train_sz]
    test_samples = samples[train_sz:]
    
    print(f"Total Samples: {len(samples)} (Train: {len(train_samples)}, Test: {len(test_samples)})")
    
    # Run Exp 1
    run_exp1_next_hop(model, train_samples, test_samples, model.device)
    
    # Run Exp 3 (Robustness)
    # Use subset for speed if needed, but 2000 is fine for inference
    run_exp3_robustness(model, samples, model.device)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    model = CLUTRRBaseline(device)
    path = "best_model_run0.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/clutrr")
    run_diagnostics_v2(model, data_root)
