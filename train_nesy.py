
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import wandb
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizerFast, RobertaModel

# Path fixes
sys.path.append(os.getcwd())

# Imports
try:
    from baseline_roberta_analysis import CLUTRRDataset, relation_id_map, id_to_relation, set_seed, evaluate
except ImportError:
    # Add root to sys.path
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(root_path)
    from baseline_roberta_analysis import CLUTRRDataset, relation_id_map, id_to_relation, set_seed, evaluate


from clutrr.nesy_utils import parse_graph_and_path, augment_bijective_swap
from clutrr.entity_alignment import align_entity_spans_to_tokens

# ==========================================
# 1. NeSy Dataset Wrapper
# ==========================================
class NeSyCLUTRRDataset(CLUTRRDataset):
    def __init__(self, root, dataset, split, data_percentage=100, tokenizer=None, augment=False):
        super().__init__(root, dataset, split, data_percentage)
        self.tokenizer = tokenizer
        self.augment = augment
        
    def __getitem__(self, i):
        # Baseline returns: ((context_sents, query), relation_id, hops, story_str)
        # We need more: entity spans and ground truth path
        
        row = self.data[i]
        
        # 1. Parse Graph Logic
        graph_info = parse_graph_and_path(row)
        if graph_info is None:
            # Fallback for broken samples
            return None
            
        story_str = row[2]
        query = eval(row[3])
        target_relation = row[5] # label string
        all_names = graph_info['all_names']
        
        # 2a. Entity Alignment (Original)
        alignments = align_entity_spans_to_tokens(story_str, all_names, self.tokenizer)
        
        # 3a. Construct Feature Dict (Original)
        node_spans = [] 
        valid_sample = True
        for name in all_names:
            res = alignments.get(name)
            if res and res['token_span']:
                node_spans.append(res['token_span'])
            else:
                node_spans.append(None)
                if graph_info['path_node_indices'].count(all_names.index(name)) > 0:
                    valid_sample = False
        
        if not valid_sample:
            return None
            
        # Target ID
        if target_relation in relation_id_map:
            target_id = relation_id_map[target_relation]
        else:
            target_id = relation_id_map['nothing']
        
        item = {
            'story': story_str,
            'query': query,
            'target_id': target_id,
            'path_indices': graph_info['path_node_indices'], 
            'node_spans': node_spans,
            'num_nodes': len(all_names),
            'hops': len(graph_info['path_node_indices']) - 1
        }
        
        # 4. Augmentation (Consistency Training)
        if self.augment:
            aug_story, aug_query = augment_bijective_swap(story_str, query, all_names)
            item['aug_story'] = aug_story
            item['aug_query'] = aug_query
            
        return item

def collate_nesy(batch):
    # Old function unused
    pass

class NeSyCollator:
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        
    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch: return None
        
        stories = [b['story'] for b in batch]
        queries = [f"{b['query'][0]} and {b['query'][1]}" for b in batch]
        targets = torch.tensor([b['target_id'] for b in batch], dtype=torch.long).to(self.device)
        hops = torch.tensor([b['hops'] for b in batch], dtype=torch.long).to(self.device)
        
        # Standard Encoding
        enc = self.tokenizer(
            stories,
            queries, 
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        ).to(self.device)
        
        # Augmented Encoding (if present)
        enc_aug = None
        if 'aug_story' in batch[0]:
            aug_stories = [b['aug_story'] for b in batch]
            aug_queries = [f"{b['aug_query'][0]} and {b['aug_query'][1]}" for b in batch]
            enc_aug = self.tokenizer(
                aug_stories,
                aug_queries,
                padding=True,
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            ).to(self.device)
        
        # Prepare Aux Labels (Same as before)
        max_path_len = max([len(b['path_indices']) for b in batch])
        path_node_ids = torch.full((len(batch), max_path_len), -1, dtype=torch.long).to(self.device)
        
        max_entities = max([b['num_nodes'] for b in batch])
        entity_spans = torch.full((len(batch), max_entities, 2), -1, dtype=torch.long).to(self.device)
        
        for i, b in enumerate(batch):
            p_len = len(b['path_indices'])
            path_node_ids[i, :p_len] = torch.tensor(b['path_indices'], dtype=torch.long)
            
            spans = b['node_spans'] 
            for j, s in enumerate(spans):
                if s is not None:
                    entity_spans[i, j, 0] = s[0]
                    entity_spans[i, j, 1] = s[1]
        
        return {
            'input_ids': enc.input_ids,
            'attention_mask': enc.attention_mask, 
            'labels': targets,
            'hops': hops,
            'path_node_ids': path_node_ids, 
            'entity_spans': entity_spans,
            'raw_batch': batch,
            'aug_input_ids': enc_aug.input_ids if enc_aug else None,
            'aug_attention_mask': enc_aug.attention_mask if enc_aug else None
        }

# ==========================================
# 2. NeSy Transformer Model
# ==========================================
class NeSyRoBERTa(nn.Module):
    def __init__(self, device, tokenizer):
        super().__init__()
        self.device = device
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.tokenizer = tokenizer # For debugging/resizing if needed
        self.hidden_size = self.roberta.config.hidden_size
        
        # Heads
        self.classifier = nn.Linear(self.hidden_size, 21) # 21 relations
        
        # Next-Hop Prediction
        # We use a query-key mechanism.
        # Query: Current Node Embedding -> Transform -> Match with Candidate Embeddings
        self.hop_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Intermediate Relation Prediction (Optional)
        # Concatenate (E_i, E_i+1) -> Rel
        self.rel_proj = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 21) # Same relation space
        )
        
    def get_entity_embeddings(self, last_hidden_state, entity_spans):
        # last_hidden_state: (B, Seq, H)
        # entity_spans: (B, N_ent, 2) - -1 pad
        
        B, N_ent, _ = entity_spans.shape
        # Output: (B, N_ent, H)
        
        # Mask for valid entities
        valid_mask = (entity_spans[:, :, 0] != -1) # (B, N_ent)
        
        # We can implement mean pooling nicely
        # But loop is easier for variable lengths and small N_ent (~10)
        
        emb_list = []
        for i in range(B):
            sample_embs = []
            hidden = last_hidden_state[i] # (Seq, H)
            for j in range(N_ent):
                if valid_mask[i, j]:
                    start, end = entity_spans[i, j]
                    # Safe clamp (though alignment should be safe)
                    start = min(start, hidden.size(0)-1)
                    end = min(end, hidden.size(0))
                    if start >= end: end = start + 1
                    
                    pool = hidden[start:end].mean(dim=0)
                    sample_embs.append(pool)
                else:
                    sample_embs.append(torch.zeros(self.hidden_size).to(self.device))
            emb_list.append(torch.stack(sample_embs))
            
        return torch.stack(emb_list) # (B, N_ent, H)

    def forward(self, batch_data, lambda1=1.0, lambda_cons=0.0):
        # Unpack
        input_ids = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        labels = batch_data['labels']
        entity_spans = batch_data['entity_spans']
        path_node_ids = batch_data['path_node_ids']
        
        # 1. RoBERTa Forward (Original)
        out_orig = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits_orig = self.classifier(out_orig.last_hidden_state[:, 0, :])
        main_loss = nn.functional.cross_entropy(logits_orig, labels)
        
        # 2. Consistency Loss (Scheme B)
        cons_loss = torch.tensor(0.0).to(self.device)
        if batch_data.get('aug_input_ids') is not None:
            # Forward Augmented
            out_aug = self.roberta(
                input_ids=batch_data['aug_input_ids'], 
                attention_mask=batch_data['aug_attention_mask']
            )
            logits_aug = self.classifier(out_aug.last_hidden_state[:, 0, :])
            
            # Loss: KL(P_orig || P_aug) + KL(P_aug || P_orig) (Symmetric KL / JS)
            # Or simplified: MSE on logits / CrossEntropy to same label?
            # User suggested "JS/KL or even CE".
            # CE is strongest: enforce Aug to match Ground Truth
            loss_aug_ce = nn.functional.cross_entropy(logits_aug, labels)
            
            # KL Consistency (Soft target)
            p_clean = nn.functional.softmax(logits_orig, dim=1)
            p_aug = nn.functional.log_softmax(logits_aug, dim=1)
            kl_loss = nn.functional.kl_div(p_aug, p_clean, reduction='batchmean')
            
            cons_loss = kl_loss # + loss_aug_ce? 
            # If we just add CE for aug, it acts as data augmentation.
            # If we add KL, it acts as consistency regularization.
            # Let's do both implicitly by weighting.
            
            main_loss = 0.5 * main_loss + 0.5 * loss_aug_ce
            # And consistency as extra term?
            cons_loss = kl_loss
        
        # 3. Auxiliary Losses (Nexthop) - On Original only
        sequence_output = out_orig.last_hidden_state
        entity_embs = self.get_entity_embeddings(sequence_output, entity_spans) 
        
        loss_nexthop = torch.tensor(0.0).to(self.device)
        total_hops = 0
        batch_nexthop_loss = 0.0
        
        for i in range(len(input_ids)):
            path = path_node_ids[i]
            valid_path = path[path != -1]
            if len(valid_path) < 2: continue
            
            sample_ent_embs = entity_embs[i]
            input_indices = valid_path[:-1]
            target_indices = valid_path[1:]
            
            curr_embs = sample_ent_embs[input_indices]
            queries = self.hop_proj(curr_embs)
            scores = torch.matmul(queries, sample_ent_embs.transpose(0, 1))
            
            step_loss = nn.functional.cross_entropy(scores, target_indices)
            batch_nexthop_loss += step_loss
            total_hops += 1
            
        if total_hops > 0:
            loss_nexthop = batch_nexthop_loss / total_hops
            
        total_loss = main_loss + lambda1 * loss_nexthop + lambda_cons * cons_loss
        
        return {
            'loss': total_loss,
            'losses': {
                'main': main_loss.item(), 
                'nexthop': loss_nexthop.item(),
                'cons': cons_loss.item()
            },
            'logits': logits_orig
        }

# ==========================================
# 3. Training Loop
# ==========================================
def run_training():
    set_seed(43)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load Tokenizer & Dataset
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    
    root = "data/clutrr"
    dset = "data_089907f8"
    
    print("Loading Data (Augmentation Enabled)...")
    train_ds = NeSyCLUTRRDataset(root, dset, "train", 100, tokenizer=tokenizer, augment=True)
    # Filter Nones
    train_ds.data = [d for d in train_ds.data if d is not None] 
    
    
    # Use baseline test loader for strict comp
    test_ds = CLUTRRDataset(root, dset, "test", 100) # Loads all tests combined
    
    collator = NeSyCollator(tokenizer, device)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collator, num_workers=0)
    test_collate_base = CLUTRRDataset.collate_fn
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=test_collate_base, num_workers=0)
    
    # 2. Model
    model = NeSyRoBERTa(device, tokenizer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # 3. Loop
    epochs = 10 
    
    print("Starting Training (Scheme A + B)...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        accum_aux = 0
        accum_cons = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for batch in pbar:
            if batch is None: continue
            
            optimizer.zero_grad()
            # lambda1 (NextHop) = 1.0, lambda_cons (Consistency) = 5.0
            out = model(batch, lambda1=0, lambda_cons=5.0) 
            
            loss = out['loss']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            accum_aux += out['losses']['nexthop']
            accum_cons += out['losses']['cons']
            
            pbar.set_postfix({
                'L_main': f"{out['losses']['main']:.3f}", 
                'L_aux': f"{out['losses']['nexthop']:.3f}",
                'L_cons': f"{out['losses']['cons']:.3f}"
            })
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Done. Loss: {avg_loss:.4f} (Aux: {accum_aux/len(train_loader):.4f}, Cons: {accum_cons/len(train_loader):.4f})")
        
        # 4. Evaluation
        print(f"--> Evaluating Epoch {epoch+1}...")
        metrics = evaluate_detailed(model, test_loader, device)
        
        print(f"  Overall Acc: {metrics['overall']:.4f}")
        print(f"  Short Hop (2-3): {metrics['short_hop']:.4f}")
        print(f"  Long Hop (>=6):  {metrics['long_hop']:.4f}")
        
        consistency = run_consistency_check(model, test_ds, device, sample_limit=200)
        print(f"  Bijective Consistency: {consistency:.4f}")
        
        # wandb.log({
        #     "epoch": epoch+1,
        #     "train_loss": avg_loss,
        #     "val_acc": metrics['overall'],
        #     "val_long_hop": metrics['long_hop'],
        #     "consistency": consistency
        # })

def evaluate_detailed(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    by_hop_correct = {}
    by_hop_total = {}
    
    with torch.no_grad():
        for batch in loader:
            (contexts, queries, context_splits, _) = batch[0]
            y_target = batch[1].to(device)
            hops = batch[2] # Tensr
            
            # Reconstruct Inputs
            batch_input_texts = []
            for i, (start, end) in enumerate(context_splits):
                story_sents = contexts[start:end]
                story_text = ". ".join(story_sents)
                sub, obj = queries[i]
                query_text = f"{sub} and {obj}"
                batch_input_texts.append((story_text, query_text))
                
            enc = model.tokenizer(batch_input_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            
            out = model.roberta(**enc)
            cls = out.last_hidden_state[:, 0, :]
            logits = model.classifier(cls)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == y_target).sum().item()
            total += len(y_target)
            
            # Hops Split
            # Assuming hops is tensor of ints
            preds_np = preds.cpu().numpy()
            targets_np = y_target.cpu().numpy()
            hops_np = hops.cpu().numpy()
            
            for h, p, t in zip(hops_np, preds_np, targets_np):
                if h not in by_hop_total:
                    by_hop_total[h] = 0
                    by_hop_correct[h] = 0
                by_hop_total[h] += 1
                if p == t:
                    by_hop_correct[h] += 1
                    
    # Metrics
    overall = correct / total if total > 0 else 0
    
    # Aggregates
    short_corr = sum([by_hop_correct.get(h,0) for h in [2,3]])
    short_tot = sum([by_hop_total.get(h,0) for h in [2,3]])
    short_acc = short_corr / short_tot if short_tot > 0 else 0
    
    long_corr = sum([by_hop_correct.get(h,0) for h in range(6, 15)])
    long_tot = sum([by_hop_total.get(h,0) for h in range(6, 15)])
    long_acc = long_corr / long_tot if long_tot > 0 else 0
    
    return {
        "overall": overall,
        "short_hop": short_acc,
        "long_hop": long_acc
    }

def run_consistency_check(model, dataset, device, sample_limit=200):
    # Simplified Bijective Logic adapted from diagnose_path.py
    import re
    model.eval()
    
    subset = dataset.data[:sample_limit]
    consistent = 0
    total = 0
    
    # Pre-map Names
    def apply_bijective_map(text, mapping):
        sorted_names = sorted(mapping.keys(), key=len, reverse=True)
        escaped_names = [re.escape(n) for n in sorted_names]
        if not escaped_names: return text
        pattern = re.compile(r'\b(' + '|'.join(escaped_names) + r')\b')
        def replacement(match): return mapping[match.group(0)]
        return pattern.sub(replacement, text)
    
    for row in tqdm(subset, desc="Consistency Check", leave=False):
        story = row[2]
        query = eval(row[3])
        target_str = row[5] # label
        
        # 1. Base Prediction
        enc_base = model.tokenizer([(story, f"{query[0]} and {query[1]}")], return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            out_base = model.roberta(**enc_base)
            pred_base = torch.argmax(model.classifier(out_base.last_hidden_state[:, 0, :])).item()
            
        # 2. Swap
        names = list(set(re.findall(r"\[(.*?)\]", story)))
        if len(names) < 2: continue
        
        # Shift Map: A->B, B->C ... Z->A
        shuffled = names[1:] + [names[0]]
        mapping = {n: s for n, s in zip(names, shuffled)}
        
        story_mod = apply_bijective_map(story, mapping)
        q_mod_0 = mapping.get(query[0], query[0])
        q_mod_1 = mapping.get(query[1], query[1])
        
        enc_mod = model.tokenizer([(story_mod, f"{q_mod_0} and {q_mod_1}")], return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            out_mod = model.roberta(**enc_mod)
            pred_mod = torch.argmax(model.classifier(out_mod.last_hidden_state[:, 0, :])).item()
            
        if pred_base == pred_mod:
            consistent += 1
        total += 1
        
    return consistent / total if total > 0 else 0


if __name__ == "__main__":
    run_training()
