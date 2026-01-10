
import sys
import os
import math
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


from clutrr.nesy_utils import parse_graph_and_path, apply_bijective_map
from clutrr.entity_alignment import align_entity_spans_to_tokens

# ==========================================
# 1. NeSy Dataset Wrapper
# ==========================================
class NeSyCLUTRRDataset(CLUTRRDataset):
    def __init__(
        self,
        root,
        dataset,
        split,
        data_percentage=100,
        tokenizer=None,
        augment=False,
        augment_seed=None,
    ):
        super().__init__(root, dataset, split, data_percentage)
        self.tokenizer = tokenizer
        self.augment = augment
        self.augment_seed = augment_seed
        
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
        path_rel_labels = graph_info.get('path_relation_labels', [])

        # Query indices are in the all_names index space
        try:
            sub_name, obj_name = query
            sub_idx = all_names.index(sub_name)
            obj_idx = all_names.index(obj_name)
        except Exception:
            return None
        
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
            'query_indices': (sub_idx, obj_idx),
            'target_id': target_id,
            'path_indices': graph_info['path_node_indices'], 
            'path_rel_labels': path_rel_labels,
            'node_spans': node_spans,
            'num_nodes': len(all_names),
            'hops': len(graph_info['path_node_indices']) - 1
        }
        
        # 4. Augmentation (Consistency Training)
        if self.augment:
            # Random bijection over names (optionally deterministic per sample)
            rng = random.Random(self.augment_seed + i) if self.augment_seed is not None else random

            names = list(all_names)
            if len(names) >= 2:
                shuffled = list(names)
                # Avoid identity mapping when possible
                for _ in range(10):
                    rng.shuffle(shuffled)
                    if any(a != b for a, b in zip(names, shuffled)):
                        break
                mapping = {n: s for n, s in zip(names, shuffled)}

                aug_story = apply_bijective_map(story_str, mapping)
                aug_query = (mapping.get(query[0], query[0]), mapping.get(query[1], query[1]))
                aug_all_names = [mapping[n] for n in all_names]

                # Alignment on augmented text
                aug_alignments = align_entity_spans_to_tokens(aug_story, aug_all_names, self.tokenizer)
                aug_node_spans = []
                valid_aug = True
                for j, name in enumerate(aug_all_names):
                    res = aug_alignments.get(name)
                    if res and res['token_span']:
                        aug_node_spans.append(res['token_span'])
                    else:
                        aug_node_spans.append(None)
                        # If the missing entity is used on the path, drop
                        # (path indices remain identical under renaming)
                        if graph_info['path_node_indices'].count(j) > 0:
                            valid_aug = False
                if not valid_aug:
                    return None

                item['aug_story'] = aug_story
                item['aug_query'] = aug_query
                item['aug_node_spans'] = aug_node_spans
            else:
                # Not enough names to permute
                item['aug_story'] = story_str
                item['aug_query'] = query
                item['aug_node_spans'] = node_spans
            
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
        query_indices = torch.tensor([b['query_indices'] for b in batch], dtype=torch.long).to(self.device)
        
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

        max_rel_len = max([len(b.get('path_rel_labels', [])) for b in batch])
        path_rel_ids = torch.full((len(batch), max_rel_len), -1, dtype=torch.long).to(self.device)
        
        max_entities = max([b['num_nodes'] for b in batch])
        entity_spans = torch.full((len(batch), max_entities, 2), -1, dtype=torch.long).to(self.device)
        aug_entity_spans = torch.full((len(batch), max_entities, 2), -1, dtype=torch.long).to(self.device)
        
        for i, b in enumerate(batch):
            p_len = len(b['path_indices'])
            path_node_ids[i, :p_len] = torch.tensor(b['path_indices'], dtype=torch.long)

            rels = b.get('path_rel_labels', [])
            # Align length: rel labels is path_len-1
            for j, rel in enumerate(rels):
                if j >= max_rel_len:
                    break
                path_rel_ids[i, j] = relation_id_map.get(rel, relation_id_map.get('nothing', 0))
            
            spans = b['node_spans'] 
            for j, s in enumerate(spans):
                if s is not None:
                    entity_spans[i, j, 0] = s[0]
                    entity_spans[i, j, 1] = s[1]

            if 'aug_node_spans' in b:
                aug_spans = b['aug_node_spans']
                for j, s in enumerate(aug_spans):
                    if s is not None:
                        aug_entity_spans[i, j, 0] = s[0]
                        aug_entity_spans[i, j, 1] = s[1]
        
        return {
            'input_ids': enc.input_ids,
            'attention_mask': enc.attention_mask, 
            'labels': targets,
            'hops': hops,
            'path_node_ids': path_node_ids, 
            'path_rel_ids': path_rel_ids,
            'entity_spans': entity_spans,
            'query_indices': query_indices,
            'raw_batch': batch,
            'aug_input_ids': enc_aug.input_ids if enc_aug else None,
            'aug_attention_mask': enc_aug.attention_mask if enc_aug else None,
            'aug_entity_spans': aug_entity_spans if enc_aug else None
        }


class RelationConditionedEntityAttention(nn.Module):
    def __init__(self, hidden_size: int, num_relations: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_relations = num_relations

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.rel_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_relations),
        )

        # Relation-conditioned FiLM on values: V_ij = V_j * (1+gamma_ij) + beta_ij
        self.rel_gamma = nn.Embedding(num_relations, hidden_size)
        self.rel_beta = nn.Embedding(num_relations, hidden_size)
        self.rel_bias = nn.Parameter(torch.zeros(num_relations))

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_size)

    def pair_rel_logits(self, e_i: torch.Tensor, e_j: torch.Tensor) -> torch.Tensor:
        # e_i, e_j: (H,)
        feats = torch.cat([e_i, e_j, e_i * e_j], dim=-1)
        return self.rel_mlp(feats)

    def forward(self, entity_embs: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # entity_embs: (B, N, H)
        # valid_mask: (B, N) bool
        B, N, H = entity_embs.shape

        # Compute per-sample to avoid O(B*N^2*H) blowups from padding.
        out = torch.zeros_like(entity_embs)
        for b in range(B):
            n_valid = int(valid_mask[b].sum().item())
            if n_valid <= 0:
                continue

            E = entity_embs[b, :n_valid]  # (n, H)
            q = self.q_proj(E)
            k = self.k_proj(E)
            v = self.v_proj(E)

            # Pairwise relation logits: (n, n, R)
            e_i = E.unsqueeze(1).expand(n_valid, n_valid, H)
            e_j = E.unsqueeze(0).expand(n_valid, n_valid, H)
            feats = torch.cat([e_i, e_j, e_i * e_j], dim=-1)
            rel_logits = self.rel_mlp(feats)
            rel_probs = torch.softmax(rel_logits, dim=-1)

            scores = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(H)  # (n, n)
            scores = scores + torch.matmul(rel_probs, self.rel_bias)    # (n, n)

            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            gamma = torch.matmul(rel_probs, self.rel_gamma.weight)  # (n, n, H)
            beta = torch.matmul(rel_probs, self.rel_beta.weight)    # (n, n, H)
            v_expand = v.unsqueeze(0).expand(n_valid, n_valid, H)   # (n, n, H)
            v_cond = v_expand * (1.0 + gamma) + beta

            msg = torch.sum(attn.unsqueeze(-1) * v_cond, dim=1)  # (n, H)
            out[b, :n_valid] = self.ln(E + self.dropout(msg))

        return out

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

        # Compositionality-aware: relation-conditioned entity attention + query-pair head
        self.entity_attn = RelationConditionedEntityAttention(
            hidden_size=self.hidden_size,
            num_relations=21,
            dropout=0.1,
        )
        self.pair_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 21),
        )
        
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

    def compute_logits(self, input_ids, attention_mask, entity_spans, query_indices):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = out.last_hidden_state
        cls_output = sequence_output[:, 0, :]
        logits_cls = self.classifier(cls_output)

        entity_embs = self.get_entity_embeddings(sequence_output, entity_spans)
        valid_mask = (entity_spans[:, :, 0] != -1)
        entity_embs_upd = self.entity_attn(entity_embs, valid_mask)

        sub_idx = query_indices[:, 0].clamp(min=0)
        obj_idx = query_indices[:, 1].clamp(min=0)
        b_idx = torch.arange(entity_embs_upd.size(0), device=entity_embs_upd.device)
        e_sub = entity_embs_upd[b_idx, sub_idx]
        e_obj = entity_embs_upd[b_idx, obj_idx]
        pair_feats = torch.cat([e_sub, e_obj, e_sub * e_obj], dim=-1)
        logits_pair = self.pair_classifier(pair_feats)

        logits = logits_cls + logits_pair
        return logits, entity_embs_upd

    def forward(self, batch_data, lambda1=1.0, lambda_cons=0.0, lambda_rel=1.0):
        # Unpack
        input_ids = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        labels = batch_data['labels']
        entity_spans = batch_data['entity_spans']
        path_node_ids = batch_data['path_node_ids']
        path_rel_ids = batch_data.get('path_rel_ids')
        query_indices = batch_data['query_indices']
        
        # 1. Forward (Original)
        logits_orig, entity_embs_upd = self.compute_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_spans=entity_spans,
            query_indices=query_indices,
        )
        main_loss = nn.functional.cross_entropy(logits_orig, labels)
        
        # 2. Consistency Loss (Scheme B)
        cons_loss = torch.tensor(0.0).to(self.device)
        if batch_data.get('aug_input_ids') is not None:
            # Forward Augmented (same architecture)
            logits_aug, _ = self.compute_logits(
                input_ids=batch_data['aug_input_ids'],
                attention_mask=batch_data['aug_attention_mask'],
                entity_spans=batch_data['aug_entity_spans'],
                query_indices=query_indices,
            )
            
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
        entity_embs = entity_embs_upd
        
        loss_nexthop = torch.tensor(0.0).to(self.device)
        total_hops = 0
        batch_nexthop_loss = torch.tensor(0.0, device=self.device)
        
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

        # 4. Relation Supervision on path edges (proof-state-like)
        loss_rel = torch.tensor(0.0).to(self.device)
        if path_rel_ids is not None:
            total_edges = 0
            rel_loss_sum = torch.tensor(0.0, device=self.device)
            for i in range(len(input_ids)):
                path = path_node_ids[i]
                valid_path = path[path != -1]
                if len(valid_path) < 2:
                    continue

                # rel ids length is path_len-1, padded with -1
                rels = path_rel_ids[i]
                sample_ent_embs = entity_embs[i]
                for t in range(len(valid_path) - 1):
                    if t >= rels.numel():
                        break
                    rel_id = rels[t].item()
                    if rel_id == -1:
                        continue
                    u = valid_path[t].item()
                    v = valid_path[t + 1].item()
                    if u < 0 or v < 0:
                        continue
                    rel_logits_uv = self.entity_attn.pair_rel_logits(sample_ent_embs[u], sample_ent_embs[v]).unsqueeze(0)
                    rel_target = torch.tensor([rel_id], dtype=torch.long, device=self.device)
                    rel_loss_sum += nn.functional.cross_entropy(rel_logits_uv, rel_target)
                    total_edges += 1

            if total_edges > 0:
                loss_rel = rel_loss_sum / total_edges
            
        total_loss = main_loss + lambda1 * loss_nexthop + lambda_cons * cons_loss + lambda_rel * loss_rel
        
        return {
            'loss': total_loss,
            'losses': {
                'main': main_loss.item(), 
                'nexthop': loss_nexthop.item(),
                'cons': cons_loss.item(),
                'rel': loss_rel.item(),
            },
            'logits': logits_orig
        }

# ==========================================
# 3. Training Loop
# ==========================================
def run_training():
    set_seed(42)
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
    
    
    # Use NeSy test loader so entity spans / path rels are available for the new architecture
    test_ds = NeSyCLUTRRDataset(
        root,
        dset,
        "test",
        100,
        tokenizer=tokenizer,
        augment=True,
        augment_seed=999,
    )
    test_ds.data = [d for d in test_ds.data if d is not None]
    
    collator = NeSyCollator(tokenizer, device)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collator, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collator, num_workers=0)
    
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
            out = model(batch, lambda1=1.0, lambda_cons=5.0, lambda_rel=1.0)
            
            loss = out['loss']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            accum_aux += out['losses']['nexthop']
            accum_cons += out['losses']['cons']
            accum_rel = out['losses']['rel'] if 'rel' in out['losses'] else 0.0
            
            pbar.set_postfix({
                'L_main': f"{out['losses']['main']:.3f}", 
                'L_aux': f"{out['losses']['nexthop']:.3f}",
                'L_cons': f"{out['losses']['cons']:.3f}",
                'L_rel': f"{out['losses']['rel']:.3f}",
            })
            
        avg_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch+1} Done. Loss: {avg_loss:.4f} "
            f"(Aux: {accum_aux/len(train_loader):.4f}, Cons: {accum_cons/len(train_loader):.4f})"
        )
        
        # 4. Evaluation
        print(f"--> Evaluating Robustness Epoch {epoch+1}...")
        metrics = evaluate_robustness(model, test_loader, device)
        
        print(f"  Overall Acc (Base):   {metrics['overall']:.4f}")
        print(f"  Renamed Acc (Mod):    {metrics['renamed']:.4f}")
        print(f"  Consistency:          {metrics['consistency']:.4f}")
        print(f"  Consistent & Correct: {metrics['consistent_and_correct']:.4f}")
        print(f"  Short Hop (2-3):      {metrics['short_hop']:.4f}")
        print(f"  Long Hop (>=6):       {metrics['long_hop']:.4f}")

def evaluate_robustness(model, loader, device):
    model.eval()
    
    # Metrics containers
    total = 0
    correct_base = 0
    correct_mod = 0
    consistent = 0
    consistent_and_correct = 0
    
    count_changed_story = 0
    count_changed_query = 0
    
    by_hop_total = {}
    by_hop_correct = {}
    
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            y_target = batch['labels']
            hops = batch['hops']

            logits_base, _ = model.compute_logits(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                entity_spans=batch['entity_spans'],
                query_indices=batch['query_indices'],
            )
            preds_base = torch.argmax(logits_base, dim=1)

            logits_mod, _ = model.compute_logits(
                input_ids=batch['aug_input_ids'],
                attention_mask=batch['aug_attention_mask'],
                entity_spans=batch['aug_entity_spans'],
                query_indices=batch['query_indices'],
            )
            preds_mod = torch.argmax(logits_mod, dim=1)

            # Changed rates from raw_batch
            for b in batch['raw_batch']:
                story_text = b['story']
                query_text = f"{b['query'][0]} and {b['query'][1]}"
                story_mod = b.get('aug_story', story_text)
                query_mod = f"{b.get('aug_query', b['query'])[0]} and {b.get('aug_query', b['query'])[1]}"
                if story_mod != story_text:
                    count_changed_story += 1
                if query_mod != query_text:
                    count_changed_query += 1
            
            # Updates
            total += len(y_target)
            
            correct_mask = (preds_base == y_target)
            correct_mod_mask = (preds_mod == y_target)
            consistent_mask = (preds_base == preds_mod)
            
            correct_base += correct_mask.sum().item()
            correct_mod += correct_mod_mask.sum().item()
            consistent += consistent_mask.sum().item()
            
            # Consistent AND Correct: (Pred_Base == Pred_Mod) AND (Pred_Base == Label)
            consistent_and_correct += (consistent_mask & correct_mask).sum().item()
            
            # Hop Analysis (on Base)
            preds_np = preds_base.cpu().numpy()
            targets_np = y_target.cpu().numpy()
            hops_np = hops.cpu().numpy()
             
            for h, p, t in zip(hops_np, preds_np, targets_np):
                if h not in by_hop_total: by_hop_total[h] = 0; by_hop_correct[h] = 0
                by_hop_total[h] += 1
                if p == t: by_hop_correct[h] += 1

    # Final Stats
    acc_overall = correct_base / total if total > 0 else 0
    acc_renamed = correct_mod / total if total > 0 else 0
    prob_consistent = consistent / total if total > 0 else 0
    prob_robust_correct = consistent_and_correct / total if total > 0 else 0
    
    changed_story_rate = count_changed_story / total if total > 0 else 0
    changed_query_rate = count_changed_query / total if total > 0 else 0
    
    short_corr = sum([by_hop_correct.get(h,0) for h in [2,3]])
    short_tot = sum([by_hop_total.get(h,0) for h in [2,3]])
    short_acc = short_corr/short_tot if short_tot>0 else 0
    
    long_corr = sum([by_hop_correct.get(h,0) for h in range(6, 15)])
    long_tot = sum([by_hop_total.get(h,0) for h in range(6, 15)])
    long_acc = long_corr/long_tot if long_tot>0 else 0
    
    print(f"[Stats] Evaluated {total} samples.")
    print(f"  Changed Story Rate:   {changed_story_rate:.4f}")
    print(f"  Changed Query Rate:   {changed_query_rate:.4f}")
    
    return {
        "overall": acc_overall,
        "renamed": acc_renamed,
        "consistency": prob_consistent,
        "consistent_and_correct": prob_robust_correct,
        "short_hop": short_acc,
        "long_hop": long_acc
    }


if __name__ == "__main__":
    run_training()
