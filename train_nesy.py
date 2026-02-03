
import sys
import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import wandb
import random
import argparse
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    RobertaTokenizerFast, RobertaModel,
    DebertaTokenizerFast, DebertaModel,
    DebertaV2TokenizerFast, DebertaV2Model,
    BertTokenizerFast, BertModel
)

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


import re
import json
import glob
import hashlib
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from clutrr.nesy_utils import parse_graph_and_path, apply_bijective_map
from clutrr.entity_alignment import align_entity_spans_to_tokens


def _unwrap_model(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _is_main_process() -> bool:
    return _rank() == 0


def _setup_ddp() -> tuple[int, int, int]:
    """Initialize torch.distributed if launched via torchrun.

    Returns: (rank, world_size, local_rank)
    """
    if not ("RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ):
        raise RuntimeError(
            "DDP 需要用 torchrun 启动，例如：\n"
            "  torchrun --standalone --nproc_per_node=4 train_nesy.py --strategy ddp --gpus 0,1,2,3 ..."
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

# ==========================================
# 1. NeSy Dataset Wrappers
# ==========================================
class NeSyRuleTakerDataset(Dataset):
    def __init__(self, split="train", tokenizer=None, augment=False, augment_seed=None, limit=None, dataset_dir=None, depths=None):
        self.tokenizer = tokenizer
        self.augment = augment
        self.augment_seed = augment_seed
        self.label_map = {'not_entailment': 0, 'entailment': 1}
        self.name_pattern = re.compile(r"\b[A-Z][a-z]+\b")
        
        self.data = []
        
        # Load Raw Data
        if dataset_dir:
            files = []
            if depths:
                print(f"Loading RuleTaker from local: {dataset_dir} (split={split}, depths={depths})")
                for d in depths:
                     # Check normal numeric depths
                     fpath = os.path.join(dataset_dir, f"depth-{d}", f"{split}.jsonl")
                     if os.path.exists(fpath):
                         files.append(fpath)
                     else:
                         # Handle extra datasets which might not match "depth-X" perfectly or user put full name in depths?
                         # The user said "--ruletaker_extra_test_depths" but also passed depths list here.
                         pass
            else:
                 print(f"Loading RuleTaker from local: {dataset_dir} (split={split}, auto-detect depths)")
                 pattern = os.path.join(dataset_dir, "depth-*", f"{split}.jsonl")
                 files = sorted(glob.glob(pattern))
                 if not files:
                      pattern = os.path.join(dataset_dir, f"{split}.jsonl")
                      files = sorted(glob.glob(pattern))
            
            print(f"  Found {len(files)} files: {[os.path.basename(os.path.dirname(f)) for f in files]}")
            for fpath in files:
                with open(fpath, 'r') as f:
                    for line in f:
                        obj = json.loads(line)
                        context = obj['context']
                        for q in obj['questions']:
                            self.data.append({
                                'context': context,
                                'question': q['text'],
                                'label': 1 if q['label'] is True else 0 
                            })
            if limit and limit < len(self.data):
                self.data = self.data[:limit]
        else:
            hf_split = "validation" if split == "dev" else split
            ds = load_dataset("tasksource/ruletaker", split=hf_split)
            if limit and limit < len(ds):
                 ds = ds.select(range(limit))
            for item in ds:
                raw_label = item['label']
                label_id = raw_label if isinstance(raw_label, int) else self.label_map.get(str(raw_label), 0)
                self.data.append({
                    'context': item.get('context', item.get('text', '')),
                    'question': item.get('question', ''),
                    'label': label_id
                })
        
        # Caching Logic
        self.cached_features = None
        if dataset_dir and self.tokenizer:
            # Create a cache key based on tokenizer and data size
            model_name = self.tokenizer.name_or_path.replace("/", "_")
            cache_name = f"cached_{split}_{model_name}_{len(self.data)}.pt"
            self.cache_path = os.path.join(dataset_dir, cache_name)
            
            if os.path.exists(self.cache_path):
                print(f"Loading cached features from {self.cache_path}...")
                self.cached_features = torch.load(self.cache_path)
            else:
                print(f"Pre-tokenizing and caching to {self.cache_path}...")
                self.cached_features = self._preprocess_all()
                torch.save(self.cached_features, self.cache_path)

    def __len__(self):
        return len(self.data)
        
    def extract_names(self, text):
        matches = self.name_pattern.findall(text)
        exclude = {"If", "All", "Then", "And", "Or", "Not", "True", "False", "The", "A", "An"}
        names = sorted(list(set([m for m in matches if m not in exclude])))
        return names
        
    def _preprocess_all(self):
        features = []
        for i, item in enumerate(tqdm(self.data, desc="Tokenizing")):
            context = item['context']
            question = item['question']
            
            # 1. Tokenize (Clean)
            enc = self.tokenizer(
                context, 
                question, 
                truncation=True, 
                max_length=512, 
                # return_tensors='pt' # We store as list to save memory then dict
            )
            
            # 2. Entity Extraction & Alignment
            full_text = f"{context} {question}"
            all_names = self.extract_names(full_text)
            alignments = align_entity_spans_to_tokens(context, all_names, self.tokenizer)
            
            node_spans = []
            for name in all_names:
                res = alignments.get(name)
                node_spans.append(res['token_span'] if (res and res['token_span']) else None)
                
            # Query Entities
            q_names = [n for n in all_names if n in question]
            if len(q_names) >= 2:
                p1 = question.find(q_names[0])
                p2 = question.find(q_names[1])
                sub_name, obj_name = (q_names[0], q_names[1]) if p1 < p2 else (q_names[1], q_names[0])
            elif len(q_names) == 1:
                sub_name, obj_name = q_names[0], q_names[0]
            else:
                sub_name, obj_name = None, None
            
            feat = {
                'input_ids': enc['input_ids'],
                'attention_mask': enc['attention_mask'],
                'node_spans': node_spans,
                'num_nodes': len(all_names),
                'query_indices': (all_names.index(sub_name) if sub_name else -1, all_names.index(obj_name) if obj_name else -1),
                'metadata': {
                    'sub_name': sub_name, 'obj_name': obj_name, 'all_names': all_names
                }
            }
            features.append(feat)
        return features

    def get_aug_map(self, names, seed_offset):
        if not self.augment or len(names) < 2:
            return None
        rng = random.Random(self.augment_seed + seed_offset) if self.augment_seed is not None else random
        shuffled = list(names)
        for _ in range(10):
            rng.shuffle(shuffled)
            if any(a != b for a, b in zip(names, shuffled)):
                break
        return {n: s for n, s in zip(names, shuffled)}

    def __getitem__(self, i):
        item = self.data[i]
        label_id = item['label']
        
        # Check Cache
        if self.cached_features:
            feat = self.cached_features[i]
            input_ids = feat['input_ids']
            attention_mask = feat['attention_mask']
            node_spans = feat['node_spans']
            num_nodes = feat['num_nodes']
            query_indices = feat['query_indices']
            # For augmentation, we need raw data
            context = item['context']
            question = item['question']
            all_names = feat['metadata']['all_names'] # Use cached names
            sub_name = feat['metadata']['sub_name']
            obj_name = feat['metadata']['obj_name']
        else:
            # Fallback (On-the-fly) - copied logic
            context = item['context']
            question = item['question']
            full_text = f"{context} {question}"
            all_names = self.extract_names(full_text)
            enc = self.tokenizer(context, question, truncation=True, max_length=512)
            input_ids = enc['input_ids']
            attention_mask = enc['attention_mask']
            alignments = align_entity_spans_to_tokens(context, all_names, self.tokenizer)
            node_spans = [alignments.get(n)['token_span'] if alignments.get(n) and alignments.get(n)['token_span'] else None for n in all_names]
            num_nodes = len(all_names)
            
            q_names = [n for n in all_names if n in question]
            if len(q_names) >= 2:
                p1 = question.find(q_names[0])
                p2 = question.find(q_names[1])
                sub_name, obj_name = (q_names[0], q_names[1]) if p1 < p2 else (q_names[1], q_names[0])
            elif len(q_names) == 1:
                sub_name, obj_name = q_names[0], q_names[0]
            else:
                sub_name, obj_name = None, None
            query_indices = (all_names.index(sub_name) if sub_name else -1, all_names.index(obj_name) if obj_name else -1)

        result = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'query_indices': query_indices,
            'target_id': label_id,
            'node_spans': node_spans,
            'num_nodes': num_nodes,
            'hops': 0,
            
            # Dummy Path
            'path_indices': [-1],
            'path_rel_labels': [],
            
            # Raw text (needed for aug if not cached, or verifying)
            'story': context,
            'query_text_raw': question
        }
        
        # 2. Augmentation (Still partly on-the-fly for tokens, but we use cached names)
        # Note: Pre-computing ALL augmentations explodes storage. We do it on dynamic epoch?
        # But user wants speed.
        # If we use cached_features, main bottleneck is gone. Augmentation overhead remains but less frequent.
        
        aug_map = self.get_aug_map(all_names, i)
        if aug_map:
            aug_story = apply_bijective_map(context, aug_map)
            aug_question = apply_bijective_map(question, aug_map)
            aug_all_names = [aug_map[n] for n in all_names]
            
            # Tokenization here is unavoidable if we want random renaming
            # But we can assume it's fast enough or set num_workers > 0
            aug_alignments = align_entity_spans_to_tokens(aug_story, aug_all_names, self.tokenizer)
            aug_node_spans = []
            for name in aug_all_names:
                res = aug_alignments.get(name)
                aug_node_spans.append(res['token_span'] if (res and res['token_span']) else None)
            
            result.update({
                'aug_story': aug_story,
                'aug_query_text_raw': aug_question,
                'aug_node_spans': aug_node_spans
            })
        else:
             # If no aug, just point to original (already tensor)
             # But collator expects strings for augmentation path if we don't return tensor
             # To be consistent, we return raw strings and let collator handle aug?
             # Or we return tensors for aug too?
             # Let's let Collator detect.
            result.update({
                'aug_story': context,
                'aug_query_text_raw': question, 
                'aug_node_spans': node_spans
            })
            
        return result



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

        # Query indices: Use pre-computed query_edge (row[13]) to avoid O(N) search and name matching issues
        try:
            # row[13] is string "(u, v)"
            query_edge = eval(row[13]) 
            sub_idx, obj_idx = query_edge
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
        else:
            # Augmentation disabled: fill aug fields with originals to prevent downstream errors
            item['aug_story'] = story_str
            item['aug_query'] = query
            item['aug_node_spans'] = node_spans
            
        return item

def collate_nesy(batch):
    # Old function unused
    pass

class NeSyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch: return None
        
        # Check if pre-tokenized (cached)
        is_pre_tokenized = isinstance(batch[0].get('input_ids'), torch.Tensor)
        
        if is_pre_tokenized:
            input_ids = pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention_mask = pad_sequence([b['attention_mask'] for b in batch], batch_first=True, padding_value=0)
        else:
            stories = [b['story'] for b in batch]
            if 'query_text_raw' in batch[0]:
                queries = [b['query_text_raw'] for b in batch]
            else:
                queries = [f"{b['query'][0]} and {b['query'][1]}" for b in batch]

            enc = self.tokenizer(
                stories,
                queries, 
                padding=True,
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            )
            input_ids = enc.input_ids
            attention_mask = enc.attention_mask

        targets = torch.tensor([b['target_id'] for b in batch], dtype=torch.long)
        hops = torch.tensor([b['hops'] for b in batch], dtype=torch.long)
        query_indices = torch.tensor([b['query_indices'] for b in batch], dtype=torch.long)
        
        # Prepare Aux Labels
        max_path_len = max([len(b['path_indices']) for b in batch])
        path_node_ids = torch.full((len(batch), max_path_len), -1, dtype=torch.long)

        max_rel_len = max([len(b.get('path_rel_labels', [])) for b in batch])
        path_rel_ids = torch.full((len(batch), max_rel_len), -1, dtype=torch.long)
        
        max_entities = max([b['num_nodes'] for b in batch])
        entity_spans = torch.full((len(batch), max_entities, 2), -1, dtype=torch.long)
        aug_entity_spans = torch.full((len(batch), max_entities, 2), -1, dtype=torch.long)
        
        for i, b in enumerate(batch):
            p_len = len(b['path_indices'])
            path_node_ids[i, :p_len] = torch.tensor(b['path_indices'], dtype=torch.long)

            rels = b.get('path_rel_labels', [])
            for j, rel in enumerate(rels):
                if j >= max_rel_len: break
                # rel might be int (RuleTaker dummy) or string (CLUTRR)
                if isinstance(rel, int):
                    path_rel_ids[i, j] = rel
                else:
                    path_rel_ids[i, j] = relation_id_map.get(rel, relation_id_map.get('nothing', 0))
            
            spans = b['node_spans'] 
            for j, s in enumerate(spans):
                if s is not None:
                    entity_spans[i, j, 0] = s[0]
                    entity_spans[i, j, 1] = s[1]

        # Augmented Encoding
        enc_aug = None
        if 'aug_story' in batch[0]:
            # Always on-the-fly for augmentation to save massive storage/pre-calc
            aug_stories = [b['aug_story'] for b in batch]
            if 'aug_query_text_raw' in batch[0]:
                 aug_queries = [b['aug_query_text_raw'] for b in batch]
            else:
                 if 'aug_query' in batch[0] and isinstance(batch[0]['aug_query'], (list, tuple)):
                     aug_queries = [f"{b['aug_query'][0]} and {b['aug_query'][1]}" for b in batch]
                 else:
                     if 'query_text_raw' in batch[0]:
                        aug_queries = [b['query_text_raw'] for b in batch]
                     else:
                        aug_queries = [f"{b['query'][0]} and {b['query'][1]}" for b in batch]

            enc_aug = self.tokenizer(
                aug_stories,
                aug_queries,
                padding=True,
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            )
            
            # Aug spans filling
            for i, b in enumerate(batch):
                if 'aug_node_spans' in b:
                    aug_spans = b['aug_node_spans']
                    for j, s in enumerate(aug_spans):
                        if s is not None:
                            aug_entity_spans[i, j, 0] = s[0]
                            aug_entity_spans[i, j, 1] = s[1]
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask, 
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
        return result


class RelationConditionedEntityAttention(nn.Module):
    def __init__(self, hidden_size: int, num_relations: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_relations = num_relations

        # Global Head (Composition)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # Hop Head (Routing/Navigation)
        self.q_hop = nn.Linear(hidden_size, hidden_size)
        self.k_hop = nn.Linear(hidden_size, hidden_size)
        self.v_hop = nn.Linear(hidden_size, hidden_size)
        # Goal-conditioned component for hop query: q(u, obj) = q_hop(u) + q_obj(obj)
        self.q_obj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rel_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_relations),
        )

        # Relation-conditioned FiLM on values
        # Shared between heads to enforce consistent relation semantics
        self.rel_gamma = nn.Embedding(num_relations, hidden_size)
        self.rel_beta = nn.Embedding(num_relations, hidden_size)
        
        # Attention Bias - Separate parameters to allow distinct connectivity patterns
        self.rel_bias = nn.Parameter(torch.zeros(num_relations))      # Global
        self.rel_bias_hop = nn.Parameter(torch.zeros(num_relations))  # Hop

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_size)

    def pair_rel_logits(self, e_i: torch.Tensor, e_j: torch.Tensor) -> torch.Tensor:
        # e_i, e_j: (H,)
        feats = torch.cat([e_i, e_j, e_i * e_j], dim=-1)
        return self.rel_mlp(feats)

    def forward(
        self,
        entity_embs: torch.Tensor,
        valid_mask: torch.Tensor,
        obj_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # entity_embs: (B, N, H)
        # valid_mask: (B, N) bool
        B, N, H = entity_embs.shape

        # Compute per-sample to avoid O(B*N^2*H) blowups from padding.
        out = torch.zeros_like(entity_embs)
        
        # Store Hop scores for supervision
        batch_scores_hop = torch.full((B, N, N), -1e4, device=entity_embs.device, dtype=entity_embs.dtype)
        
        for b in range(B):
            n_valid = int(valid_mask[b].sum().item())
            if n_valid <= 0:
                continue

            E = entity_embs[b, :n_valid]  # (n, H)
            
            # --- Common Relation Prediction ---
            e_i = E.unsqueeze(1).expand(n_valid, n_valid, H)
            e_j = E.unsqueeze(0).expand(n_valid, n_valid, H)
            feats = torch.cat([e_i, e_j, e_i * e_j], dim=-1)
            rel_logits = self.rel_mlp(feats)
            rel_probs = torch.softmax(rel_logits, dim=-1)
            
            # FiLM Parameters (Shared)
            gamma = torch.matmul(rel_probs, self.rel_gamma.weight)
            beta = torch.matmul(rel_probs, self.rel_beta.weight)

            # --- 1. Global Head (Composition) ---
            q = self.q_proj(E)
            k = self.k_proj(E)
            v = self.v_proj(E)
            
            scores = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(H)
            scores = scores + torch.matmul(rel_probs, self.rel_bias)
            attn = self.dropout(torch.softmax(scores, dim=-1))
            
            v_expand = v.unsqueeze(0).expand(n_valid, n_valid, H)
            v_cond = v_expand * (1.0 + gamma) + beta
            msg_global = torch.sum(attn.unsqueeze(-1) * v_cond, dim=1)

            # --- 2. Hop Head (Routing) ---
            q_h = self.q_hop(E)
            if obj_indices is not None:
                obj_idx = int(obj_indices[b].item())
                if 0 <= obj_idx < n_valid:
                    q_h = q_h + self.q_obj(E[obj_idx]).unsqueeze(0)
            k_h = self.k_hop(E)
            v_h = self.v_hop(E)
            
            scores_hop = torch.matmul(q_h, k_h.transpose(0, 1)) / math.sqrt(H)
            scores_hop = scores_hop + torch.matmul(rel_probs, self.rel_bias_hop)
            
            # Save for supervision
            batch_scores_hop[b, :n_valid, :n_valid] = scores_hop
            
            attn_hop = self.dropout(torch.softmax(scores_hop, dim=-1))
            
            v_h_expand = v_h.unsqueeze(0).expand(n_valid, n_valid, H)
            v_h_cond = v_h_expand * (1.0 + gamma) + beta # Reuse FiLM for consistency
            msg_hop = torch.sum(attn_hop.unsqueeze(-1) * v_h_cond, dim=1)

            # --- Fusion ---
            # Inject both messages into the update
            out[b, :n_valid] = self.ln(E + self.dropout(msg_global + msg_hop))

        return out, batch_scores_hop

# ==========================================
# 2. NeSy Transformer Model
# ==========================================
class NeSyRoBERTa(nn.Module):
    def __init__(self, device, tokenizer, model_type="roberta", num_labels=21, num_relations=21):
        super().__init__()
        self.device = device
        self.num_labels = num_labels
        
        self.model_type = model_type.lower()
        if self.model_type == "roberta":
            self.encoder = RobertaModel.from_pretrained("roberta-base")
        elif self.model_type == "roberta-large":
            self.encoder = RobertaModel.from_pretrained("roberta-large")
        elif self.model_type == "deberta":
            self.encoder = DebertaModel.from_pretrained("microsoft/deberta-base")
        elif self.model_type == "deberta-v3":
            self.encoder = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
        elif self.model_type == "deberta-v3-large":
            self.encoder = DebertaV2Model.from_pretrained("microsoft/deberta-v3-large")
        elif self.model_type == "bert":
            self.encoder = BertModel.from_pretrained("bert-base-uncased")
        elif self.model_type == "modernbert":
            # ModernBERT is hosted on HF Hub and may require remote code.
            try:
                self.encoder = AutoModel.from_pretrained("answerdotai/ModernBERT-base", trust_remote_code=True)
            except TypeError:
                # Older transformers versions may not accept trust_remote_code here.
                self.encoder = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.tokenizer = tokenizer # For debugging/resizing if needed
        self.hidden_size = self.encoder.config.hidden_size
        
        # Heads
        self.classifier = nn.Linear(self.hidden_size, num_labels) 

        # Compositionality-aware: relation-conditioned entity attention + query-pair head
        self.entity_attn = RelationConditionedEntityAttention(
            hidden_size=self.hidden_size,
            num_relations=num_relations,
            dropout=0.1,
        )
        self.pair_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_labels),
        )
        
        # Intermediate Relation Prediction (Optional)
        # Concatenate (E_i, E_i+1) -> Rel
        self.rel_proj = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_relations) # Relation space
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
            
            if len(sample_embs) == 0:
                 # Handle case where N_ent=0 (batch has no entities)
                 # Return empty 0xH tensor
                 emb_list.append(torch.zeros(0, self.hidden_size).to(self.device))
            else:
                 emb_list.append(torch.stack(sample_embs))
            
        return torch.stack(emb_list) # (B, N_ent, H)

    def compute_logits(self, input_ids, attention_mask, entity_spans, query_indices):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = out.last_hidden_state
        cls_output = sequence_output[:, 0, :]
        logits_cls = self.classifier(cls_output)

        entity_embs = self.get_entity_embeddings(sequence_output, entity_spans)
        valid_mask = (entity_spans[:, :, 0] != -1)
        # Returns: (updated_embs, attn_scores)
        obj_indices = query_indices[:, 1]
        entity_embs_upd, attn_scores = self.entity_attn(entity_embs, valid_mask, obj_indices=obj_indices)

        sub_idx = query_indices[:, 0].clamp(min=0)
        obj_idx = query_indices[:, 1].clamp(min=0)
        b_idx = torch.arange(entity_embs_upd.size(0), device=entity_embs_upd.device)
        e_sub = entity_embs_upd[b_idx, sub_idx]
        e_obj = entity_embs_upd[b_idx, obj_idx]
        pair_feats = torch.cat([e_sub, e_obj, e_sub * e_obj], dim=-1)
        logits_pair = self.pair_classifier(pair_feats)

        # Fusion: direct sum
        logits = logits_cls + logits_pair
        
        return logits, entity_embs_upd, attn_scores

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
        logits_orig, entity_embs_upd, attn_scores = self.compute_logits(
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
            logits_aug, _, _ = self.compute_logits(
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
            
            # Revised Next-Hop Logic:
            # Instead of hop_proj, we use the attention scores from `RelationConditionedEntityAttention`.
            # attn_scores[i] is (N, N). We want scores from node u to all other nodes.
            # The "logit" for node v is score(u, v).
            
            # Get the row of scores for each node u in the path
            # slice: [input_indices, :] -> (Steps, N)
            step_logits = attn_scores[i, input_indices, :]
            
            # Target is the index of node v.
            step_loss = nn.functional.cross_entropy(step_logits, target_indices)
            
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
                'rel': loss_rel.item()
            },
            'logits': logits_orig
        }

# ==========================================
# 3. Training Loop
# ==========================================
def run_training():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="deberta",
        choices=["roberta", "roberta-large", "deberta", "deberta-v3", "deberta-v3-large", "bert", "modernbert"],
        help="Model backbone type",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Train batch size (per process for DDP)")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Eval batch size (per process for DDP; eval runs on rank0 only)")
    parser.add_argument("--num_workers", type=int, default=12, help="DataLoader num_workers")
    parser.add_argument("--dataset", type=str, default="clutrr", choices=["clutrr", "ruletaker"], help="Dataset to use")
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="选择可见 GPU，例如 '0' 或 '0,1,2,3'。脚本会设置 CUDA_VISIBLE_DEVICES。",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="single",
        choices=["single", "dp", "ddp"],
        help="single=单卡；dp=DataParallel；ddp=DistributedDataParallel(推荐，多进程同步)",
    )
    # RuleTaker Specific Arguments
    parser.add_argument("--ruletaker_root", type=str, default="data/rule-reasoning-dataset-V2020.2.5.0/original")
    parser.add_argument("--ruletaker_train_depths", type=str, default="1,2", help="Comma separated depths for training, e.g. '0,1,2'")
    parser.add_argument("--ruletaker_test_depths", type=str, default="0,1,2,3,4,5", help="Comma separated depths for testing")
    parser.add_argument("--ruletaker_extra_test_depths", type=str, default=None, help="Extra depths e.g. '3ext,3ext-NatLang'")
    
    args = parser.parse_args()

    # IMPORTANT: set visible devices BEFORE any CUDA usage
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Distributed init (if needed)
    if args.strategy == "ddp":
        rank, world_size, local_rank = _setup_ddp()
        device = torch.device("cuda", local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(42)
    if _is_main_process():
        print(f"Device: {device}")
        print(f"Selected Model: {args.model_type}")
        print(f"Dataset: {args.dataset}")
        print(f"GPU Visible (CUDA_VISIBLE_DEVICES): {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        print(f"Strategy: {args.strategy} (rank={rank}, world_size={world_size}, local_rank={local_rank})")
    
    # 1. Load Tokenizer & Dataset
    if args.model_type == "roberta":
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    elif args.model_type == "roberta-large":
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
    elif args.model_type == "deberta":
        tokenizer = DebertaTokenizerFast.from_pretrained("microsoft/deberta-base")
    elif args.model_type == "deberta-v3":
        tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")
    elif args.model_type == "deberta-v3-large":
        tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-large")
    elif args.model_type == "bert":
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    elif args.model_type == "modernbert":
        try:
            tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", use_fast=True, trust_remote_code=True)
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", use_fast=True)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Config per dataset
    test_loaders = {} # Dictionary for RuleTaker multi-depth eval
    
    if args.dataset == "clutrr":
        num_labels = 21 # Relations
        num_latent_relations = 21
        root = "data"
        dset = "data_089907f8"
        print("Loading CLUTRR Data (Augmentation Enabled)...")
        train_ds = NeSyCLUTRRDataset(root, dset, "train", 100, tokenizer=tokenizer, augment=True)
        test_ds = NeSyCLUTRRDataset(root, dset, "test", 100, tokenizer=tokenizer, augment=True, augment_seed=999)
        test_loaders['default'] = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=NeSyCollator(tokenizer), num_workers=args.num_workers)

    elif args.dataset == "ruletaker":
        num_labels = 2 # Entailment vs Not
        num_latent_relations = 21 # Internal slots
        
        # Check local path
        local_path = args.ruletaker_root
        
        if os.path.exists(local_path):
             # Train Set
             train_depths = args.ruletaker_train_depths.split(",") if args.ruletaker_train_depths else None
             if train_depths: train_depths = [d.strip() for d in train_depths if d.strip()]
             
             print(f"Loading RuleTaker TRAIN from {local_path}...")
             train_ds = NeSyRuleTakerDataset("train", tokenizer=tokenizer, augment=True, dataset_dir=local_path, depths=train_depths)
             
             # Test Sets (Multi-depth)
             test_depth_list = args.ruletaker_test_depths.split(",") if args.ruletaker_test_depths else []
             if args.ruletaker_extra_test_depths:
                 test_depth_list += args.ruletaker_extra_test_depths.split(",")
             
             test_depth_list = [d.strip() for d in test_depth_list if d.strip()]
             
             print(f"Loading RuleTaker TEST Sets: {test_depth_list}...")
             for d in test_depth_list:
                 ds_name = f"depth-{d}"
                 ds = NeSyRuleTakerDataset("test", tokenizer=tokenizer, augment=True, augment_seed=999, dataset_dir=local_path, depths=[d])
                 if len(ds) > 0:
                     test_loaders[ds_name] = DataLoader(
                        ds,
                        batch_size=args.eval_batch_size,
                        shuffle=False,
                        collate_fn=NeSyCollator(tokenizer),
                        num_workers=args.num_workers,
                        pin_memory=True
                     )
                 else:
                     print(f" [Warn] Skipping empty test depth: {d}")
                     
        else:
             print("[Warn] Local RuleTaker path not found, falling back to HF (ignoring depth args)...")
             train_ds = NeSyRuleTakerDataset("train", tokenizer=tokenizer, augment=True)
             test_ds = NeSyRuleTakerDataset("test", tokenizer=tokenizer, augment=True, augment_seed=999)
             test_loaders['default'] = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=NeSyCollator(tokenizer), num_workers=args.num_workers)
     
    # Filter Nones (if NeSyCLUTRRDataset returns Nones)
    # NeSyRuleTakerDataset doesn't return Nones in current impl
    if hasattr(train_ds, 'data') and isinstance(train_ds.data, list):
         train_ds.data = [d for d in train_ds.data if d is not None] 

    
    collator = NeSyCollator(tokenizer)

    if args.strategy == "ddp":
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        # Eval只在主进程做即可（避免重复跑一遍 test）
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=train_sampler,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True, # Optimized
        )
        # DDP mode: Test loaders logic is handled above, but here we don't need sampler for eval
        # Just use what we created in test_loaders dict
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False
        )
        # Test loaders already created above
    
    # 2. Model
    model = NeSyRoBERTa(
        device, 
        tokenizer, 
        model_type=args.model_type,
        num_labels=num_labels,
        num_relations=num_latent_relations
    ).to(device)

    if args.strategy == "dp":
        if not torch.cuda.is_available():
            raise RuntimeError("dp 需要 CUDA 可用")
        n = torch.cuda.device_count()
        if n <= 1:
            if _is_main_process():
                print("[Warn] dp 但当前可见 GPU <= 1，将退化为单卡")
        else:
            model = nn.DataParallel(model, device_ids=list(range(n)))
    elif args.strategy == "ddp":
        # DDP recommended for multi-GPU sync training
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # find_unused_parameters=True prevents crash when some layers (e.g. rel_mlp) 
            # are not used in a specific dataset (RuleTaker has no rel labels)
            find_unused_parameters=True, 
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda') # AMP Scaler
    
    def recursive_to_device(obj, device, non_blocking=False):
        if torch.is_tensor(obj):
            return obj.to(device, non_blocking=non_blocking)
        elif isinstance(obj, dict):
            return {k: recursive_to_device(v, device, non_blocking=non_blocking) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_to_device(v, device, non_blocking=non_blocking) for v in obj]
        else:
            return obj

    # 3. Loop
    epochs = args.epochs 
    
    print("Starting Training (Scheme A + B) with AMP...")
    for epoch in range(epochs):
        if args.strategy == "ddp":
            # type: ignore[name-defined]
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0
        accum_aux = 0
        accum_cons = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for batch in pbar:
            if batch is None: continue

            # Move to device with non_blocking=True (since pinned memory)
            batch = recursive_to_device(batch, device, non_blocking=True)

            # Avoid DataParallel scatter issues with non-tensor fields
            batch_for_model = dict(batch)
            if "raw_batch" in batch_for_model:
                batch_for_model.pop("raw_batch")
            
            optimizer.zero_grad()
            
            # AMP Context
            with torch.amp.autocast('cuda'):
                # lambda1 (NextHop) = 1.0, lambda_cons (Consistency) = 5.0，lambda_rel=1.0
                out = model(batch_for_model, lambda1=1.0, lambda_cons=5.0, lambda_rel=1.0)
                loss = out['loss']
            
            # Scaler Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
        if _is_main_process():
            print(
                f"Epoch {epoch+1} Done. Loss: {avg_loss:.4f} "
                f"(Aux: {accum_aux/len(train_loader):.4f}, Cons: {accum_cons/len(train_loader):.4f})"
            )
        
        # 4. Evaluation
        if _is_main_process():
            if args.dataset == "ruletaker":
                evaluate_ruletaker_suite(model, test_loaders, device)
            else:
                print(f"--> Evaluating Robustness Epoch {epoch+1}...")
                metrics = evaluate_robustness(model, test_loaders['default'], device)
                
                print(f"  Overall Acc (Base):   {metrics['overall']:.4f}")
                print(f"  Renamed Acc (Mod):    {metrics['renamed']:.4f}")
                print(f"  Consistency:          {metrics['consistency']:.4f}")
                print(f"  Consistent & Correct: {metrics['consistent_and_correct']:.4f}")
                print(f"  Short Hop (2-3):      {metrics['short_hop']:.4f}")
                print(f"  Long Hop (>=6):       {metrics['long_hop']:.4f}")

    if args.strategy == "ddp" and _is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def evaluate_ruletaker(model, loader, device):
    model.eval()
    base_model = _unwrap_model(model)
    
    total = 0
    correct = 0
    
    def recursive_to_device(obj, device):
        if torch.is_tensor(obj):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: recursive_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_to_device(v, device) for v in obj]
        else:
            return obj

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval RuleTaker", leave=False):
            if batch is None:
                continue
            
            # 手动移到 device (因为 collate 可能返回 cpu)
            batch = recursive_to_device(batch, device)
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            # 只走 Forward / CLS Logits (Skip NeSy path to avoid empty entity crashes)
            out = base_model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = out.last_hidden_state
            cls_output = sequence_output[:, 0, :]
            logits = base_model.classifier(cls_output)
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    
    acc = correct / total if total > 0 else 0
    return {"overall": acc, "total": total}


def evaluate_ruletaker_suite(model, loaders_dict, device):
    print(f"--> Evaluating RuleTaker Suite...")
    results = {}
    total_samples = 0
    weighted_acc = 0.0
    
    for name, loader in loaders_dict.items():
        metrics = evaluate_ruletaker(model, loader, device)
        acc = metrics['overall']
        n = metrics['total']
        results[name] = acc
        print(f"  {name}: {acc:.4f} (n={n})")
        
        weighted_acc += acc * n
        total_samples += n
        
    macro = sum(results.values()) / len(results) if results else 0
    micro = weighted_acc / total_samples if total_samples > 0 else 0
    print(f"  [Summary] Macro Avg: {macro:.4f}, Micro Avg: {micro:.4f}")
    return {"overall": micro}


def evaluate_robustness(model, loader, device):
    model.eval()
    base_model = _unwrap_model(model)
    
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
    
    def recursive_to_device(obj, device):
        if torch.is_tensor(obj):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: recursive_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_to_device(v, device) for v in obj]
        else:
            return obj

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            # Move to device manually since collator returns CPU tensors now
            batch = recursive_to_device(batch, device)
            
            y_target = batch['labels']
            hops = batch['hops']

            logits_base, _, _, _ = base_model.compute_logits(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                entity_spans=batch['entity_spans'],
                query_indices=batch['query_indices'],
            )
            preds_base = torch.argmax(logits_base, dim=1)

            logits_mod, _, _, _ = base_model.compute_logits(
                input_ids=batch['aug_input_ids'],
                attention_mask=batch['aug_attention_mask'],
                entity_spans=batch['aug_entity_spans'],
                query_indices=batch['query_indices'],
            )
            preds_mod = torch.argmax(logits_mod, dim=1)

            # Changed rates from raw_batch
            for b_idx, b in enumerate(batch['raw_batch']):
                if 'query' in b:
                    query_text = f"{b['query'][0]} and {b['query'][1]}"
                else:
                    query_text = b.get('query_text_raw', '')

                story_text = b['story']
                story_mod = b.get('aug_story', story_text)
                
                if 'aug_query' in b:
                    query_mod = f"{b['aug_query'][0]} and {b['aug_query'][1]}"
                else:
                    query_mod = b.get('aug_query_text_raw', query_text)

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
