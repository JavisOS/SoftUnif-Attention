import sys
import os
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    RobertaTokenizerFast, RobertaModel,
    DebertaTokenizerFast, DebertaModel,
    DebertaV2TokenizerFast, DebertaV2Model,
    BertTokenizerFast, BertModel
)

# Fix paths to allow imports from root
sys.path.append(os.getcwd())

try:
    from baseline_roberta_analysis import relation_id_map, set_seed
except ImportError:
    pass

from clutrr.nesy_utils import parse_graph_and_path

# ==========================================
# 1. Dataset & Collator
# ==========================================
class BaselineDataset(Dataset):
    def __init__(self, root, dataset, split, data_percentage=100):
        self.dataset_dir = os.path.join(root, f"{dataset}/")
        
        if os.path.exists(self.dataset_dir):
            self.file_names = [os.path.join(self.dataset_dir, d) for d in os.listdir(self.dataset_dir) if f"_{split}.csv" in d]
            # Use basic csv reader
            import csv
            self.data = []
            for f in self.file_names:
                with open(f, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader) # skip header
                    self.data.extend(list(reader))
        else:
            self.data = []
            print(f"Warning: Directory {self.dataset_dir} not found.")

        # Subset
        import math
        self.data_num = math.floor(len(self.data) * data_percentage / 100)
        self.data = self.data[:self.data_num]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data[i]
        
        # Raw text
        story = row[2]
        query_tuple = eval(row[3]) # ('sub', 'obj')
        target_rel = row[5]
        
        # Calculate hops for evaluation (requires graph parsing)
        # If parsing fails, default to -1
        try:
            graph_info = parse_graph_and_path(row)
            if graph_info:
                # Path includes start node, so hops = len - 1
                hops = len(graph_info['path_node_indices']) - 1
            else:
                hops = -1
        except:
            hops = -1

        target_id = relation_id_map.get(target_rel, relation_id_map['nothing'])

        return {
            'story': story,
            'query': f"{query_tuple[0]} and {query_tuple[1]}",
            'target_id': target_id,
            'hops': hops
        }

class BaselineCollator:
    def __init__(self, tokenizer, device, model_type="roberta"):
        self.tokenizer = tokenizer
        self.device = device
        self.model_type = model_type

    def __call__(self, batch):
        stories = [b['story'] for b in batch]
        queries = [b['query'] for b in batch]
        targets = torch.tensor([b['target_id'] for b in batch], dtype=torch.long).to(self.device)
        hops = torch.tensor([b['hops'] for b in batch], dtype=torch.long).to(self.device)

        # Tokenization: 
        # For BERT/RoBERTa/DeBERTa, standard is [CLS] Sequence A [SEP] Sequence B [SEP]
        # Transformers library handles this via `text` and `text_pair` arguments.
        enc = self.tokenizer(
            text=stories,
            text_pair=queries,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(self.device)

        return {
            'input_ids': enc.input_ids,
            'attention_mask': enc.attention_mask,
            'labels': targets,
            'hops': hops
        }

# ==========================================
# 2. Model
# ==========================================
class BaselineModel(nn.Module):
    def __init__(self, model_type, num_labels):
        super().__init__()
        self.model_type = model_type
        
        if model_type == 'bert':
            self.encoder = BertModel.from_pretrained("bert-base-uncased")
        elif model_type == 'roberta':
            self.encoder = RobertaModel.from_pretrained("roberta-base")
        elif model_type == 'roberta-large':
            self.encoder = RobertaModel.from_pretrained("roberta-large")
        elif model_type == 'deberta':
            self.encoder = DebertaModel.from_pretrained("microsoft/deberta-base")
        elif model_type == 'deberta-v3':
            self.encoder = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
        elif model_type == 'deberta-v3-large':
            self.encoder = DebertaV2Model.from_pretrained("microsoft/deberta-v3-large")
        elif model_type == 'modernbert':
            try:
                self.encoder = AutoModel.from_pretrained("answerdotai/ModernBERT-base", trust_remote_code=True)
            except TypeError:
                self.encoder = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pooling strategy depends on model
        # BERT: pooler_output (CLS + Dense + Tanh)
        # RoBERTa: usually just take CLS (last_hidden_state[:,0,:])
        # DeBERTa: last_hidden_state[:,0,:]
        
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
             pooled = out.pooler_output
        else:
             pooled = out.last_hidden_state[:, 0, :]
             
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            
        return {'loss': loss, 'logits': logits}

# ==========================================
# 3. Training & Eval
# ==========================================
def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0
    
    # Hop stats
    short_hops = [2, 3] # k=2,3
    long_hops = range(4, 20) # k>=4 per some papers, or >=6 user requested
    
    hop_correct = {}
    hop_total = {}

    with torch.no_grad():
        for batch in loader:
            labels = batch['labels']
            hops = batch['hops']
            
            out = model(batch['input_ids'], batch['attention_mask'])
            preds = torch.argmax(out['logits'], dim=1)
            
            total += len(labels)
            correct += (preds == labels).sum().item()
            
            # Hop analysis
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            hops_np = hops.cpu().numpy()
            
            for h, p, t in zip(hops_np, preds_np, labels_np):
                if h == -1: continue # invalid
                if h not in hop_total:
                    hop_total[h] = 0
                    hop_correct[h] = 0
                hop_total[h] += 1
                if p == t:
                    hop_correct[h] += 1
                    
    overall_acc = correct / total if total > 0 else 0.0
    
    # Short Hop (2-3)
    s_corr = sum(hop_correct.get(h, 0) for h in [2, 3])
    s_tot = sum(hop_total.get(h, 0) for h in [2, 3])
    short_acc = s_corr / s_tot if s_tot > 0 else 0.0
    
    # Long Hop (>=6) (User requested >=6, though CLUTRR gen is usually up to ~10)
    l_corr = sum(hop_correct.get(h, 0) for h in range(6, 20))
    l_tot = sum(hop_total.get(h, 0) for h in range(6, 20))
    long_acc = l_corr / l_tot if l_tot > 0 else 0.0

    return overall_acc, short_acc, long_acc


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="roberta", 
                        choices=["bert", "roberta", "roberta-large", "deberta", "deberta-v3", "deberta-v3-large", "modernbert"], 
                        help="Model type")
    parser.add_argument("--root", type=str, default="data", help="Data root directory")
    parser.add_argument("--dataset", type=str, default="data_089907f8", help="Dataset folder name")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_type}")

    # Tokenizer
    if args.model_type == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    elif args.model_type == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    elif args.model_type == 'roberta-large':
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
    elif args.model_type == 'deberta':
        tokenizer = DebertaTokenizerFast.from_pretrained("microsoft/deberta-base")
    elif args.model_type == 'deberta-v3':
        tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")
    elif args.model_type == 'deberta-v3-large':
        tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-large")
    elif args.model_type == 'modernbert':
        try:
            tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", use_fast=True, trust_remote_code=True)
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", use_fast=True)
    
    # Data
    print("Loading Data...")
    train_ds = BaselineDataset(args.root, args.dataset, "train")
    test_ds = BaselineDataset(args.root, args.dataset, "test")
    
    collator = BaselineCollator(tokenizer, device, args.model_type)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collator)
    
    # Model
    model = BaselineModel(args.model_type, num_labels=len(relation_id_map)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Train
    print("Starting Training...")
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for batch in pbar:
            optimizer.zero_grad()
            out = model(batch['input_ids'], batch['attention_mask'], batch['labels'])
            loss = out['loss']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Eval
        overall, short_h, long_h = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
        print(f"  Overall Acc (Base): {overall:.4f}")
        print(f"  Short Hop (2-3):    {short_h:.4f}")
        print(f"  Long Hop (>=6):     {long_h:.4f}")
        
if __name__ == "__main__":
    run()
