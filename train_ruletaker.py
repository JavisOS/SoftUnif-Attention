
import sys
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
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
from datasets import load_dataset

# Path fixes
sys.path.append(os.getcwd())

# Setup DDP (Reused from train_nesy.py)
def _setup_ddp() -> tuple[int, int, int]:
    if not ("RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ):
        # Default to single process if not launched with torchrun
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def _is_main_process():
    rank, _, _ = _setup_ddp()
    return rank == 0

# ==========================================
# 1. New RuleTaker Dataset Wrapper
# ==========================================
class RuleTakerDataset(Dataset):
    def __init__(self, split="train", limit=None):
        import glob
        import json
        
        self.data = []
        
        # 1. Try Local Load
        # Pattern: data/rule-reasoning-dataset-V2020.2.5.0/original/depth-*/{split}.jsonl
        # or simplified path
        local_dir = "data/rule-reasoning-dataset-V2020.2.5.0/original"
        files = []
        if os.path.exists(local_dir):
            if _is_main_process(): print(f"Looking for local RuleTaker in {local_dir}...")
            # Look in depth subfolders
            files = sorted(glob.glob(os.path.join(local_dir, "depth-*", f"{split}.jsonl")))
            if not files:
                 # Try direct
                 files = sorted(glob.glob(os.path.join(local_dir, f"{split}.jsonl")))
            
        if files:
            if _is_main_process(): print(f"Found {len(files)} local files. Loading...")
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
        else:
            # 2. Fallback to HF
            if _is_main_process(): print(f"Local files not found, downloading from HuggingFace...")
            hf_split = "validation" if split == "dev" else split
            ds = load_dataset("tasksource/ruletaker", split=hf_split)
            
            # Convert HF dataset to list of dicts for unified interface
            for item in ds:
                raw_label = item['label']
                if isinstance(raw_label, str):
                    label = 1 if raw_label == 'entailment' else 0
                else:
                    label = int(raw_label)
                    
                self.data.append({
                    'context': item.get('context', item.get('text', '')),
                    'question': item.get('question', ''),
                    'label': label
                })

        if limit and limit < len(self.data):
             self.data = self.data[:limit]

        if len(self.data) > 0 and _is_main_process():
            print(f"[{split}] Loaded {len(self.data)} samples.")
            print(f"[{split}] Example: {self.data[0]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class RuleTakerCollator:
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        
    def __call__(self, batch):
        if not batch: return None
        
        contexts = [b['context'] for b in batch]
        questions = [b['question'] for b in batch]
        labels = torch.tensor([b['label'] for b in batch], dtype=torch.long).to(self.device)
        
        # Tokenize (Context + Question)
        enc = self.tokenizer(
            contexts,
            questions,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True,
            max_length=512 
        ).to(self.device)
        
        # Diagnostic print (once)
        if not hasattr(self, 'debug_printed'):
             print("\n[DEBUG Collator] First batch input example (decoded):")
             print(self.tokenizer.decode(enc.input_ids[0]))
             print(f"[DEBUG Collator] Label: {labels[0].item()}")
             self.debug_printed = True
        
        return {
            'input_ids': enc.input_ids,
            'attention_mask': enc.attention_mask,
            'labels': labels
        }

# ==========================================
# 2. Simplified Model (CLS only)
# ==========================================
class SimpleNeSyRoBERTa(nn.Module):
    def __init__(self, device, tokenizer, model_type="roberta", num_labels=2):
        super().__init__()
        self.device = device
        self.model_type = model_type.lower()
        
        # Model Factory (Same as train_nesy.py)
        if "roberta" in self.model_type:
             self.encoder = AutoModel.from_pretrained("roberta-base" if "base" in model_type or model_type == "roberta" else "roberta-large")
        elif "deberta-v3" in self.model_type:
             self.encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
        else:
             self.encoder = AutoModel.from_pretrained("bert-base-uncased") # fallback

        self.hidden_size = self.encoder.config.hidden_size
        
        # Simple Classifier Head (CLS)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
    def forward(self, batch_data):
        input_ids = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        labels = batch_data['labels']
        
        # Encoder
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = out.last_hidden_state
        cls_output = sequence_output[:, 0, :]
        
        # Debug: Print logits stats once
        if not hasattr(self, 'debug_printed'):
             with torch.no_grad():
                 logits_test = self.classifier(cls_output)
                 print(f"\n[DEBUG Model] Logits mean: {logits_test.mean().item():.3f}, std: {logits_test.std().item():.3f}")
             self.debug_printed = True

        # Logs
        logits = self.classifier(cls_output)
        
        # Loss
        loss = nn.functional.cross_entropy(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }

# ==========================================
# 3. Training Loop (Adapted)
# ==========================================
def run_training():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="roberta")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dataset", type=str, default="ruletaker")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # GPU Setup
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base") # Default
    if "deberta" in args.model_type:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    
    # Dataset
    print(f"Loading {args.dataset}...")
    # NOTE: RuleTaker has 5 datasets, using "small" by default if not specified in load_dataset(path, name)
    # But user said load_dataset("tasksource/ruletaker"). We'll try default.
    train_ds = RuleTakerDataset(split="train", limit=args.limit)
    test_ds = RuleTakerDataset(split="test", limit=args.limit) # or validation

    collator = RuleTakerCollator(tokenizer, device)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    
    # Model
    model = SimpleNeSyRoBERTa(device, tokenizer, model_type=args.model_type, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for batch in pbar:
            optimizer.zero_grad()
            out = model(batch)
            loss = out['loss']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Acc
            preds = torch.argmax(out['logits'], dim=1)
            correct += (preds == batch['labels']).sum().item()
            total += len(batch['labels'])
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct/total:.3f}"})
            
        # Eval
        print(f"Evaluating Epoch {epoch+1}...")
        model.eval()
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch)
                preds = torch.argmax(out['logits'], dim=1)
                eval_correct += (preds == batch['labels']).sum().item()
                eval_total += len(batch['labels'])
        
        print(f"Dev/Test Acc: {eval_correct/eval_total:.4f}")

if __name__ == "__main__":
    run_training()
