import os
import csv
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import random

import sys
import os

# Add the current directory to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the model
from models.transformer_baseline import CLUTRRTransformerModel

class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.zeros(rank, original_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, rank))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        original_out = self.original_layer(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_out + lora_out

def apply_lora(model, rank, alpha, dropout):
    # Freeze the entire model first
    for param in model.parameters():
        param.requires_grad = False
        
    # Apply LoRA to Attention layers
    for name, module in model.named_modules():
        # Check if it's an attention module by looking for specific attributes
        # We target q_proj, k_proj, v_proj, out_proj in Vanilla and SoftUnif attention
        if hasattr(module, 'q_proj') and isinstance(module.q_proj, nn.Linear):
            module.q_proj = LoRALinear(module.q_proj, rank, alpha, dropout)
        if hasattr(module, 'k_proj') and isinstance(module.k_proj, nn.Linear):
            module.k_proj = LoRALinear(module.k_proj, rank, alpha, dropout)
        if hasattr(module, 'v_proj') and isinstance(module.v_proj, nn.Linear):
            module.v_proj = LoRALinear(module.v_proj, rank, alpha, dropout)
        if hasattr(module, 'out_proj') and isinstance(module.out_proj, nn.Linear):
            module.out_proj = LoRALinear(module.out_proj, rank, alpha, dropout)
            
    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"LoRA Applied. Trainable params: {trainable_params} / {all_params} ({100 * trainable_params / all_params:.2f}%)")
    
    return model

# Define relation map (copied from run_with_constraints.py)
relation_id_map = {
  'daughter': 0,
  'sister': 1,
  'son': 2,
  'aunt': 3,
  'father': 4,
  'husband': 5,
  'granddaughter': 6,
  'brother': 7,
  'nephew': 8,
  'mother': 9,
  'uncle': 10,
  'grandfather': 11,
  'wife': 12,
  'grandmother': 13,
  'niece': 14,
  'grandson': 15,
  'son-in-law': 16,
  'father-in-law': 17,
  'daughter-in-law': 18,
  'mother-in-law': 19,
  'nothing': 20,
}

class CLUTRRDataset:
  def __init__(self, root, dataset, split, data_percentage):
    self.dataset_dir = os.path.join(root, f"{dataset}/")
    # Filter files based on split
    self.file_names = [os.path.join(self.dataset_dir, d) for d in os.listdir(self.dataset_dir) if f"_{split}.csv" in d]
    self.data = [row for f in self.file_names for row in list(csv.reader(open(f)))[1:]]
    self.data_num = math.floor(len(self.data) * data_percentage / 100)
    self.data = self.data[:self.data_num]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    # Context is a list of sentences
    context = [s.strip().lower() for s in self.data[i][2].split(".") if s.strip() != ""]

    # Query is of type (sub, obj)
    query_sub_obj = eval(self.data[i][3])
    query = (query_sub_obj[0].lower(), query_sub_obj[1].lower())

    # Answer is one of 20 classes
    answer = self.data[i][5]
    
    # Also return the file name or k (length) for analysis if possible
    # The file name is not easily accessible per row here without storing it.
    # But we can infer k from the file name if we structured it differently.
    # For now, let's just return what the original did.
    return ((context, query), answer)

  @staticmethod
  def collate_fn(batch):
    queries = [query for ((_, query), _) in batch]
    contexts = [fact for ((context, _), _) in batch for fact in context]
    context_lens = [len(context) for ((context, _), _) in batch]
    context_splits = [(sum(context_lens[:i]), sum(context_lens[:i + 1])) for i in range(len(context_lens))]
    answers = torch.stack([torch.tensor(relation_id_map[answer]) for (_, answer) in batch])
    return ((contexts, queries, context_splits), answers)

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.num_relations = len(relation_id_map)

    print(f"Using device: {device}")
    print(f"Attention Type: {args.attention_type}")

    # Load Data
    # Use absolute path relative to this script to be safe
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(script_dir, "../data/clutrr")
    
    train_dataset = CLUTRRDataset(root, args.dataset, "train", args.data_percentage)
    test_dataset = CLUTRRDataset(root, args.dataset, "test", 100)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        collate_fn=CLUTRRDataset.collate_fn, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        collate_fn=CLUTRRDataset.collate_fn, 
        shuffle=False
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Initialize Model
    model = CLUTRRTransformerModel(args).to(device)
    
    if args.use_lora:
        print(f"Applying LoRA with rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        model = apply_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for (contexts, queries, context_splits), answers in pbar:
            answers = answers.to(device)
            
            optimizer.zero_grad()
            logits = model(contexts, context_splits, queries)
            
            loss = criterion(logits, answers)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == answers).sum().item()
            total += answers.size(0)
            
            pbar.set_postfix({'loss': total_loss / (total/args.batch_size + 1e-9), 'acc': correct / total})

        train_acc = correct / total
        print(f"Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f} Train Acc: {train_acc:.4f}")

        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for (contexts, queries, context_splits), answers in test_loader:
                answers = answers.to(device)
                logits = model(contexts, context_splits, queries)
                preds = torch.argmax(logits, dim=1)
                test_correct += (preds == answers).sum().item()
                test_total += answers.size(0)
        
        test_acc = test_correct / test_total
        print(f"Epoch {epoch+1} Test Acc: {test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"clutrr_transformer_{args.attention_type}_best.pth")
            print("Saved best model.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data_db9b8f04", help="Dataset folder name")
    parser.add_argument("--data_percentage", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    
    # Model Config
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp_layers", type=int, default=1)
    
    # Attention Type
    parser.add_argument("--attention_type", type=str, default="vanilla", choices=["vanilla", "softunif"])
    
    # LoRA Config
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")

    args = parser.parse_args()
    train(args)
