import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import random
import transformers
import sys

# Add the parent directory to sys.path to allow imports from clutrr package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import from new structure
from clutrr.data.dataset import CLUTRRDataset, relation_id_map
from clutrr.models.model_roberta import CLUTRRRobertaDropIn, CLUTRRRobertaWrapper

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.num_relations = len(relation_id_map)

    print(f"Using device: {device}")
    print(f"Model Type: {args.model_type}")

    # Load Data
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
    # We use CLUTRRRobertaDropIn as the default for this script as per user request
    # But if the user wants to use the wrapper, we could add a flag.
    # The user instructions said: "Refactored from run_roberta_dropin.py"
    # So we use CLUTRRRobertaDropIn.
    model = CLUTRRRobertaDropIn(args).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
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
            save_dir = os.path.join(script_dir, "../model/clutrr")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"clutrr_{args.model_type}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data_db9b8f04", help="Dataset folder name")
    parser.add_argument("--data_percentage", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5) # Low LR for fine-tuning
    # parser.add_argument("--weight_decay", type=float, default=1e-2)
    
    # Model Config
    parser.add_argument("--model_type", type=str, default="roberta_vanilla", choices=["roberta_vanilla", "roberta_softunif"])
    parser.add_argument("--mlp_layers", type=int, default=1)
    
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Seeding
    if args.seed is not None:
      torch.manual_seed(args.seed)
      torch.cuda.manual_seed_all(args.seed)
      random.seed(args.seed)
      np.random.seed(args.seed)
      transformers.set_seed(args.seed)

    train(args)
