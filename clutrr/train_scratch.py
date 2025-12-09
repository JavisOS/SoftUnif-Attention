import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import random
import sys

# Add the parent directory to sys.path to allow imports from clutrr package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import from new structure
from clutrr.data.dataset import CLUTRRDataset, relation_id_map
from clutrr.models.model_scratch import CLUTRRTransformerModel

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
        for (input_ids, attention_mask, type_labels, unify_mat, sub_indices, obj_indices), answers in pbar:
            answers = answers.to(device)
            type_labels = type_labels.to(device)
            unify_mat = unify_mat.to(device)
            
            optimizer.zero_grad()
            task_logits, all_type_logits, all_unify_logits, attention_mask = model(input_ids, attention_mask, sub_indices, obj_indices)
            
            # Task Loss
            task_loss = criterion(task_logits, answers)
            
            # Auxiliary Losses
            total_type_loss = 0
            total_unify_loss = 0
            
            if args.attention_type == "softunif":
                # Type Loss
                for layer_logits in all_type_logits:
                    # layer_logits: [B, L, 5] -> Flatten: [B*L, 5]
                    # type_labels: [B, L] -> Flatten: [B*L]
                    # Note: We treat padding (0) as NOISE (0), which is acceptable.
                    total_type_loss += nn.CrossEntropyLoss()(layer_logits.view(-1, 5), type_labels.view(-1))
                
                # Unify Loss
                # Broadcast unify_mat to [B, 1, L, L] to match heads [B, H, L, L]
                unify_target = unify_mat.unsqueeze(1) 
                
                # Create Mask for Unification Loss
                # attention_mask: [B, L] -> [B, 1, L, 1] * [B, 1, 1, L] -> [B, 1, L, L]
                active_loss_mask = attention_mask.unsqueeze(1).unsqueeze(2) * attention_mask.unsqueeze(1).unsqueeze(3)
                
                for layer_logits in all_unify_logits:
                    # layer_logits: [B, H, L, L]
                    H = layer_logits.size(1)
                    target = unify_target.expand(-1, H, -1, -1)
                    
                    # Calculate BCE Loss with Masking
                    loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                    loss = loss_fct(layer_logits, target)
                    
                    # Expand mask to heads
                    mask = active_loss_mask.expand(-1, H, -1, -1)
                    
                    masked_loss = (loss * mask).sum() / (mask.sum() + 1e-9)
                    total_unify_loss += masked_loss

            loss = task_loss + args.lambda_type * total_type_loss + args.lambda_unify * total_unify_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(task_logits, dim=1)
            correct += (preds == answers).sum().item()
            total += answers.size(0)
            
            pbar.set_postfix({
                'loss': total_loss / (total/args.batch_size + 1e-9), 
                'acc': correct / total,
                'task': f"{task_loss.item():.2f}",
                'type': f"{total_type_loss.item() if isinstance(total_type_loss, torch.Tensor) else 0:.2f}",
                'unif': f"{total_unify_loss.item() if isinstance(total_unify_loss, torch.Tensor) else 0:.2f}"
            })

        train_acc = correct / total
        print(f"Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f} Train Acc: {train_acc:.4f}")

        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for (input_ids, attention_mask, _, _, sub_indices, obj_indices), answers in test_loader:
                answers = answers.to(device)
                task_logits, _, _, _ = model(input_ids, attention_mask, sub_indices, obj_indices)
                preds = torch.argmax(task_logits, dim=1)
                test_correct += (preds == answers).sum().item()
                test_total += answers.size(0)
        
        test_acc = test_correct / test_total
        print(f"Epoch {epoch+1} Test Acc: {test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            save_dir = os.path.join(script_dir, "../model/clutrr")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"clutrr_transformer_{args.attention_type}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data_db9b8f04", help="Dataset folder name")
    parser.add_argument("--data_percentage", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
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
    
    # Auxiliary Loss Weights
    parser.add_argument("--lambda_type", type=float, default=0.5, help="Weight for Type Classification Loss")
    parser.add_argument("--lambda_unify", type=float, default=0.1, help="Weight for Unification Loss")
    
    args = parser.parse_args()
    train(args)
