import os
import csv
import re
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm

# ==========================================
# 1. Reproducibility
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

# ==========================================
# 2. Data Loading & Constants
# ==========================================
relation_id_map = {
  'daughter': 0, 'sister': 1, 'son': 2, 'aunt': 3, 'father': 4, 'husband': 5,
  'granddaughter': 6, 'brother': 7, 'nephew': 8, 'mother': 9, 'uncle': 10,
  'grandfather': 11, 'wife': 12, 'grandmother': 13, 'niece': 14, 'grandson': 15,
  'son-in-law': 16, 'father-in-law': 17, 'daughter-in-law': 18, 'mother-in-law': 19,
  'nothing': 20,
}
id_to_relation = {v: k for k, v in relation_id_map.items()}

class CLUTRRDataset(Dataset):
    def __init__(self, root, dataset, split, data_percentage):
        self.dataset_dir = os.path.join(root, f"{dataset}/")
        # Try to find files, if directory doesn't exist handles gracefully or let it fail
        if os.path.exists(self.dataset_dir):
            self.file_names = [os.path.join(self.dataset_dir, d) for d in os.listdir(self.dataset_dir) if f"_{split}.csv" in d]
            self.data = [row for f in self.file_names for row in list(csv.reader(open(f)))[1:]]
        else:
            self.data = []
            
        self.data_num = math.floor(len(self.data) * data_percentage / 100)
        self.data = self.data[:self.data_num]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # Context is a list of sentences
        context_str = self.data[i][2]
        context = [s.strip().lower() for s in context_str.split(".") if s.strip() != ""]

        # Query is of type (sub, obj)
        query_sub_obj = eval(self.data[i][3])
        query = (query_sub_obj[0].lower(), query_sub_obj[1].lower())

        # Answer
        answer = self.data[i][5]
        
        # Hops (Edge count)
        try:
            edges = eval(self.data[i][11])
            hops = len(edges)
        except:
            hops = 0
            
        return ((context, query), answer, hops, context_str)

    @staticmethod
    def collate_fn(batch):
        queries = [query for ((_, query), _, _, _) in batch]
        context_strs = [ctx_str for ((_, _), _, _, ctx_str) in batch]
        contexts = [fact for ((context, _), _, _, _) in batch for fact in context]
        context_lens = [len(context) for ((context, _), _, _, _) in batch]
        context_splits = [(sum(context_lens[:i]), sum(context_lens[:i + 1])) for i in range(len(context_lens))]
        answers = torch.stack([torch.tensor(relation_id_map[answer]) for (_, answer, _, _) in batch])
        hops = torch.tensor([h for (_, _, h, _) in batch])
        
        return ((contexts, queries, context_splits, context_strs), answers, hops)

def clutrr_loader(root, dataset, batch_size, training_data_percentage):
    train_dataset = CLUTRRDataset(root, dataset, "train", training_data_percentage)
    # Using 0 workers to avoid potential multiprocessing issues in this script for now
    train_loader = DataLoader(
        train_dataset, 
        batch_size, 
        collate_fn=CLUTRRDataset.collate_fn, 
        shuffle=True, 
        num_workers=0 
    )
    
    test_dataset = CLUTRRDataset(root, dataset, "test", 100)
    test_loader = DataLoader(
        test_dataset, 
        batch_size, 
        collate_fn=CLUTRRDataset.collate_fn, 
        shuffle=False, # Don't shuffle test for consistent visualization if needed
        num_workers=0 
    )
    return train_loader, test_loader

# ==========================================
# 3. Model Architecture (Standard Baseline)
# ==========================================
class CLUTRRBaseline(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        # output_attentions=True for visualization
        self.roberta = RobertaModel.from_pretrained("roberta-base", output_attentions=True)
        self.hidden_size = self.roberta.config.hidden_size
        
        # Simple classifier head on [CLS]
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, len(relation_id_map))
        )

    def forward(self, x):
        (contexts, queries, context_splits, _) = x
        
        # We need to construct the input string for RoBERTa.
        # Format: [CLS] context [SEP] query [SEP]
        # Since clutrr_loader gives split contexts, we reconstruct them.
        
        batch_input_texts = []
        for i, (start, end) in enumerate(context_splits):
            # Reconstruct story
            story_sents = contexts[start:end]
            story_text = ". ".join(story_sents)
            
            # Construct Query: "What is relationship between sub and obj?"
            # Or simpler: "sub and obj"
            sub, obj = queries[i]
            query_text = f"{sub} and {obj}"
            
            # RoBERTa format: <s> story </s> </s> query </s>
            # The tokenizer handles the special tokens if we pass pairs
            batch_input_texts.append((story_text, query_text))
            
        # Tokenize
        inputs = self.tokenizer(
            batch_input_texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.roberta(**inputs)
        
        # Use [CLS] token (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier(cls_embedding)
        
        return {
            "logits": logits,
            "attentions": outputs.attentions, # Tuple of (B, H, Seq, Seq)
            "input_ids": inputs.input_ids
        }

# ==========================================
# 4. Evaluation & Visualization
# ==========================================
def visualize_attention(model, x_data, y_target, hop_count, run_id, epoch):
    """
    Visualizes attention for the LAST layer of RoBERTa for the FIRST element in the batch.
    """
    model.eval()
    with torch.no_grad():
        out = model(x_data)
    
    # Get attentions from last layer: (Batch, NumHeads, Seq, Seq)
    attentions = out["attentions"][-1] 
    input_ids = out["input_ids"]
    
    # Pick the first example in the batch
    idx = 0
    att = attentions[idx] # (NumHeads, Seq, Seq)
    # Average over heads
    att_avg = att.mean(dim=0).cpu().numpy() # (Seq, Seq)
    
    tokens = model.tokenizer.convert_ids_to_tokens(input_ids[idx])
    
    # Filter long sequences for visualization clarity
    if len(tokens) > 100:
        tokens = tokens[:100]
        att_avg = att_avg[:100, :100]
        
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(att_avg, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    plt.title(f"RoBERTa Last Layer Attention (Avg Head) - Hop {hop_count[idx].item()}")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    # Save with epoch info
    ensure_dir = "plots"
    if not os.path.exists(ensure_dir): os.makedirs(ensure_dir)
    filename = f"{ensure_dir}/attention_run{run_id}_ep{epoch}_hop{hop_count[idx].item()}.png"
    plt.savefig(filename)
    plt.close()
    # print(f"Saved attention map to {filename}")

def evaluate(model, test_loader, device, run_id=0, epoch=0, verbose=True):
    model.eval()
    correct = 0
    total = 0
    
    # Grouped accuracy
    correct_by_hops = {}
    total_by_hops = {}
    
    visualized_hops = set()
    visualize_targets = [2, 3, 4, 6, 7, 9, 10] # Try to visualize these lengths
    
    with torch.no_grad():
        for batch in test_loader: # Remove tqdm for per-epoch eval to reduce clutter
            (x_data, y_target, hops) = batch
            
            y_target = y_target.to(device)
            out = model(x_data)
            logits = out["logits"]
            preds = torch.argmax(logits, dim=1)
            
            # Global Stats
            correct += (preds == y_target).sum().item()
            total += len(y_target)
            
            # Hops Stats
            for i in range(len(y_target)):
                h = hops[i].item()
                if h not in total_by_hops:
                    total_by_hops[h] = 0
                    correct_by_hops[h] = 0
                total_by_hops[h] += 1
                if preds[i] == y_target[i]:
                    correct_by_hops[h] += 1
                    
                # Visualization Trigger: Only visualize on specific epochs (e.g., last one or specific intervals) 
                # or if it's the Best Epoch (controlled by caller usually, but here we just do it sparsely)
                # To avoid slowing down training too much, we only visualize periodically or at the end.
                # Let's visualize ONLY if verbose=True (which we'll set for the last epoch or new best)
                if verbose and h in visualize_targets and h not in visualized_hops:
                    if i == 0: 
                         visualize_attention(model, x_data, y_target, hops, run_id, epoch)
                         visualized_hops.add(h)

    acc = correct / total
    
    if verbose:
        print(f"\n[Epoch {epoch}] Global Accuracy: {acc:.4f}")
        print("Hop | Accuracy | Count")
        print("----|----------|------")
        for h in sorted(total_by_hops.keys()):
            h_acc = correct_by_hops[h] / total_by_hops[h]
            print(f"{h:3d} | {h_acc:.4f}   | {total_by_hops[h]}")
        print("----------------------")
        
    return acc

# ==========================================
# 5. Training Loop
# ==========================================
def train(args, run_id):
    set_seed(args.seed + run_id)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Correct path resolution assuming script is in project root
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/clutrr")
    
    train_loader, test_loader = clutrr_loader(data_root, args.dataset, args.batch_size, 100)
    
    model = CLUTRRBaseline(device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting training run {run_id}...")
    
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            (x_data, y_target, hops) = batch
            y_target = y_target.to(device)
            
            optimizer.zero_grad()
            out = model(x_data)
            loss = criterion(out["logits"], y_target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        
        # Validation every epoch
        # We only print detailed table (verbose=True) if we hit a new best record to reduce log spam,
        # OR if it's the last epoch.
        is_last = (epoch == args.epochs)
        
        # Calculate accuracy (verbose=False initially to get the score first)
        val_acc = evaluate(model, test_loader, device, run_id, epoch, verbose=False)
        
        # Check if best
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            print(f"Epoch {epoch}: Loss {avg_loss:.4f} | Val Acc {val_acc:.4f} * NEW BEST *")
            # Re-run evaluate with verbose=True to print the table and generate plots for the best model
            evaluate(model, test_loader, device, run_id, epoch, verbose=True)
            # Optional: Save model
            torch.save(model.state_dict(), f"best_model_run{run_id}.pth")
        else:
            print(f"Epoch {epoch}: Loss {avg_loss:.4f} | Val Acc {val_acc:.4f}")

    print(f"\nTraining run {run_id} finished.")
    print(f"Best Accuracy: {best_acc:.4f} at Epoch {best_epoch}")
    return best_acc

# ==========================================
# 6. Main
# ==========================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data_089907f8")
    parser.add_argument("--epochs", type=int, default=10) # 10 epochs is usually enough for RoBERTa on this size
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=1) # Default 1 for speed, user can increase
    args = parser.parse_args()

    accuracies = []
    for i in range(args.n_runs):
        acc = train(args, i)
        accuracies.append(acc)
        
    print(f"\n=================================")
    print(f"Final Results over {args.n_runs} runs:")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
    print(f"Std Dev: {np.std(accuracies):.4f}")
    print(f"=================================")
