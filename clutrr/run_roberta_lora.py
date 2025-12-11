import os
import csv
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig

# Define relation map
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

def apply_lora_to_roberta(model, rank, alpha, dropout):
    # Freeze the entire model first
    for param in model.parameters():
        param.requires_grad = False
        
    # Apply LoRA to Attention layers (Query and Value)
    # RoBERTa structure: roberta.encoder.layer.X.attention.self.query/key/value
    
    for name, module in model.named_modules():
        if "attention.self.query" in name or "attention.self.value" in name:
            if isinstance(module, nn.Linear):
                # We need to replace the module in the parent
                # But named_modules() returns the module, not the parent.
                # So we need a different approach or iterate and replace.
                pass

    # Better approach: iterate over layers
    for layer in model.roberta.encoder.layer:
        # Query
        layer.attention.self.query = LoRALinear(layer.attention.self.query, rank, alpha, dropout)
        # Value
        layer.attention.self.value = LoRALinear(layer.attention.self.value, rank, alpha, dropout)
        
    # Also unfreeze the classifier head if we want to train it?
    # Usually for classification, we want to train the head.
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"LoRA Applied. Trainable params: {trainable_params} / {all_params} ({100 * trainable_params / all_params:.2f}%)")
    
    return model

class CLUTRRRobertaDataset(Dataset):
    def __init__(self, root, dataset, split, data_percentage, tokenizer, max_length=256):
        self.dataset_dir = os.path.join(root, f"{dataset}/")
        self.file_names = [os.path.join(self.dataset_dir, d) for d in os.listdir(self.dataset_dir) if f"_{split}.csv" in d]
        self.data = [row for f in self.file_names for row in list(csv.reader(open(f)))[1:]]
        self.data_num = math.floor(len(self.data) * data_percentage / 100)
        self.data = self.data[:self.data_num]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # Context
        context_str = self.data[i][2]
        # Query: (sub, obj)
        query_sub_obj = eval(self.data[i][3])
        query_str = f"What is the relation between {query_sub_obj[0]} and {query_sub_obj[1]}?"
        
        # Use tokenizer to encode the pair (Context, Query)
        # This handles [SEP] tokens correctly and allows for proper truncation
        encoding = self.tokenizer(
            context_str,
            query_str,
            max_length=self.max_length,
            padding="max_length",
            truncation="only_first", # Truncate context if too long, keep query
            return_tensors="pt"
        )
        
        answer = self.data[i][5]
        label = relation_id_map[answer]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer and Model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    roberta_path = os.path.join(script_dir, "../roberta-base")
    
    print(f"Loading RoBERTa from {roberta_path}")
    tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
    config = RobertaConfig.from_pretrained(roberta_path, num_labels=len(relation_id_map))
    model = RobertaForSequenceClassification.from_pretrained(roberta_path, config=config)

    if args.use_lora:
        print(f"Applying LoRA with rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        model = apply_lora_to_roberta(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
    
    model.to(device)

    # Load Data
    root = os.path.join(script_dir, "../data/clutrr")
    train_dataset = CLUTRRRobertaDataset(root, args.dataset, "train", args.data_percentage, tokenizer)
    test_dataset = CLUTRRRobertaDataset(root, args.dataset, "test", 100, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': total_loss / (total/args.batch_size + 1e-9), 'acc': correct / total})

        train_acc = correct / total
        print(f"Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f} Train Acc: {train_acc:.4f}")

        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
        
        test_acc = test_correct / test_total
        print(f"Epoch {epoch+1} Test Acc: {test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"clutrr_roberta_lora_best.pth")
            print("Saved best model.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data_db9b8f04", help="Dataset folder name")
    parser.add_argument("--data_percentage", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    
    # LoRA Config
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")

    args = parser.parse_args()
    train(args)
