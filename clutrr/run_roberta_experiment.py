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
import transformers
from transformers import RobertaTokenizerFast, RobertaModel
import re

import sys

# Add the current directory to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import custom modules
from models.transformer_baseline import TransformerEncoderLayer, RotaryEmbedding, MLP

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
    
    return ((context, query), answer)

  @staticmethod
  def collate_fn(batch):
    queries = [query for ((_, query), _) in batch]
    contexts = [fact for ((context, _), _) in batch for fact in context]
    context_lens = [len(context) for ((context, _), _) in batch]
    context_splits = [(sum(context_lens[:i]), sum(context_lens[:i + 1])) for i in range(len(context_lens))]
    answers = torch.stack([torch.tensor(relation_id_map[answer]) for (_, answer) in batch])
    return ((contexts, queries, context_splits), answers)

class CLUTRRRobertaWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device
        self.model_type = config.model_type
        
        # Tokenizer and Backbone
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        
        if config.freeze_roberta:
            print("Freezing RoBERTa backbone...")
            for param in self.roberta.parameters():
                param.requires_grad = False
                
        self.hidden_dim = 768 # RoBERTa base hidden dim
        
        if self.model_type == 'roberta_softunif':
            print("Initializing SoftUnif Layers...")
            # SoftUnif Layers
            # Note: RoBERTa base has 12 heads, 768 dim. 
            # We can use config.num_heads if provided, but it must divide 768.
            # Defaulting to 12 if not specified or ensuring it matches.
            num_heads = config.num_heads if config.num_heads > 0 else 12
            
            self.softunif_layers = nn.ModuleList([
                TransformerEncoderLayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=config.ffn_dim,
                    dropout=config.dropout,
                    attention_type='softunif'
                ) for _ in range(config.num_layers)
            ])
            
            # RoPE
            # SoftUnif splits head_dim in half
            head_dim = self.hidden_dim // num_heads
            rope_dim = head_dim // 2
            self.rotary_emb = RotaryEmbedding(rope_dim, max_position_embeddings=512)
            
        # Classifier
        self.relation_classifier = MLP(
            in_dim=self.hidden_dim * 3,
            embed_dim=self.hidden_dim,
            out_dim=config.num_relations,
            num_layers=config.mlp_layers
        )

    def forward(self, contexts, context_splits, queries):
        # Preprocess batch into stories
        story_texts = []
        
        for i, (start, end) in enumerate(context_splits):
            sentences = contexts[start:end]
            full_story = " ".join(sentences)
            story_texts.append(full_story)
            
        # Tokenize
        encodings = self.tokenizer(story_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        input_ids = encodings.input_ids.to(self.device)
        attention_mask = encodings.attention_mask.to(self.device)
        
        # Pass through RoBERTa
        roberta_output = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = roberta_output.last_hidden_state # [B, L, D]
        
        if self.model_type == 'roberta_softunif':
            # === Static Anchoring ===
            # Extract raw word embeddings (before position/layer norms)
            static_x = self.roberta.embeddings.word_embeddings(input_ids)
            
            # Prepare RoPE
            seq_len = input_ids.size(1)
            cos, sin = self.rotary_emb(sequence_output, seq_len=seq_len)
            
            # Pass through SoftUnif Layers
            x = sequence_output
            for layer in self.softunif_layers:
                x = layer(x, attention_mask, rotary_pos_emb=(cos, sin), static_x=static_x)
            
            sequence_output = x
            
        # Extract Entities and Story Rep
        batch_features = []
        
        for i, (sub, obj) in enumerate(queries):
            story_text = story_texts[i]
            
            # Helper to get embedding for a name
            def get_name_embedding(name):
                pattern = re.escape(f"[{name}]")
                matches = list(re.finditer(pattern, story_text))
                
                if not matches:
                    pattern = re.escape(name)
                    matches = list(re.finditer(pattern, story_text))
                
                if not matches:
                    return torch.zeros(self.hidden_dim, device=self.device)

                token_indices = []
                for match in matches:
                    start_char, end_char = match.span()
                    for char_idx in range(start_char, end_char):
                        token_idx = encodings.char_to_token(i, char_idx)
                        if token_idx is not None:
                            token_indices.append(token_idx)
                
                if not token_indices:
                    return torch.zeros(self.hidden_dim, device=self.device)
                
                token_indices = list(set(token_indices))
                embs = sequence_output[i, token_indices, :]
                return torch.mean(embs, dim=0)

            sub_emb = get_name_embedding(sub)
            obj_emb = get_name_embedding(obj)
            
            # Global story rep
            seq_emb = sequence_output[i]
            mask = attention_mask[i].unsqueeze(-1)
            sum_emb = (seq_emb * mask).sum(dim=0)
            count = mask.sum()
            story_emb = sum_emb / (count + 1e-9)
            
            pair_feature = torch.cat([sub_emb, obj_emb, story_emb], dim=0)
            batch_features.append(pair_feature)
            
        batch_features = torch.stack(batch_features)
        
        # Classification
        logits = self.relation_classifier(batch_features)
        return logits

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.num_relations = len(relation_id_map)

    print(f"Using device: {device}")
    print(f"Model Type: {args.model_type}")

    # Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(script_dir, "../data")
    
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
    model = CLUTRRRobertaWrapper(args).to(device)
    
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
            torch.save(model.state_dict(), f"clutrr_{args.model_type}_best.pth")
            print("Saved best model.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data_db9b8f04", help="Dataset folder name")
    parser.add_argument("--data_percentage", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5) # Lower LR for RoBERTa
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    
    # Model Config
    parser.add_argument("--model_type", type=str, default="roberta_vanilla", choices=["roberta_vanilla", "roberta_softunif"])
    parser.add_argument("--freeze_roberta", action="store_true", help="Freeze RoBERTa backbone")
    
    # SoftUnif Config (Only used if model_type is roberta_softunif)
    parser.add_argument("--num_layers", type=int, default=2, help="Number of SoftUnif layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads (must divide 768)")
    parser.add_argument("--ffn_dim", type=int, default=3072, help="FFN dim for SoftUnif layers")
    parser.add_argument("--dropout", type=float, default=0.1)
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
