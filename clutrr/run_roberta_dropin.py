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
from transformers import RobertaTokenizerFast, RobertaModel, RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
import re
import sys

# Add the current directory to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

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

# --- Soft Unification Drop-in Module ---

class SoftUnificationRobertaSelfAttention(RobertaSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        # Value Stream uses original self.query, self.key, self.value from RobertaSelfAttention
        
        # Type Stream (New Independent Projections)
        self.type_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.type_key = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Gating parameters
        # Initialize bias to +5.0 so Sigmoid starts near 1.0 (Open Gate)
        # This ensures smooth transition from pre-trained weights
        self.type_bias = nn.Parameter(torch.ones(1) * 5.0) 
        
        # Temperature for scaling (optional, but good for stability)
        self.type_temp = math.sqrt(self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        **kwargs,
    ):
        # 1. Value Stream (Standard RoBERTa Attention)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 2. Type Stream (Independent Gated Logic)
        mixed_type_query = self.type_query(hidden_states)
        mixed_type_key = self.type_key(hidden_states)
        
        type_query_layer = self.transpose_for_scores(mixed_type_query)
        type_key_layer = self.transpose_for_scores(mixed_type_key)
        
        # Type Scores
        type_scores = torch.matmul(type_query_layer, type_key_layer.transpose(-1, -2))
        type_scores = type_scores / self.type_temp
        
        # Gating: Sigmoid(TypeScore + Bias)
        # Bias starts at +5.0, so Sigmoid is ~0.993 initially
        type_gate = torch.sigmoid(type_scores + self.type_bias)
        
        # 3. Fusion
        # Score = Value_Score + log(Gate + epsilon)
        epsilon = 1e-6
        attention_scores = attention_scores + torch.log(type_gate + epsilon)

        # 4. Standard Masking and Softmax
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

def convert_roberta_to_soft_unification(model, config):
    """
    Recursively replace RobertaSelfAttention with SoftUnificationRobertaSelfAttention
    and copy pre-trained weights.
    """
    for name, module in model.named_children():
        if isinstance(module, RobertaSelfAttention) and not isinstance(module, SoftUnificationRobertaSelfAttention):
            print(f"Converting layer: {name} to SoftUnification")
            new_module = SoftUnificationRobertaSelfAttention(config)
            
            # Copy Weights (CRITICAL)
            new_module.query.weight.data = module.query.weight.data
            new_module.query.bias.data = module.query.bias.data
            new_module.key.weight.data = module.key.weight.data
            new_module.key.bias.data = module.key.bias.data
            new_module.value.weight.data = module.value.weight.data
            new_module.value.bias.data = module.value.bias.data
            
            # Initialize Type Projections (Xavier)
            nn.init.xavier_uniform_(new_module.type_query.weight)
            nn.init.zeros_(new_module.type_query.bias)
            nn.init.xavier_uniform_(new_module.type_key.weight)
            nn.init.zeros_(new_module.type_key.bias)
            
            # Replace in parent
            setattr(model, name, new_module)
        else:
            convert_roberta_to_soft_unification(module, config)

class MLP(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim, num_layers=1):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, embed_dim), nn.ReLU()]
        for _ in range(num_layers):
            layers += [nn.Linear(embed_dim, embed_dim), nn.ReLU()]
        layers += [nn.Linear(embed_dim, out_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class CLUTRRRobertaDropIn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        
        # Load Config & Model
        self.config = RobertaConfig.from_pretrained("roberta-base")
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        
        # Apply Drop-in Replacement if requested
        if args.model_type == 'roberta_softunif':
            print("Applying Soft Unification Drop-in Replacement...")
            convert_roberta_to_soft_unification(self.roberta, self.config)
            
        # Classifier
        self.relation_classifier = MLP(
            in_dim=self.config.hidden_size * 3,
            embed_dim=self.config.hidden_size,
            out_dim=args.num_relations,
            num_layers=args.mlp_layers
        )
        
        # Tokenizer
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)

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
        
        # Pass through RoBERTa (Modified or Vanilla)
        roberta_output = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = roberta_output.last_hidden_state # [B, L, D]
        
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
                    return torch.zeros(self.config.hidden_size, device=self.device)

                token_indices = []
                for match in matches:
                    start_char, end_char = match.span()
                    for char_idx in range(start_char, end_char):
                        token_idx = encodings.char_to_token(i, char_idx)
                        if token_idx is not None:
                            token_indices.append(token_idx)
                
                if not token_indices:
                    return torch.zeros(self.config.hidden_size, device=self.device)
                
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
            torch.save(model.state_dict(), f"clutrr_dropin_{args.model_type}_best.pth")
            print("Saved best model.")

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
