import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast
from .layers import TransformerEncoderLayer, MLP

class SmallTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, ffn_dim, dropout,
                 max_len=512, attention_type='vanilla'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Learnable positional embedding（恢复顺序信息）
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout, attention_type)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [B, L]
        attention_mask: [B, L]
        """
        batch_size, seq_len = input_ids.size()
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")

        # Token + Position embeddings
        token_emb = self.embedding(input_ids)                              # [B, L, D]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)                            # [B, L, D]

        x = token_emb + pos_emb
        x = self.dropout(x)
        
        all_type_logits = []
        all_unify_logits = []
        
        for layer in self.layers:
            x, type_logits, unify_logits = layer(x, attention_mask)
            if type_logits is not None:
                all_type_logits.append(type_logits)
            if unify_logits is not None:
                all_unify_logits.append(unify_logits)
            
        x = self.norm(x)
        return x, all_type_logits, all_unify_logits

class CLUTRRTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device
        
        # Tokenizer (just for vocab size and processing)
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
        vocab_size = self.tokenizer.vocab_size
        
        self.encoder = SmallTransformerEncoder(
            vocab_size=vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            attention_type=config.attention_type
        )
        
        # Relation Classifier
        # Input: [sub_embed; obj_embed; story_embed] -> 3 * hidden_dim
        self.relation_classifier = MLP(
            in_dim=config.hidden_dim * 3,
            embed_dim=config.hidden_dim,
            out_dim=config.num_relations,
            num_layers=config.mlp_layers
        )

    def forward(self, input_ids, attention_mask, sub_indices, obj_indices):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Run Encoder
        # [batch_size, seq_len, hidden_dim]
        sequence_output, all_type_logits, all_unify_logits = self.encoder(input_ids, attention_mask)
        
        # Extract Entities and Story Rep
        batch_features = []
        
        for i in range(len(input_ids)):
            # Helper to get embedding for indices
            def get_emb(indices):
                if not indices:
                    return torch.zeros(self.config.hidden_dim, device=self.device)
                idx_tensor = torch.tensor(indices, device=self.device)
                embs = sequence_output[i, idx_tensor, :]
                return embs.mean(dim=0)
            
            sub_emb = get_emb(sub_indices[i])
            obj_emb = get_emb(obj_indices[i])
            
            # Story embedding: CLS token (index 0)
            story_emb = sequence_output[i, 0, :]
            
            # Concatenate
            features = torch.cat([sub_emb, obj_emb, story_emb], dim=0)
            batch_features.append(features)
            
        batch_features = torch.stack(batch_features)
        
        # Classifier
        logits = self.relation_classifier(batch_features)
        
        return logits, all_type_logits, all_unify_logits, attention_mask
