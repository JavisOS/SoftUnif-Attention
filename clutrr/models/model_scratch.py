import torch
import torch.nn as nn
import re
from transformers import RobertaTokenizerFast
from .layers import TransformerEncoderLayer, RotaryEmbedding, MLP

class SmallTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, ffn_dim, dropout, max_len=512, attention_type='vanilla'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # Absolute positional embedding is intentionally removed to prevent leakage into Type subspace
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout, attention_type)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Initialize RoPE
        if attention_type == 'softunif':
            # SoftUnif splits head_dim in half, only applies RoPE to value part
            rope_dim = (hidden_dim // num_heads) // 2
        else:
            # Vanilla applies RoPE to full head_dim
            rope_dim = hidden_dim // num_heads
            
        self.rotary_emb = RotaryEmbedding(rope_dim, max_position_embeddings=max_len)

    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.size(1)
        
        # Raw semantic embeddings (No positional info added here)
        x = self.embedding(input_ids)
        
        # Capture static embeddings for Type Anchoring
        static_x = x
        
        x = self.dropout(x)
        
        # Get RoPE embeddings
        cos, sin = self.rotary_emb(x, seq_len=seq_len)
        
        for layer in self.layers:
            x = layer(x, attention_mask, rotary_pos_emb=(cos, sin), static_x=static_x)
            
        x = self.norm(x)
        return x

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

    def forward(self, contexts, context_splits, queries):
        # 1. Preprocess batch into stories
        # contexts: list of sentences
        # context_splits: list of (start, end) indices into contexts
        # queries: list of (sub, obj) tuples
        
        story_texts = []
        
        # Reconstruct stories and find name indices
        # We need to be careful to map names to the NEW token indices in the concatenated story.
        
        for i, (start, end) in enumerate(context_splits):
            # Get sentences for this story
            sentences = contexts[start:end]
            
            # Join sentences
            full_story = " ".join(sentences)
            story_texts.append(full_story)
            
        # Tokenize batch of stories
        encodings = self.tokenizer(story_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        input_ids = encodings.input_ids.to(self.device)
        attention_mask = encodings.attention_mask.to(self.device)
        
        # Run Encoder
        # [batch_size, seq_len, hidden_dim]
        sequence_output = self.encoder(input_ids, attention_mask)
        
        # Extract Entities and Story Rep
        batch_features = []
        
        for i, (sub, obj) in enumerate(queries):
            # We need to find the token indices for 'sub' and 'obj' in story_texts[i]
            # Since we tokenized story_texts[i], we can search for the tokens corresponding to the names.
            
            story_text = story_texts[i]
            
            # Helper to get embedding for a name
            def get_name_embedding(name):
                # Find all occurrences of [name]
                # We need to escape brackets for regex
                pattern = re.escape(f"[{name}]")
                matches = list(re.finditer(pattern, story_text))
                
                if not matches:
                    # Fallback: try without brackets if not found (should not happen given dataset)
                    pattern = re.escape(name)
                    matches = list(re.finditer(pattern, story_text))
                
                if not matches:
                    # If still not found, return zero vector or global rep (should not happen)
                    return torch.zeros(self.config.hidden_dim, device=self.device)

                # Collect all token indices for this name
                token_indices = []
                for match in matches:
                    start_char, end_char = match.span()
                    # char_to_token returns None if char is not in a token (e.g. whitespace)
                    # We scan the range
                    for char_idx in range(start_char, end_char):
                        token_idx = encodings.char_to_token(i, char_idx)
                        if token_idx is not None:
                            token_indices.append(token_idx)
                
                if not token_indices:
                    return torch.zeros(self.config.hidden_dim, device=self.device)
                
                token_indices = list(set(token_indices))
                # Select embeddings
                # [num_tokens, hidden_dim]
                embs = sequence_output[i, token_indices, :]
                # Pool (mean or max)
                return torch.mean(embs, dim=0)

            sub_emb = get_name_embedding(sub)
            obj_emb = get_name_embedding(obj)
            
            # Global story rep: mean of all tokens (masked)
            # [seq_len, hidden_dim]
            seq_emb = sequence_output[i]
            mask = attention_mask[i].unsqueeze(-1) # [seq_len, 1]
            # Sum valid tokens
            sum_emb = (seq_emb * mask).sum(dim=0)
            count = mask.sum()
            story_emb = sum_emb / (count + 1e-9)
            
            # Concatenate
            # [3 * hidden_dim]
            pair_feature = torch.cat([sub_emb, obj_emb, story_emb], dim=0)
            batch_features.append(pair_feature)
            
        batch_features = torch.stack(batch_features)
        
        # Classification
        logits = self.relation_classifier(batch_features)
        return logits
