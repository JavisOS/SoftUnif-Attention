import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
from transformers import RobertaTokenizerFast

class VanillaAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, hidden_dim]
        # mask: [batch_size, seq_len] (1 for valid, 0 for padding)
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scores: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # mask is [batch_size, seq_len]
            # Expand to [batch_size, 1, 1, seq_len]
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Output: [batch_size, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        # Combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)
        return output

class SoftUnifAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        # Split head_dim into value_dim and type_dim
        # For simplicity, let's split evenly or define a ratio. 
        # The prompt says "explicitly partition". Let's say 50/50 for now, or make it configurable.
        # If head_dim is odd, this might be an issue. Let's assume even for now.
        self.value_dim = self.head_dim // 2
        self.type_dim = self.head_dim - self.value_dim

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable bilinear/affine for type unification
        # We want a score between q_type and k_type.
        # Let's use a bilinear form: q_type @ W @ k_type^T
        self.type_bilinear = nn.Parameter(torch.empty(self.num_heads, self.type_dim, self.type_dim))
        nn.init.xavier_uniform_(self.type_bilinear)
        
        # Bias for the type score before sigmoid
        # Initialize to a positive value so that initially the model allows most interactions (soft mask ~ 1)
        self.type_bias = nn.Parameter(torch.ones(self.num_heads) * 2.0)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Split into value and type
        q_val = q[..., :self.value_dim]
        q_type = q[..., self.value_dim:]
        k_val = k[..., :self.value_dim]
        k_type = k[..., self.value_dim:]

        # 1. Value Similarity (Vanilla-like)
        # [batch_size, num_heads, seq_len, seq_len]
        val_scores = torch.matmul(q_val, k_val.transpose(-2, -1)) / math.sqrt(self.value_dim)

        # 2. Type Unification
        # We want to compute q_type @ W @ k_type^T for each head.
        # q_type: [B, H, L, D_t]
        # W: [H, D_t, D_t]
        # k_type: [B, H, L, D_t]
        
        # Let's reshape to use matmul efficiently
        # q_type_W = q_type @ W
        # But W is per head.
        # We can use einsum.
        # B: batch, H: heads, L: seq_len, S: seq_len (target), D: type_dim
        # q: BHLD, k: BHSD, W: HDD
        # result: BHLS
        
        # q_type @ W
        # [B, H, L, D] @ [H, D, D] -> [B, H, L, D] (broadcasting over B)
        # Actually torch.matmul handles broadcasting if dimensions match.
        # We need to align W to [1, H, D, D]
        W_expanded = self.type_bilinear.unsqueeze(0) # [1, H, D, D]
        q_type_transformed = torch.matmul(q_type, W_expanded) # [B, H, L, D]
        
        # Now dot product with k_type
        # [B, H, L, D] @ [B, H, D, S] -> [B, H, L, S]
        type_scores_raw = torch.matmul(q_type_transformed, k_type.transpose(-2, -1))
        
        # Add bias and sigmoid
        type_scores_raw = type_scores_raw + self.type_bias.view(1, self.num_heads, 1, 1)
        type_unification = torch.sigmoid(type_scores_raw)

        # 3. Combine
        # Prompt: "combine value similarity and type unification... e.g. multiplication, addition"
        # Let's use multiplication in probability space, which corresponds to addition in log space.
        # But val_scores are logits (unbounded).
        # If we treat type_unification as a gating factor (0-1), we can multiply the attention weights?
        # Or we can add log(type_unification) to val_scores?
        # Let's try: final_score = val_scores + log(type_unification + epsilon)
        # Or simply: final_score = val_scores * type_unification? No, val_scores can be negative.
        
        # Let's interpret type_unification as a soft mask.
        # If type_unification is 1, we keep val_scores.
        # If type_unification is 0, we want score to be -inf.
        # So adding log(type_unification) makes sense.
        
        epsilon = 1e-6
        combined_scores = val_scores + torch.log(type_unification + epsilon)

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            combined_scores = combined_scores.masked_fill(mask_expanded == 0, float('-inf'))

        attn_weights = F.softmax(combined_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)
        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout, attention_type='vanilla'):
        super().__init__()
        if attention_type == 'vanilla':
            self.attention = VanillaAttention(hidden_dim, num_heads, dropout)
        elif attention_type == 'softunif':
            self.attention = SoftUnifAttention(hidden_dim, num_heads, dropout)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self Attention
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class SmallTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, ffn_dim, dropout, max_len=512, attention_type='vanilla'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout, attention_type)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
            
        x = self.norm(x)
        return x

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
        story_name_maps = []
        
        # Reconstruct stories and find name indices
        # We need to be careful to map names to the NEW token indices in the concatenated story.
        
        for i, (start, end) in enumerate(context_splits):
            # Get sentences for this story
            sentences = contexts[start:end]
            
            # Join sentences
            # We use a separator that the tokenizer handles well, or just space.
            # RoBERTa uses <s> and </s>.
            # Let's just join with ". " as in the original code's union_sentence logic
            # But wait, the original code merges sentences.
            # Here we want to join ALL sentences in the story.
            
            full_story = " ".join(sentences)
            story_texts.append(full_story)
            
            # Find names in the full story
            # We can reuse the regex logic
            names = set(re.findall(r"\[(\w+)\]", full_story))
            
            # We need to tokenize the full story and find where the names are.
            # This is slightly expensive to do inside forward, but necessary if we don't pre-tokenize.
            # To make it efficient, we can tokenize the batch of stories.
        
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
            # However, names in text are like "[Alice]".
            # The tokenizer might split "[Alice]" into multiple tokens.
            # A robust way is to find the character offsets of the names in the string, 
            # and map them to token indices using `encodings.char_to_token`.
            
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

