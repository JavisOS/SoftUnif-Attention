import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Attention Modules ---

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

        # --- Value Stream ---
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # --- Latent Variable Soft-Unification (LVSU) ---
        self.num_types = 5 # NOISE, ENT, REL_VERT, REL_HOR, REL_MIX
        
        # 1. Type Inference
        self.type_inferer = nn.Linear(hidden_dim, self.num_types)
        self.type_embeddings = nn.Embedding(self.num_types, self.head_dim)
        
        # 2. Unification Kernel (Independent Projections for Type Stream)
        self.q_type = nn.Linear(self.head_dim, self.head_dim)
        self.k_type = nn.Linear(self.head_dim, self.head_dim)
        
        # Bilinear Weight for Unification
        self.type_bilinear = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.head_dim))
        nn.init.xavier_uniform_(self.type_bilinear)
        
        # Removed explicit type_bias initialization to 2.0

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # === Step A: Value Stream ===
        q_val = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_val = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard Scaled Dot Product Attention for Value
        val_scores = torch.matmul(q_val, k_val.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # === Step B: Latent Type Inference ===
        # [B, L, 5]
        type_logits = self.type_inferer(x)
        type_probs = F.softmax(type_logits, dim=-1)
        
        # Soft Type Vector: [B, L, 5] @ [5, Head_Dim] -> [B, L, Head_Dim]
        latent_type_vec = torch.matmul(type_probs, self.type_embeddings.weight)

        # === Step C: Unification Stream ===
        # Project Type Vectors
        q_type_vec = self.q_type(latent_type_vec) # [B, L, Head_Dim]
        k_type_vec = self.k_type(latent_type_vec) # [B, L, Head_Dim]
        
        # Expand for heads: [B, 1, L, Head_Dim]
        q_type_vec = q_type_vec.unsqueeze(1)
        k_type_vec = k_type_vec.unsqueeze(1)
        
        # Bilinear: q @ W @ k.T
        W_expanded = self.type_bilinear.unsqueeze(0) # [1, H, D, D]
        q_type_transformed = torch.matmul(q_type_vec, W_expanded) # [B, H, L, D]
        
        # [B, H, L, D] @ [B, 1, D, L] -> [B, H, L, L]
        unification_logits = torch.matmul(q_type_transformed, k_type_vec.transpose(-2, -1))
        
        # Scale
        unification_logits = unification_logits / math.sqrt(self.head_dim)
        
        unification_prob = torch.sigmoid(unification_logits)

        # === Step D: Fusion ===
        epsilon = 1e-6
        combined_scores = val_scores + torch.log(unification_prob + epsilon)

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            combined_scores = combined_scores.masked_fill(mask_expanded == 0, float('-inf'))

        # === Step E: Output ===
        attn_weights = F.softmax(combined_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)
        
        return output, type_logits, unification_logits

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
        if isinstance(self.attention, SoftUnifAttention):
            attn_out, type_logits, unify_logits = self.attention(x, mask)
        else:
            attn_out = self.attention(x, mask)
            type_logits, unify_logits = None, None
            
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, type_logits, unify_logits