import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- RoPE Implementation ---

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for cos/sin
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from complex number implementation, let's use the sin/cos cache approach
        # freqs: [seq_len, dim/2]
        # We want [seq_len, dim] where we interleave or concat. 
        # Standard RoPE usually does pairs. 
        # Let's use the cat approach: [cos, cos] corresponding to [x1, x2]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [batch, num_heads, seq_len, head_dim]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
            
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype, device=x.device),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [batch, num_heads, seq_len, head_dim]
    # cos, sin: [1, 1, seq_len, head_dim]
    # Ensure dimensions match for broadcasting
    # cos/sin are [1, 1, seq_len, head_dim]
    # q/k are [batch, num_heads, seq_len, head_dim]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

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

    def forward(self, x, mask=None, rotary_pos_emb=None, static_x=None):
        # x: [batch_size, seq_len, hidden_dim]
        # mask: [batch_size, seq_len] (1 for valid, 0 for padding)
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if rotary_pos_emb is not None:
            cos, sin = rotary_pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

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

        # === New: Post-SDPA Gating ===
        # Learnable gate for each head output
        # Shape: [1, num_heads, 1, 1] for broadcasting
        self.output_gate_params = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))

    def forward(self, x, mask=None, rotary_pos_emb=None, static_x=None):
        batch_size, seq_len, _ = x.size()

        # Value projections use the dynamic hidden state x
        q_val_full = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_val_full = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Type projections use the STATIC embedding static_x if provided, else x
        # This enforces "Static Type Anchoring"
        type_input = static_x if static_x is not None else x
        q_type_full = self.q_proj(type_input).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_type_full = self.k_proj(type_input).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Split into value and type
        # Note: We use the same projection matrices q_proj/k_proj for both, but apply them to different inputs
        # and then slice. This effectively means we have shared weights but different inputs.
        
        q_val = q_val_full[..., :self.value_dim]
        k_val = k_val_full[..., :self.value_dim]
        
        q_type = q_type_full[..., self.value_dim:]
        k_type = k_type_full[..., self.value_dim:]

        # Apply RoPE ONLY to value part
        if rotary_pos_emb is not None:
            cos, sin = rotary_pos_emb
            # Note: cos, sin must match value_dim
            q_val, k_val = apply_rotary_pos_emb(q_val, k_val, cos, sin)
            # q_type, k_type are NOT rotated -> Position Invariant

        # 1. Value Similarity (Vanilla-like)
        # [batch_size, num_heads, seq_len, seq_len]
        val_scores = torch.matmul(q_val, k_val.transpose(-2, -1)) / math.sqrt(self.value_dim)

        # 2. Type Unification
        # We want to compute q_type @ W @ k_type^T for each head.
        # q_type: [B, H, L, D_t]
        # W: [H, D_t, D_t]
        # k_type: [B, H, L, D_t]
        
        # q_type @ W
        # [B, H, L, D] @ [H, D, D] -> [B, H, L, D] (broadcasting over B)
        # We need to align W to [1, H, D, D]
        W_expanded = self.type_bilinear.unsqueeze(0) # [1, H, D, D]
        q_type_transformed = torch.matmul(q_type, W_expanded) # [B, H, L, D]
        
        # Now dot product with k_type
        # [B, H, L, D] @ [B, H, D, S] -> [B, H, L, S]
        type_scores_raw = torch.matmul(q_type_transformed, k_type.transpose(-2, -1))
        
        
        # === 【新增：缩放 + Bias】 ===
        # 1. 缩放：除以 sqrt(d) 以稳定梯度方差
        type_scores_raw = type_scores_raw / math.sqrt(self.type_dim)
        
        # 2. 加 Bias：让初始门控开启
        # view(1, heads, 1, 1) 用于广播到 batch 和 seq_len
        type_scores_raw = type_scores_raw + self.type_bias.view(1, self.num_heads, 1, 1)
        # ============================

        type_unification = torch.sigmoid(type_scores_raw)

        # 3. Combine
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
        
        # === New: Apply Post-SDPA Gating ===
        # gate: [1, num_heads, 1, 1]
        gate = torch.sigmoid(self.output_gate_params)
        attn_output = attn_output * gate
        # ===================================

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)
        return output

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

    def forward(self, x, mask=None, rotary_pos_emb=None, static_x=None):
        # Self Attention
        attn_out = self.attention(x, mask, rotary_pos_emb, static_x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x
