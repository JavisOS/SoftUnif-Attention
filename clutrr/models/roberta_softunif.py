import torch
import torch.nn as nn
import math

# --- RoPE Implementation (Copied from transformer_baseline.py) ---

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

class RobertaSoftUnifSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # SoftUnif specific
        # We use full head_dim for Value Similarity (to preserve pretrained knowledge)
        # And we add separate projections for Type Unification
        self.value_dim = self.attention_head_size
        self.type_dim = self.attention_head_size // 2 # Can be smaller or same
        
        self.type_query = nn.Linear(config.hidden_size, self.num_attention_heads * self.type_dim)
        self.type_key = nn.Linear(config.hidden_size, self.num_attention_heads * self.type_dim)
        
        # Initialize Type Projections with small weights to minimize initial impact
        nn.init.normal_(self.type_query.weight, std=0.01)
        nn.init.zeros_(self.type_query.bias)
        nn.init.normal_(self.type_key.weight, std=0.01)
        nn.init.zeros_(self.type_key.bias)
        
        self.type_bilinear = nn.Parameter(torch.zeros(self.num_attention_heads, self.type_dim, self.type_dim))
        # nn.init.xavier_uniform_(self.type_bilinear) # Too aggressive
        # Initialize bilinear to near zero so type_score is dominated by bias initially
        nn.init.normal_(self.type_bilinear, std=0.01)
        
        self.type_bias = nn.Parameter(torch.ones(self.num_attention_heads) * 5.0) # Higher bias to ensure sigmoid ~ 1.0 initially

        # RoPE
        # Apply RoPE to the Value part (standard RoBERTa attention)
        rope_dim = self.value_dim
        self.rotary_emb = RotaryEmbedding(rope_dim, max_position_embeddings=config.max_position_embeddings)

    def transpose_for_scores(self, x, head_dim=None):
        if head_dim is None:
            head_dim = self.attention_head_size
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, head_dim)
        x = x.view(*new_x_shape)
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
        past_key_values=None,
        **kwargs,
    ):
        if past_key_value is None and past_key_values is not None:
            past_key_value = past_key_values

        # Projections
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # Type Projections (New)
        mixed_type_query_layer = self.type_query(hidden_states)
        mixed_type_key_layer = self.type_key(hidden_states)

        # Transpose
        # Value (Standard Attention)
        q_val = self.transpose_for_scores(mixed_query_layer)
        k_val = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Type
        q_type = self.transpose_for_scores(mixed_type_query_layer, head_dim=self.type_dim)
        k_type = self.transpose_for_scores(mixed_type_key_layer, head_dim=self.type_dim)

        # Apply RoPE to Value part (Standard RoBERTa behavior)
        seq_len = hidden_states.size(1)
        cos, sin = self.rotary_emb(q_val, seq_len=seq_len)
        q_val, k_val = apply_rotary_pos_emb(q_val, k_val, cos, sin)
        
        # 1. Value Similarity (Standard Attention)
        # [batch, num_heads, seq_len, seq_len]
        val_scores = torch.matmul(q_val, k_val.transpose(-1, -2)) / math.sqrt(self.value_dim)
        
        # 2. Type Unification
        # q_type: [B, H, L, D_t]
        # W: [H, D_t, D_t]
        W_expanded = self.type_bilinear.unsqueeze(0) # [1, H, D, D]
        q_type_transformed = torch.matmul(q_type, W_expanded) # [B, H, L, D]
        
        type_scores_raw = torch.matmul(q_type_transformed, k_type.transpose(-1, -2)) # [B, H, L, L]
        type_scores_raw = type_scores_raw + self.type_bias.view(1, self.num_attention_heads, 1, 1)
        type_unification = torch.sigmoid(type_scores_raw)
        
        # 3. Combine
        epsilon = 1e-6
        attention_scores = val_scores + torch.log(type_unification + epsilon)

        # Apply Attention Mask
        if attention_mask is not None:
            # RoBERTa attention_mask is usually additive (0 for keep, -large for mask)
            attention_scores = attention_scores + attention_mask

        # Softmax
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Context
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

def convert_roberta_to_softunif(model, config):
    print("Converting RoBERTa to use SoftUnifAttention...")
    for i, layer in enumerate(model.roberta.encoder.layer):
        # Get original weights
        original_self_attn = layer.attention.self
        
        # Create new SoftUnif module
        new_self_attn = RobertaSoftUnifSelfAttention(config)
        
        # Copy weights (Q, K, V)
        # We copy the FULL weights now, as we use them for Value Similarity
        new_self_attn.query.weight.data = original_self_attn.query.weight.data
        new_self_attn.query.bias.data = original_self_attn.query.bias.data
        
        new_self_attn.key.weight.data = original_self_attn.key.weight.data
        new_self_attn.key.bias.data = original_self_attn.key.bias.data
        
        new_self_attn.value.weight.data = original_self_attn.value.weight.data
        new_self_attn.value.bias.data = original_self_attn.value.bias.data
        
        # Initialize Type Projections (New)
        # We can initialize them randomly, or maybe from a slice of Q/K to give them a head start?
        # Let's stick to random initialization (Xavier) as defined in __init__ (by nn.Linear default)
        # But we need to make sure they are on the correct device
        new_self_attn.type_query.to(original_self_attn.query.weight.device)
        new_self_attn.type_key.to(original_self_attn.key.weight.device)
        new_self_attn.type_bilinear.data = new_self_attn.type_bilinear.data.to(original_self_attn.query.weight.device)
        new_self_attn.type_bias.data = new_self_attn.type_bias.data.to(original_self_attn.query.weight.device)
        
        # Replace
        layer.attention.self = new_self_attn
        
    print("Conversion complete.")
    return model
