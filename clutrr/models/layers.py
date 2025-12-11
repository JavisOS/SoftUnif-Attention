import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        B, L, D = x.size()
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
            scores = scores.masked_fill(mask_expanded == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B,H,L,Dh]
        out = out.transpose(1, 2).contiguous().view(B, L, self.hidden_dim)
        out = self.out_proj(out)
        return out


class SoftUnifAttention(nn.Module):
    """
    Multi-head attention + unification bias:

        scores = value_scores + alpha * unify_scores

    unify_scores 来自 x 经 Q_u / K_u 计算；type 头只做辅助监督，不再限制 unify 的表达力。
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        # Value stream
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Type head (5 类：NOISE, ENT, REL_VERT, REL_HOR, REL_MIX)
        self.num_types = 5
        self.type_inferer = nn.Linear(hidden_dim, self.num_types)

        # Unification stream
        self.q_unif = nn.Linear(hidden_dim, hidden_dim)
        self.k_unif = nn.Linear(hidden_dim, hidden_dim)

        # Learnable scale for unify bias: 初始化为 0，让主任务自己决定是否“打开”这条路
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, mask=None):
        """
        x: [B, L, D]
        mask: [B, L]
        Returns:
            output: [B, L, D]
            type_logits: [B, L, 5]
            unify_logits: [B, H, L, L]
        """
        B, L, D = x.size()

        # Value stream
        q_val = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k_val = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        value_scores = torch.matmul(q_val, k_val.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,L,L]

        # Type logits
        type_logits = self.type_inferer(x)  # [B,L,5]

        # Unification stream
        q_u = self.q_unif(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k_u = self.k_unif(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        unify_logits = torch.matmul(q_u, k_u.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,L,L]

        # Fusion
        scores = value_scores + self.alpha * unify_logits

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
            scores = scores.masked_fill(mask_expanded == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B,H,L,Dh]
        out = out.transpose(1, 2).contiguous().view(B, L, self.hidden_dim)
        out = self.out_proj(out)

        return out, type_logits, unify_logits


class MLP(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim, num_layers=1):
        super().__init__()
        layers = [nn.Linear(in_dim, embed_dim), nn.ReLU()]
        for _ in range(num_layers):
            layers += [nn.Linear(embed_dim, embed_dim), nn.ReLU()]
        layers += [nn.Linear(embed_dim, out_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout, attention_type="vanilla"):
        super().__init__()
        if attention_type == "vanilla":
            self.attention = VanillaAttention(hidden_dim, num_heads, dropout)
        elif attention_type == "softunif":
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
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if isinstance(self.attention, SoftUnifAttention):
            attn_out, type_logits, unify_logits = self.attention(x, mask)
        else:
            attn_out = self.attention(x, mask)
            type_logits, unify_logits = None, None

        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x, type_logits, unify_logits
