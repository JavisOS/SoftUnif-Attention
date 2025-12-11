import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast
from .layers import TransformerEncoderLayer, MLP


class SmallTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers,
                 num_heads, ffn_dim, dropout,
                 max_len=512, attention_type='vanilla'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # 这里依然不显式加 position embedding，避免把逻辑信息“硬编码”进 Type/Unify 分支
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_dim, num_heads, ffn_dim, dropout, attention_type
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: [B,L]
        x = self.embedding(input_ids)
        x = self.dropout(x)

        all_type_logits = []
        all_unify_logits = []

        for layer in self.layers:
            x, type_logits, unify_logits = layer(x, attention_mask)
            if type_logits is not None:
                all_type_logits.append(type_logits)      # [B,L,5]
            if unify_logits is not None:
                all_unify_logits.append(unify_logits)    # [B,H,L,L]

        x = self.norm(x)
        return x, all_type_logits, all_unify_logits


class CLUTRRTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device

        # Tokenizer (just for vocab size)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            "roberta-base", add_prefix_space=True
        )
        vocab_size = self.tokenizer.vocab_size

        self.encoder = SmallTransformerEncoder(
            vocab_size=vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            attention_type=config.attention_type,
        )

        # Relation classifier: [sub; obj; story] -> relation logits
        self.relation_classifier = MLP(
            in_dim=config.hidden_dim * 3,
            embed_dim=config.hidden_dim,
            out_dim=config.num_relations,
            num_layers=config.mlp_layers,
        )

    def forward(self, input_ids, attention_mask, sub_indices, obj_indices):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        sequence_output, all_type_logits, all_unify_logits = \
            self.encoder(input_ids, attention_mask)

        batch_features = []

        for i in range(len(input_ids)):
            def get_emb(indices):
                if not indices:
                    return torch.zeros(self.config.hidden_dim,
                                       device=self.device)
                idx_tensor = torch.tensor(indices, device=self.device)
                embs = sequence_output[i, idx_tensor, :]
                return embs.mean(dim=0)

            sub_emb = get_emb(sub_indices[i])
            obj_emb = get_emb(obj_indices[i])

            # 全序列平均作为 story embedding
            valid_len = attention_mask[i].sum().item()
            if valid_len > 0:
                story_emb = sequence_output[i,
                                            :int(valid_len), :].mean(dim=0)
            else:
                story_emb = sequence_output[i].mean(dim=0)

            feature = torch.cat([sub_emb, obj_emb, story_emb], dim=-1)
            batch_features.append(feature)

        batch_features = torch.stack(batch_features, dim=0)
        logits = self.relation_classifier(batch_features)

        return logits, all_type_logits, all_unify_logits, attention_mask
