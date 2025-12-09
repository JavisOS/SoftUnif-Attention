import torch
import torch.nn as nn
import math
import re
from transformers import RobertaTokenizerFast, RobertaModel, RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
from .layers import TransformerEncoderLayer, RotaryEmbedding, MLP

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
