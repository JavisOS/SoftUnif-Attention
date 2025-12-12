import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.zeros(rank, original_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, rank))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        original_out = self.original_layer(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_out + lora_out

def apply_lora_to_roberta(model, rank, alpha, dropout):
    # Freeze the entire model first
    for param in model.parameters():
        param.requires_grad = False
        
    # Apply LoRA to Attention layers (Query and Value)
    # RoBERTa structure: roberta.encoder.layer.X.attention.self.query/key/value
    
    # Better approach: iterate over layers
    for layer in model.roberta.encoder.layer:
        # Query
        layer.attention.self.query = LoRALinear(layer.attention.self.query, rank, alpha, dropout)
        # Value
        layer.attention.self.value = LoRALinear(layer.attention.self.value, rank, alpha, dropout)
        
        # Unfreeze SoftUnif parameters if they exist
        if hasattr(layer.attention.self, 'type_query'):
            for param in layer.attention.self.type_query.parameters():
                param.requires_grad = True
        if hasattr(layer.attention.self, 'type_key'):
            for param in layer.attention.self.type_key.parameters():
                param.requires_grad = True
        if hasattr(layer.attention.self, 'type_bilinear'):
            layer.attention.self.type_bilinear.requires_grad = True
        if hasattr(layer.attention.self, 'type_bias'):
            layer.attention.self.type_bias.requires_grad = True
        
    # Also unfreeze the classifier head if we want to train it?
    # Usually for classification, we want to train the head.
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"LoRA Applied. Trainable params: {trainable_params} / {all_params} ({100 * trainable_params / all_params:.2f}%)")
    
    return model
