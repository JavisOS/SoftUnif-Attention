"""Shared model/tokenizer factory helpers for training scripts."""

from transformers import (
    AutoModel,
    AutoTokenizer,
    BertModel,
    BertTokenizerFast,
    DebertaModel,
    DebertaTokenizerFast,
    DebertaV2Model,
    DebertaV2TokenizerFast,
    RobertaModel,
    RobertaTokenizerFast,
)


def build_tokenizer(model_type: str):
    model_type = model_type.lower()
    if model_type == "bert":
        return BertTokenizerFast.from_pretrained("bert-base-uncased")
    if model_type == "roberta":
        return RobertaTokenizerFast.from_pretrained("roberta-base")
    if model_type == "roberta-large":
        return RobertaTokenizerFast.from_pretrained("roberta-large")
    if model_type == "deberta":
        return DebertaTokenizerFast.from_pretrained("microsoft/deberta-base")
    if model_type == "deberta-v3":
        return DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")
    if model_type == "deberta-v3-large":
        return DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-large")
    if model_type == "modernbert":
        try:
            return AutoTokenizer.from_pretrained(
                "answerdotai/ModernBERT-base",
                use_fast=True,
                trust_remote_code=True,
            )
        except TypeError:
            return AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", use_fast=True)
    raise ValueError(f"Unknown model type: {model_type}")


def build_backbone_model(model_type: str):
    model_type = model_type.lower()
    if model_type == "bert":
        return BertModel.from_pretrained("bert-base-uncased")
    if model_type == "roberta":
        return RobertaModel.from_pretrained("roberta-base")
    if model_type == "roberta-large":
        return RobertaModel.from_pretrained("roberta-large")
    if model_type == "deberta":
        return DebertaModel.from_pretrained("microsoft/deberta-base")
    if model_type == "deberta-v3":
        return DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
    if model_type == "deberta-v3-large":
        return DebertaV2Model.from_pretrained("microsoft/deberta-v3-large")
    if model_type == "modernbert":
        try:
            return AutoModel.from_pretrained("answerdotai/ModernBERT-base", trust_remote_code=True)
        except TypeError:
            return AutoModel.from_pretrained("answerdotai/ModernBERT-base")
    raise ValueError(f"Unknown model type: {model_type}")
