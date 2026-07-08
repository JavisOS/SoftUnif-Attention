"""Shared model/tokenizer factory helpers for training scripts."""

import torch

from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    BertModel,
    BertTokenizerFast,
    DebertaModel,
    DebertaTokenizerFast,
    DebertaV2Model,
    DebertaV2TokenizerFast,
    RobertaModel,
    RobertaTokenizerFast,
)


DECODER_ONLY_MODEL_IDS = {
    "gpt2": "openai-community/gpt2",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-0.6b-base": "Qwen/Qwen3-0.6B-Base",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-1.7b-base": "Qwen/Qwen3-1.7B-Base",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-8b-base": "Qwen/Qwen3-8B-Base",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B",
}


def is_decoder_only_model(model_type: str) -> bool:
    return model_type.lower() in DECODER_ONLY_MODEL_IDS


def _maybe_enable_lora(model, *, use_qlora: bool, lora_r: int, lora_alpha: int, lora_dropout: float):
    if not use_qlora:
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("QLoRA requires `peft`. Please install it (e.g. `pip install peft`).") from exc

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(model, lora_config)


def build_tokenizer(model_type: str, model_name_or_path: str | None = None):
    model_type = model_type.lower()
    if model_name_or_path:
        tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        return tok
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
    if model_type in DECODER_ONLY_MODEL_IDS:
        tok = AutoTokenizer.from_pretrained(DECODER_ONLY_MODEL_IDS[model_type], use_fast=True, trust_remote_code=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        return tok
    raise ValueError(f"Unknown model type: {model_type}")


def build_backbone_model(
    model_type: str,
    *,
    model_name_or_path: str | None = None,
    use_qlora: bool = False,
    load_in_4bit: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    model_type = model_type.lower()
    if model_name_or_path:
        model_kwargs = {"trust_remote_code": True}
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
        model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
        return _maybe_enable_lora(
            model,
            use_qlora=use_qlora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

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
    if model_type in DECODER_ONLY_MODEL_IDS:
        model_kwargs = {}
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
        model_kwargs["trust_remote_code"] = True
        model = AutoModel.from_pretrained(DECODER_ONLY_MODEL_IDS[model_type], **model_kwargs)
        return _maybe_enable_lora(
            model,
            use_qlora=use_qlora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
    raise ValueError(f"Unknown model type: {model_type}")
