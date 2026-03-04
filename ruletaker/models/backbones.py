"""RuleTaker model/tokenizer builders."""

from transformers import AutoModel, AutoTokenizer


def build_ruletaker_tokenizer(model_type: str):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    if "deberta" in model_type.lower():
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    return tokenizer


def build_ruletaker_encoder(model_type: str):
    model_type = model_type.lower()
    if "roberta" in model_type:
        if "base" in model_type or model_type == "roberta":
            return AutoModel.from_pretrained("roberta-base")
        return AutoModel.from_pretrained("roberta-large")
    if "deberta-v3" in model_type:
        return AutoModel.from_pretrained("microsoft/deberta-v3-base")
    return AutoModel.from_pretrained("bert-base-uncased")

