from types import SimpleNamespace

import torch
from torch import nn

from scripts import transformer_trua_prop


class _DummyEncoder(nn.Module):
    def __init__(self, hidden_size=12):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_encoder_core_returns_answer_logits_without_selection_scores(monkeypatch):
    monkeypatch.setattr(
        transformer_trua_prop.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: _DummyEncoder(),
    )
    model = transformer_trua_prop.TransformerTruaProp(
        "dummy",
        freeze_encoder=False,
        core_type="encoder",
    )
    logits, scores = model(
        {
            "text_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "text_mask": torch.ones(2, 3, dtype=torch.long),
        }
    )

    assert logits.shape == (2, 2)
    assert scores is None
    assert model.use_goal_guidance is False
    assert model.use_query_anchor is False


def test_validation_selection_key_handles_missing_evidence_metric():
    assert transformer_trua_prop.validation_selection_key(
        {"accuracy": 0.75, "evidence_at_1": None}
    ) == (0.75, 0.0)
