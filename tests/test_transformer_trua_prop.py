import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts import generic_trua_prop, transformer_trua_prop


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
    assert model.use_relation_conditioning is False
    assert model.use_aggregation_branch is False
    assert model.use_step_branch is False


def test_validation_selection_key_handles_missing_evidence_metric():
    assert transformer_trua_prop.validation_selection_key(
        {"accuracy": 0.75, "evidence_at_1": None}
    ) == (0.75, 0.0)


def test_proofwriter_classifier_uses_three_output_labels(monkeypatch):
    monkeypatch.setattr(
        transformer_trua_prop.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: _DummyEncoder(),
    )
    model = transformer_trua_prop.TransformerTruaProp(
        "dummy",
        freeze_encoder=False,
        core_type="encoder",
        num_labels=3,
    )
    logits, _ = model(
        {
            "text_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "text_mask": torch.ones(2, 3, dtype=torch.long),
        }
    )

    assert logits.shape == (2, 3)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [(False, 0), (True, 1), ("Unknown", 2)],
)
def test_proofwriter_owa_label_mapping(answer, expected):
    assert generic_trua_prop._proofwriter_label(answer) == expected


def test_proofwriter_unknown_failure_path_is_not_reference_evidence(tmp_path):
    depth_dir = tmp_path / "OWA" / "depth-0"
    depth_dir.mkdir(parents=True)
    record = {
        "id": "theory-1",
        "maxD": 0,
        "triples": {"triple1": {"text": "The cow is blue."}},
        "rules": {
            "rule1": {"text": "If something is blue then it is cold."}
        },
        "questions": {
            "Q1": {
                "question": "The cow is blue.",
                "answer": True,
                "QDep": 0,
                "proofs": "[(triple1)]",
            },
            "Q2": {
                "question": "The cow is not blue.",
                "answer": False,
                "QDep": 0,
                "proofs": "[(triple1)]",
            },
            "Q3": {
                "question": "The cow is warm.",
                "answer": "Unknown",
                "QDep": 0,
                "proofs": "[@0: deepest failure = (rule1 <- FAIL)]",
            },
        },
    }
    (depth_dir / "meta-train.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )

    samples = generic_trua_prop.load_proofwriter(tmp_path, [0], "train")

    assert [sample["label"] for sample in samples] == [1, 0, 2]
    assert samples[0]["trace_labels"] == [1, 0]
    assert samples[1]["trace_labels"] == [1, 0]
    assert samples[2]["trace_labels"] == [0, 0]
