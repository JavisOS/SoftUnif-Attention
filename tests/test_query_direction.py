import torch

from clutrr.training.directionality import evaluate_paired_query_direction
from clutrr.utils.query_direction import (
    inverse_relation,
    parse_gender_map,
    reverse_path_relations,
)


def test_inverse_relation_is_gender_aware_and_involutive():
    assert inverse_relation("granddaughter", "male") == "grandfather"
    assert inverse_relation("granddaughter", "female") == "grandmother"
    assert inverse_relation("father", "male") == "son"
    assert inverse_relation("father", "female") == "daughter"

    original = "aunt"
    reversed_relation = inverse_relation(original, "male")
    assert reversed_relation == "nephew"
    assert inverse_relation(reversed_relation, "female") == original


def test_reverse_path_relations_uses_each_original_source_gender():
    names = ["Alex", "Beth", "Chris"]
    genders = {"Alex": "male", "Beth": "female", "Chris": "male"}
    # Beth is Alex's daughter; Chris is Beth's son. The reversed path is
    # Chris -> Beth (mother), then Beth -> Alex (father).
    assert reverse_path_relations(
        ["daughter", "son"],
        [0, 1, 2],
        names,
        genders,
    ) == ["mother", "father"]


def test_gender_parser_handles_full_names_and_rejects_unknown_values():
    assert parse_gender_map("Alex Smith:male,Beth Jones:female,X:unknown") == {
        "Alex Smith": "male",
        "Beth Jones": "female",
    }


class _PredictionFromInput(torch.nn.Module):
    def compute_logits(
        self,
        input_ids,
        attention_mask,
        entity_spans,
        query_indices,
    ):
        logits = torch.full((input_ids.size(0), 3), -10.0)
        logits.scatter_(1, input_ids[:, :1], 10.0)
        return logits, None, None


def _batch(predictions, labels, queries):
    size = len(predictions)
    return {
        "input_ids": torch.tensor(predictions).unsqueeze(1),
        "attention_mask": torch.ones(size, 1, dtype=torch.long),
        "entity_spans": torch.zeros(size, 1, 2, dtype=torch.long),
        "query_indices": torch.zeros(size, 2, dtype=torch.long),
        "labels": torch.tensor(labels),
        "raw_batch": [
            {"story": f"story-{index}", "query": query}
            for index, query in enumerate(queries)
        ],
    }


def test_paired_direction_metrics_exclude_same_label_cases_from_response_rate():
    original = _batch([0, 1, 0], [0, 1, 2], [("a", "b"), ("c", "d"), ("e", "f")])
    reversed_ = _batch([1, 2, 2], [1, 0, 2], [("b", "a"), ("d", "c"), ("f", "e")])

    metrics = evaluate_paired_query_direction(
        _PredictionFromInput(),
        [original],
        [reversed_],
        torch.device("cpu"),
    )

    assert metrics["original_accuracy"] == 2 / 3
    assert metrics["reversed_accuracy"] == 2 / 3
    assert metrics["both_correct_rate"] == 1 / 3
    assert metrics["changed_label_total"] == 2
    assert metrics["same_label_total"] == 1
    assert metrics["prediction_change_rate_changed_labels"] == 1.0
    assert metrics["both_correct_rate_changed_labels"] == 0.5
