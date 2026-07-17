import sys
from tempfile import TemporaryDirectory
from pathlib import Path

import torch

from clutrr.cli.train import build_arg_parser
from clutrr.data.trua_collator import TruaBatchCollator
from clutrr.models.relation_attention import RelationConditionedEntityAttention
from clutrr.models.trua_core import ENTITY_PATH_ADAPTER, PROPOSITION_EVIDENCE_ADAPTER
from clutrr.training.model_selection import stratified_train_validation_split
from clutrr.training.robustness import _count_reference_transition_hits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from transformer_trua_prop import (
    complete_input_limits,
    select_query_anchor,
    validation_selection_key,
)
from generic_trua_prop import _first_existing


def _dense_hop_scores(edges, unit_count):
    dense = edges["hop_logits"].new_full(edges["hop_logits"].shape[:2] + (unit_count,), -1e4)
    return dense.scatter(2, edges["edge_index"], edges["hop_logits"])


class _MockEncoding(dict):
    def __getattr__(self, key):
        return self[key]


class _RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, text_pair=None, **kwargs):
        texts = list(texts)
        paired = list(text_pair) if text_pair is not None else None
        self.calls.append((texts, paired, kwargs.get("max_length")))
        width = 6
        return _MockEncoding(
            input_ids=torch.ones(len(texts), width, dtype=torch.long),
            attention_mask=torch.ones(len(texts), width, dtype=torch.long),
        )


def _collator_example():
    return {
        "story": "Alice is Bob's mother.",
        "query": ("Alice", "Bob"),
        "target_id": 0,
        "hops": 2,
        "query_indices": (0, 1),
        "path_indices": [0, 1],
        "path_rel_labels": [],
        "num_nodes": 2,
        "node_spans": [(1, 2), (4, 5)],
    }


def test_separate_unit_encoding_keeps_query_out_of_unit_sequence():
    tokenizer = _RecordingTokenizer()
    collator = TruaBatchCollator(
        tokenizer,
        model_type="bert",
        unit_encoding_mode="separate",
    )

    output = collator([_collator_example()])

    assert tokenizer.calls == [
        (["Alice is Bob's mother."], None, 512),
        (["Alice and Bob"], None, 64),
    ]
    assert output["query_input_ids"] is not None
    assert output["query_attention_mask"] is not None


def test_joint_unit_encoding_preserves_native_sequence_pair_input():
    tokenizer = _RecordingTokenizer()
    collator = TruaBatchCollator(
        tokenizer,
        model_type="bert",
        unit_encoding_mode="joint",
    )

    output = collator([_collator_example()])

    assert tokenizer.calls == [
        (["Alice is Bob's mother."], ["Alice and Bob"], 512),
    ]
    assert output["query_input_ids"] is None
    assert output["query_attention_mask"] is None


def test_unit_encoding_mode_is_exposed_by_training_cli():
    parser = build_arg_parser()
    defaults = parser.parse_args([])
    assert defaults.unit_encoding_mode == "joint"
    assert defaults.sparse_top_k == 0
    assert defaults.force_gold_edges is False
    assert defaults.goal_guidance_mode == "additive"
    separate = parser.parse_args(["--unit_encoding_mode", "separate"])
    assert separate.unit_encoding_mode == "separate"


def test_goal_embedding_changes_step_scores_only_when_enabled():
    torch.manual_seed(7)
    units = torch.randn(2, 5, 12)
    mask = torch.ones(2, 5, dtype=torch.bool)
    goal_a = torch.randn(2, 12)
    goal_b = torch.randn(2, 12)
    layer = RelationConditionedEntityAttention(
        hidden_size=12,
        num_relations=4,
        dropout=0.0,
        top_k=None,
    ).eval()

    _, edges_a = layer(units, mask, goal_embedding=goal_a)
    _, edges_b = layer(units, mask, goal_embedding=goal_b)
    assert not torch.allclose(_dense_hop_scores(edges_a, 5), _dense_hop_scores(edges_b, 5))

    layer.use_goal_guidance = False
    _, edges_a = layer(units, mask, goal_embedding=goal_a)
    _, edges_b = layer(units, mask, goal_embedding=goal_b)
    assert torch.allclose(_dense_hop_scores(edges_a, 5), _dense_hop_scores(edges_b, 5))


def test_modulated_goal_guidance_starts_unguided_and_can_change_scores():
    torch.manual_seed(11)
    units = torch.randn(2, 5, 12)
    mask = torch.ones(2, 5, dtype=torch.bool)
    goal_a = torch.randn(2, 12)
    goal_b = torch.randn(2, 12)
    layer = RelationConditionedEntityAttention(
        hidden_size=12,
        num_relations=4,
        dropout=0.0,
        top_k=None,
        goal_guidance_mode="modulated",
    ).eval()

    assert torch.count_nonzero(layer.q_obj.weight) == 0
    _, initial_a = layer(units, mask, goal_embedding=goal_a)
    _, initial_b = layer(units, mask, goal_embedding=goal_b)
    assert torch.allclose(
        _dense_hop_scores(initial_a, 5),
        _dense_hop_scores(initial_b, 5),
    )

    with torch.no_grad():
        layer.q_obj.weight.copy_(torch.eye(12))
    _, modulated_a = layer(units, mask, goal_embedding=goal_a)
    _, modulated_b = layer(units, mask, goal_embedding=goal_b)
    assert not torch.allclose(
        _dense_hop_scores(modulated_a, 5),
        _dense_hop_scores(modulated_b, 5),
    )


def test_validation_split_is_fixed_and_stratified():
    dataset = list(range(40))
    strata = [(index % 2, index % 4) for index in dataset]
    train_a, validation_a = stratified_train_validation_split(dataset, strata, 0.2, 2027)
    train_b, validation_b = stratified_train_validation_split(dataset, strata, 0.2, 2027)

    assert train_a.indices == train_b.indices
    assert validation_a.indices == validation_b.indices
    assert set(train_a.indices).isdisjoint(validation_a.indices)
    assert sorted(train_a.indices + validation_a.indices) == dataset


def test_proposition_query_anchor_can_be_controlled_independently():
    query = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    shared_anchor = torch.tensor([7.0, 8.0])

    guided = select_query_anchor(query, shared_anchor, use_query_anchor=True)
    unguided = select_query_anchor(query, shared_anchor, use_query_anchor=False)

    assert guided is query
    assert torch.equal(unguided, torch.tensor([[7.0, 8.0], [7.0, 8.0]]))
    assert not torch.equal(unguided[0], query[0])
    assert not torch.equal(unguided[1], query[1])


def test_proposition_selection_uses_evidence_only_as_an_accuracy_tiebreaker():
    higher_accuracy = {"accuracy": 0.9, "evidence_at_1": 0.1}
    lower_accuracy = {"accuracy": 0.8, "evidence_at_1": 1.0}
    same_accuracy_better_evidence = {"accuracy": 0.9, "evidence_at_1": 0.7}

    assert validation_selection_key(higher_accuracy) > validation_selection_key(lower_accuracy)
    assert validation_selection_key(same_accuracy_better_evidence) > validation_selection_key(higher_accuracy)


def test_transition_at_one_counts_each_reference_path_edge():
    edge_index = torch.tensor(
        [
            [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
            [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        ]
    )
    hop_logits = torch.tensor(
        [
            [[0.0, 4.0, 1.0], [5.0, 0.0, 1.0], [0.0, 1.0, 2.0]],
            [[1.0, 0.0, 2.0], [0.0, 2.0, 1.0], [3.0, 0.0, 1.0]],
        ]
    )
    sparse_edges = {
        "edge_index": edge_index,
        "hop_logits": hop_logits,
        "edge_valid": torch.ones_like(edge_index, dtype=torch.bool),
    }
    paths = torch.tensor([[0, 1, 2, -1], [2, 0, -1, -1]])

    hits, total = _count_reference_transition_hits(sparse_edges, paths)

    assert hits == 2
    assert total == 3


def test_adapter_specs_distinguish_paths_from_evidence_sets():
    assert ENTITY_PATH_ADAPTER.supervision_type == "ordered_path"
    assert PROPOSITION_EVIDENCE_ADAPTER.supervision_type == "evidence_set"


def test_proposition_defaults_preserve_complete_paper_inputs():
    assert complete_input_limits("proofwriter", 16, 192) == (32, 512)
    assert complete_input_limits("ruletaker_raw", 24, 192) == (32, 512)
    assert complete_input_limits("prontoqa", 24, 192) == (24, 512)
    assert complete_input_limits("proofwriter", 16, 192, allow_truncation=True) == (16, 192)


def test_dataset_variant_resolution_does_not_mix_existing_directories():
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        standard = root / "depth-3"
        extended = root / "depth-3ext-NatLang"
        standard.mkdir()
        extended.mkdir()

        assert _first_existing([standard, extended]) == standard
