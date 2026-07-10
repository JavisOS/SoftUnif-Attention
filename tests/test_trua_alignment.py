import sys
from pathlib import Path

import torch

from clutrr.models.relation_attention import RelationConditionedEntityAttention
from clutrr.training.model_selection import stratified_train_validation_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from transformer_trua_prop import select_query_anchor, validation_selection_key


def _dense_hop_scores(edges, unit_count):
    dense = edges["hop_logits"].new_full(edges["hop_logits"].shape[:2] + (unit_count,), -1e4)
    return dense.scatter(2, edges["edge_index"], edges["hop_logits"])


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


def test_validation_split_is_fixed_and_stratified():
    dataset = list(range(40))
    strata = [(index % 2, index % 4) for index in dataset]
    train_a, validation_a = stratified_train_validation_split(dataset, strata, 0.2, 2027)
    train_b, validation_b = stratified_train_validation_split(dataset, strata, 0.2, 2027)

    assert train_a.indices == train_b.indices
    assert validation_a.indices == validation_b.indices
    assert set(train_a.indices).isdisjoint(validation_a.indices)
    assert sorted(train_a.indices + validation_a.indices) == dataset


def test_proposition_no_goal_uses_a_shared_query_free_anchor():
    query = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    shared_anchor = torch.tensor([7.0, 8.0])

    guided = select_query_anchor(query, shared_anchor, use_goal_guidance=True)
    unguided = select_query_anchor(query, shared_anchor, use_goal_guidance=False)

    assert guided is query
    assert torch.equal(unguided, torch.tensor([[7.0, 8.0], [7.0, 8.0]]))
    assert not torch.equal(unguided[0], query[0])
    assert not torch.equal(unguided[1], query[1])


def test_proposition_selection_uses_evidence_only_as_an_accuracy_tiebreaker():
    higher_accuracy = {"accuracy": 0.9, "trace_top1": 0.1}
    lower_accuracy = {"accuracy": 0.8, "trace_top1": 1.0}
    same_accuracy_better_evidence = {"accuracy": 0.9, "trace_top1": 0.7}

    assert validation_selection_key(higher_accuracy) > validation_selection_key(lower_accuracy)
    assert validation_selection_key(same_accuracy_better_evidence) > validation_selection_key(higher_accuracy)
