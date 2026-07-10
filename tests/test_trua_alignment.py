import torch

from clutrr.models.relation_attention import RelationConditionedEntityAttention
from clutrr.training.model_selection import stratified_train_validation_split


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
