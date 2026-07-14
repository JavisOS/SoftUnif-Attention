import torch

from clutrr.models.controlled_unit_baselines import (
    ContentSelfAttentionCore,
    MacStyleUnitCore,
    RcaStyleUnitCore,
)


def _inputs():
    torch.manual_seed(7)
    units = torch.randn(2, 5, 16)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, False]]
    )
    query_a = torch.randn(2, 16)
    query_b = query_a + 0.75
    return units, mask, query_a, query_b


def test_all_controlled_cores_preserve_shape_and_mask_padding():
    units, mask, query, _ = _inputs()
    cores = [
        ContentSelfAttentionCore(16, num_heads=4, dropout=0.0),
        MacStyleUnitCore(16, steps=3, dropout=0.0),
        RcaStyleUnitCore(16, dropout=0.0),
    ]
    for core in cores:
        core.eval()
        output = core(units, mask, query)
        assert output.shape == units.shape
        assert torch.isfinite(output).all()
        assert torch.count_nonzero(output[~mask]) == 0


def test_content_attention_is_goal_independent_but_controlled_cores_are_not():
    units, mask, query_a, query_b = _inputs()
    content = ContentSelfAttentionCore(16, num_heads=4, dropout=0.0).eval()
    mac = MacStyleUnitCore(16, steps=3, dropout=0.0).eval()
    rca = RcaStyleUnitCore(16, dropout=0.0).eval()

    assert torch.allclose(content(units, mask, query_a), content(units, mask, query_b))
    assert not torch.allclose(mac(units, mask, query_a), mac(units, mask, query_b))
    assert not torch.allclose(rca(units, mask, query_a), rca(units, mask, query_b))
