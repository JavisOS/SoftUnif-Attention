from clutrr.data.clutrr_dataset import official_hop_count


def _row(task_name, edges):
    row = [""] * 12
    row[10] = task_name
    row[11] = edges
    return row


def test_official_hop_count_prefers_released_task_name():
    row = _row("task_1.10", "[(0, 1), (1, 2)]")
    assert official_hop_count(row) == 10


def test_official_hop_count_falls_back_to_story_edges():
    row = _row("", "[(0, 1), (1, 2), (2, 3)]")
    assert official_hop_count(row) == 3


def test_official_hop_count_rejects_missing_metadata():
    row = _row("", "not a literal")
    try:
        official_hop_count(row)
    except ValueError as error:
        assert "task name nor story edges" in str(error)
    else:
        raise AssertionError("Expected invalid CLUTRR metadata to raise ValueError")
