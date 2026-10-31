import hashlib

import pytest

from p24.config import LABELS, StudyConfig, read_config
from p24.contracts import Row
from p24.data import PreparationError, prepare, read_training, write_freeze


def test_duplicate_chain_with_conflicting_labels_is_excluded(rows, config):
    base = hashlib.sha512(b"duplicate-chain").hexdigest()
    middle = "X" + base[1:]
    last = middle[:40] + "Y" + middle[41:80] + "Z" + middle[81:]
    extras = [
        Row("train:000001", "train", LABELS[0], base),
        Row("train:000002", "train", LABELS[0], middle),
        Row("test:000003", "test", LABELS[1], last),
    ]
    result = prepare(rows + extras, config, {})
    assert set(result.audit["exclusions"]["conflicting_labels"]) == {
        "train:000001",
        "train:000002",
        "test:000003",
    }


def test_test_component_never_enters_train_or_validation(rows, config):
    extras = [
        Row("train:000001", "train", LABELS[0], "same message"),
        Row("test:000001", "test", LABELS[0], "SAME   MESSAGE"),
        Row("test:000002", "test", LABELS[0], "same message"),
    ]
    result = prepare(rows + extras, config, {})
    assert "train:000001" not in {r.row_id for r in result.train.rows + result.validation.rows}
    assert "test:000001" in {r.row_id for r in result.test.rows}
    assert "test:000002" not in {r.row_id for r in result.test.rows}


def test_whole_groups_caps_and_minimums(rows, config):
    result = prepare(rows, config, {})
    for label in LABELS:
        assert sum(r.label == label for r in result.train.rows + result.validation.rows) == 8
    groups = [
        set(s.manifest["groups"].values()) for s in (result.train, result.validation, result.test)
    ]
    assert not groups[0] & groups[1]
    assert not (groups[0] & groups[2] | groups[1] & groups[2])
    with pytest.raises(PreparationError, match="minimum"):
        prepare(rows, StudyConfig(), {})


def test_unresolved_privacy_flag_is_excluded_without_relabel(rows, config):
    flagged = Row("train:000001", "train", LABELS[0], "email me at someone@example.org")
    result = prepare(rows + [flagged], config, {})
    assert result.audit["exclusions"]["privacy"] == ["train:000001"]
    original = {r.row_id: r.label for r in rows}
    assert all(
        original[r.row_id] == r.label
        for s in (result.train, result.validation, result.test)
        for r in s.rows
    )


def test_training_loader_never_opens_test(tmp_path, rows, config):
    freeze = write_freeze(prepare(rows, config, {}), tmp_path)
    (freeze / "test.json").unlink()
    train, validation = read_training(freeze)
    assert train.rows and validation.rows
    assert all(r.upstream_split == "train" for r in train.rows + validation.rows)


def test_changed_freeze_is_rejected(tmp_path, rows, config):
    freeze = write_freeze(prepare(rows, config, {}), tmp_path)
    (freeze / "train.json").write_text("[]", encoding="utf-8")
    with pytest.raises(PreparationError, match="digest"):
        read_training(freeze)


def test_actual_config_rejects_fixture_overrides(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"cap": 8}', encoding="utf-8")
    with pytest.raises(ValueError, match="approved"):
        read_config(path)
