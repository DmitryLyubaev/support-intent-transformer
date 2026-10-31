import math
from dataclasses import replace

import pytest

from p24.config import LABELS, StudyConfig
from p24.contracts import PreparedSplit, Row
from p24.encoder import fit_encoder, validate_adaptation


def training_rows():
    return PreparedSplit(
        (
            Row("train:000000", "train", LABELS[0], "where is my card"),
            Row("train:000001", "train", LABELS[1], "my card broken"),
        ),
        {},
    )


def test_actual_optimizer_changes_both_encoder_layers(tiny_base, tmp_path):
    result = fit_encoder(
        training_rows(),
        None,
        tiny_base,
        tmp_path / "out",
        replace(StudyConfig(), epochs=1, batch_size=2),
        "measured",
        lambda e: None,
    )
    assert result["completed_steps"] == 1
    assert result["epochs_completed"] == 1
    assert result["trainable_parameters"] == result["total_parameters"] > 0
    assert math.isfinite(result["last_loss"])
    assert len(result["encoder_changes"]) == 2
    assert all(c["max_abs_delta"] > 0 for c in result["encoder_changes"])
    assert all(c["before_sha256"] != c["after_sha256"] for c in result["encoder_changes"])


def test_fresh_run_reloads_base_not_feasibility_weights(tiny_base, tmp_path):
    config = replace(StudyConfig(), epochs=1, batch_size=2)
    first = fit_encoder(
        training_rows(), None, tiny_base, tmp_path / "a", config, "measured", lambda e: None
    )
    second = fit_encoder(
        training_rows(), None, tiny_base, tmp_path / "b", config, "measured", lambda e: None
    )
    assert [c["before_sha256"] for c in first["encoder_changes"]] == [
        c["before_sha256"] for c in second["encoder_changes"]
    ]
    assert [c["after_sha256"] for c in first["encoder_changes"]] == [
        c["after_sha256"] for c in second["encoder_changes"]
    ]


def test_nonfinite_loss_is_rejected(tiny_base, tmp_path, monkeypatch):
    from torch.nn import functional

    original = functional.cross_entropy

    def nan_loss(*args, **kwargs):
        return original(*args, **kwargs) * float("nan")

    monkeypatch.setattr(functional, "cross_entropy", nan_loss)
    with pytest.raises(ValueError, match="finite"):
        fit_encoder(
            training_rows(),
            None,
            tiny_base,
            tmp_path / "bad",
            replace(StudyConfig(), epochs=1),
            "measured",
            lambda e: None,
        )
    assert not (tmp_path / "bad" / "model.safetensors").exists()


def test_head_only_change_is_rejected():
    with pytest.raises(ValueError, match="encoder"):
        validate_adaptation(
            {
                "encoder_changes": [
                    {"layer": 0, "max_abs_delta": 0},
                    {"layer": 1, "max_abs_delta": 0},
                ]
            }
        )


def test_training_and_validation_truncation_counts_are_separate(tiny_base, tmp_path):
    train = PreparedSplit(
        (
            Row("train:000000", "train", LABELS[0], "hello " * 20),
            Row("train:000001", "train", LABELS[1], "card"),
        ),
        {},
    )
    validation = PreparedSplit(
        (
            Row("train:000002", "train", LABELS[0], "card " * 20),
            Row("train:000003", "train", LABELS[1], "hello " * 20),
        ),
        {},
    )
    result = fit_encoder(
        train,
        validation,
        tiny_base,
        tmp_path / "truncated",
        replace(StudyConfig(), epochs=1, batch_size=2, length=16),
        "measured",
        lambda event: None,
    )
    assert result["training_truncated"] == 1
    assert result.get("validation_truncated") == 2
