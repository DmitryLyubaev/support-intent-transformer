from dataclasses import asdict

import pytest

from p24 import study
from p24.config import LABELS, StudyConfig
from p24.contracts import Prediction, Row, RunIdentity
from p24.io import write_json
from p24.metrics import compare

IDENTITY = RunIdentity("src", "lock", "cfg", "freeze")


def test_final_test_is_inaccessible_before_complete_measured_fit(tmp_path, monkeypatch):
    root = tmp_path / ".local" / "study"
    root.mkdir(parents=True)
    config_path = tmp_path / "study.json"
    write_json(config_path, asdict(StudyConfig()))
    monkeypatch.setattr(study, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(study, "current_identity", lambda config, root: IDENTITY)

    def forbidden(*args, **kwargs):
        raise AssertionError("Final test data was opened before eligibility")

    monkeypatch.setattr(study, "read_split", forbidden)
    with pytest.raises(study.StudyNotReady, match="measured"):
        study.run_stage("evaluate", config_path, root)
    assert not (root / "evaluation.claim.json").exists()


def test_changed_training_lineage_is_rejected(tmp_path):
    write_json(tmp_path / "training.json", {"identity": asdict(IDENTITY), "status": "succeeded"})
    other = RunIdentity("changed", "lock", "cfg", "freeze")
    with pytest.raises(study.StudyNotReady, match="lineage"):
        study.require_evaluation(tmp_path, other)


def test_second_evaluation_claim_never_overwrites_predictions(tmp_path):
    study.claim_evaluation(tmp_path, IDENTITY, "model-id")
    predictions = tmp_path / "predictions.json"
    predictions.write_bytes(b"original predictions")
    with pytest.raises(FileExistsError):
        study.claim_evaluation(tmp_path, IDENTITY, "model-id")
    assert predictions.read_bytes() == b"original predictions"


def test_inferior_encoder_is_valid_when_mechanical_metrics_are_valid():
    rows = [Row(str(i), "test", label, "fixture") for i, label in enumerate(LABELS)]
    baseline = [Prediction(r.row_id, r.label) for r in rows]
    encoder = [Prediction(r.row_id, LABELS[0]) for r in rows]
    report = compare(rows, baseline, encoder)
    assert report["delta_macro_f1"] < 0
    assert study.quality_report_valid(report)
    report["encoder"]["count"] = 999
    assert not study.quality_report_valid(report)


def test_modified_pinned_input_is_rejected_before_fitting(tmp_path):
    from p24.sources import SourceError, fetch_sources

    write_json(tmp_path / "config.json", asdict(StudyConfig()))
    write_json(
        tmp_path / "study.json", {"origin": "actual_public_inputs", "identity": asdict(IDENTITY)}
    )
    receipt = fetch_sources(StudyConfig(), tmp_path, lambda *args: b"fixture source")
    (tmp_path / "sources" / receipt["files"][0]["path"]).write_bytes(b"modified")
    with pytest.raises(SourceError, match="digest"):
        study._snapshot_valid(tmp_path, IDENTITY)
