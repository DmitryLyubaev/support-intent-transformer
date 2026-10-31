import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from p24.config import LABELS, StudyConfig
from p24.contracts import RunIdentity
from p24.io import canonical, digest_bytes, digest_file, read_json, write_json
from p24.tracking import _client, log_run


@pytest.fixture
def metadata_study(tmp_path):
    """Only synthetic metadata; this fixture never fits or claims model quality."""
    project = tmp_path / "project"
    root = project / ".local/study"
    (project / "src").mkdir(parents=True)
    (project / "src/fixture.py").write_bytes(b"# fixture\n")
    (project / "uv.lock").write_bytes(b"fixture lock\n")
    (project / "pyproject.toml").write_bytes(b"[project]\n")
    config = asdict(StudyConfig())
    write_json(project / "config/study.json", config)
    files = {
        name: digest_file(project / name)
        for name in ("src/fixture.py", "uv.lock", "pyproject.toml", "config/study.json")
    }
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.org",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(["git", *args], cwd=project, check=True, capture_output=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
    identity = RunIdentity(
        digest_bytes(canonical(files)),
        files["uv.lock"],
        digest_bytes(canonical(config)),
        "fixture-freeze",
    )
    write_json(root / "config.json", config)
    write_json(root / "freeze/manifest.json", {"identity": identity.freeze_id})
    write_json(
        root / "study.json",
        {
            "origin": "actual_public_inputs",
            "source_commit": commit,
            "source_files": files,
            "identity": asdict(identity),
            "environment": {"fixture_only": True},
        },
    )
    write_json(root / "sources/receipt.json", {"schema_version": 1, "files": []})
    write_json(
        root / "preparation.json",
        {"identity": asdict(identity), "counts": {"train": {label: 40 for label in LABELS}}},
    )
    run_ids = {}
    for purpose, name in (
        ("baseline", "baseline"),
        ("feasibility", "feasibility"),
        ("measured", "training"),
    ):
        run_id = log_run(root / "mlflow", purpose, identity, {"fixture_only": True}, [])
        run_ids[purpose] = run_id
        write_json(
            root / f"{name}.json",
            {
                "identity": asdict(identity),
                "mlflow_run_id": run_id,
                "model_id": "fixture-model",
                "fit": {
                    "validation_history": [
                        {"epoch": 1, "validation_loss": 1.5, "validation_macro_f1": 0.125}
                    ]
                },
            },
        )
    write_json(
        root / "evaluation.json", {"identity": asdict(identity), "model_id": "fixture-model"}
    )
    write_json(root / "metrics.json", {"schema_version": 1, "fixture_only": True})
    write_json(root / "report.json", {"model_id": "fixture-model", "fixture_only": True})
    return root, project, run_ids


def test_enrichment_logs_existing_lineage_without_rewriting_study(metadata_study, tmp_path):
    from p24.enrichment import enrich_runs

    root, project, ids = metadata_study
    originals = {
        p.relative_to(root): digest_file(p) for p in root.rglob("*.json") if "mlflow" not in p.parts
    }
    result = enrich_runs(root, project)
    assert result["run_ids"] == ids
    client = _client(root / "mlflow")
    experiment = client.get_experiment_by_name("p24-support-intent")
    assert len(client.search_runs([experiment.experiment_id])) == 3
    for run_id in ids.values():
        assert client.get_run(run_id).info.status == "FINISHED"
        path = Path(client.download_artifacts(run_id, "lineage/study.json", str(tmp_path)))
        assert read_json(path)["environment"] == {"fixture_only": True}
    measured = ids["measured"]
    config_path = Path(client.download_artifacts(measured, "lineage/config.json", str(tmp_path)))
    assert read_json(config_path)["batch_size"] == 16
    evaluation = Path(client.download_artifacts(measured, "evaluation/receipt.json", str(tmp_path)))
    assert read_json(evaluation)["model_id"] == "fixture-model"
    assert {p: digest_file(root / p) for p in originals} == originals
    enrich_runs(root, project)
    history = client.get_metric_history(measured, "validation_macro_f1")
    assert [(point.step, point.value) for point in history] == [(1, 0.125)]


@pytest.mark.parametrize("failure", ["identity", "missing_run"])
def test_enrichment_refuses_mixed_records_before_tracking_writes(metadata_study, failure):
    from p24.enrichment import enrich_runs

    root, project, _ = metadata_study
    record = read_json(root / "feasibility.json")
    if failure == "identity":
        record["identity"]["source_hash"] = "wrong-study"
    else:
        record["mlflow_run_id"] = "missing-run"
    write_json(root / "feasibility.json", record)
    store = root / "mlflow"
    before = {
        p.relative_to(store): (digest_file(p), p.stat().st_mtime_ns)
        for p in store.rglob("*")
        if p.is_file()
    }
    with pytest.raises(ValueError, match="lineage|absent"):
        enrich_runs(root, project)
    after = {
        p.relative_to(store): (digest_file(p), p.stat().st_mtime_ns)
        for p in store.rglob("*")
        if p.is_file()
    }
    assert after == before


def test_enrichment_refusal_does_not_create_missing_study_root(tmp_path, monkeypatch):
    from p24 import study

    monkeypatch.setattr(study, "PROJECT_ROOT", tmp_path)
    root = tmp_path / ".local/study"
    with pytest.raises(study.StudyNotReady):
        study.run_stage("enrich", tmp_path / "unused.json", root)
    assert not root.exists()
