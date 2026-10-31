import pytest

from p24 import study
from p24.io import write_json


@pytest.fixture
def recorded_study(tmp_path, tiny_base):
    """Synthetic records for verifier tests, never an actual study or quality claim."""
    import hashlib
    import subprocess
    from dataclasses import asdict

    from p24.bundle import seal_bundle
    from p24.config import LABELS, StudyConfig
    from p24.contracts import Prediction, Row
    from p24.data import prepare, write_freeze
    from p24.io import canonical, digest_bytes, digest_file
    from p24.metrics import compare
    from p24.sources import fetch_sources

    project = tmp_path / "fixture-project"
    root = project / ".local/study"
    (project / "src").mkdir(parents=True)
    (project / "src/fixture.py").write_text(
        "# independent fixture\n", encoding="utf-8", newline="\n"
    )
    (project / "uv.lock").write_bytes(b"fixture lock\n")
    (project / "pyproject.toml").write_bytes(b"[project]\n")
    config = StudyConfig()
    write_json(project / "config/study.json", asdict(config))
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
        subprocess.run(
            ["git", "-c", "core.autocrlf=true", *args],
            cwd=project,
            check=True,
            capture_output=True,
        )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
    rows = [
        Row(
            f"{split}:{j * count + i:06}",
            split,
            label,
            hashlib.sha256(f"{split}:{label}:{i}".encode()).hexdigest()[:32],
        )
        for split, count in (("train", 64), ("test", 12))
        for j, label in enumerate(LABELS)
        for i in range(count)
    ]
    frozen = prepare(rows, config, {})
    write_freeze(frozen, root)
    identity = {
        "source_hash": digest_bytes(canonical(files)),
        "lock_hash": files["uv.lock"],
        "config_hash": digest_bytes(canonical(asdict(config))),
        "freeze_id": frozen.identity,
    }
    write_json(root / "config.json", asdict(config))
    write_json(
        root / "study.json",
        {
            "schema_version": 1,
            "origin": "actual_public_inputs",
            "source_commit": commit,
            "source_files": files,
            "identity": {**identity, "freeze_id": "unfrozen"},
        },
    )
    fetch_sources(
        config, root, lambda repo, rev, name, kind: b"cc-by-4.0 apache-2.0 independent fixture"
    )
    counts = {
        name: getattr(frozen, name).manifest["counts"] for name in ("train", "validation", "test")
    }
    write_json(
        root / "preparation.json", {"identity": identity, "status": "frozen", "counts": counts}
    )
    sample = [
        row_id
        for label in LABELS
        for row_id in sorted(r.row_id for r in frozen.train.rows if r.label == label)[:10]
    ]
    write_json(
        root / "selection-review.json",
        {
            "accepted": True,
            "reviewer_kind": "assistant",
            "candidate_identity": frozen.identity,
            "sample_row_ids": sample,
        },
    )
    model_id = seal_bundle(tiny_base, root / "model", config).model_id
    write_json(root / "training.json", {"identity": identity, "model_id": model_id})
    predictions = [Prediction(r.row_id, r.label) for r in frozen.test.rows]
    payload = {
        "schema_version": 1,
        "identity": identity,
        "model_id": model_id,
        "baseline": [asdict(p) for p in predictions],
        "encoder": [asdict(p) for p in predictions],
    }
    write_json(root / "predictions.json", payload)
    write_json(root / "metrics.json", compare(frozen.test.rows, predictions, predictions))
    write_json(
        root / "evaluation.json",
        {
            "identity": identity,
            "model_id": model_id,
            "predictions_sha256": digest_file(root / "predictions.json"),
            "metrics_sha256": digest_file(root / "metrics.json"),
        },
    )
    write_json(root / "evaluation.claim.json", {"identity": identity, "model_id": model_id})
    return root, project


@pytest.mark.parametrize(
    "filename", ["evaluation.json", "predictions.json", "evaluation.claim.json", "training.json"]
)
@pytest.mark.parametrize("field", ["identity", "model_id"])
def test_mixed_evaluation_lineage_never_passes(recorded_study, filename, field):
    from p24.acceptance import verify_acceptance
    from p24.io import digest_file, read_json

    root, project = recorded_study
    assert verify_acceptance(root, project)["cases"]["A5"]["status"] == "PASS"
    record = read_json(root / filename)
    record[field] = (
        {**record[field], "source_hash": "different study"}
        if field == "identity"
        else "different model"
    )
    write_json(root / filename, record)
    if filename == "predictions.json":
        evaluation = read_json(root / "evaluation.json")
        evaluation["predictions_sha256"] = digest_file(root / filename)
        write_json(root / "evaluation.json", evaluation)
    assert verify_acceptance(root, project)["cases"]["A5"]["status"] == "FAIL"


@pytest.mark.parametrize(
    "field,value",
    [("candidate_identity", "another freeze"), ("sample_row_ids", ["not the reviewed sample"])],
)
def test_unbound_preparation_review_cannot_pass_provenance(recorded_study, field, value):
    from p24.acceptance import verify_acceptance
    from p24.io import read_json

    root, project = recorded_study
    assert verify_acceptance(root, project)["cases"]["A1"]["status"] == "PASS"
    record = read_json(root / "selection-review.json")
    record[field] = value
    write_json(root / "selection-review.json", record)
    assert verify_acceptance(root, project)["cases"]["A1"]["status"] == "FAIL"


def test_measured_snapshot_must_match_its_preserved_git_source(recorded_study):
    from p24.acceptance import verify_acceptance
    from p24.io import read_json

    root, project = recorded_study
    assert verify_acceptance(root, project)["cases"]["A5"]["status"] == "PASS"
    record = read_json(root / "study.json")
    record["source_commit"] = "0" * 40
    write_json(root / "study.json", record)
    assert verify_acceptance(root, project)["cases"]["A5"]["status"] == "FAIL"


def test_acceptance_cannot_initialize_a_missing_mlflow_database(recorded_study):
    import shutil

    from p24.acceptance import verify_acceptance
    from p24.io import read_json

    root, project = recorded_study
    training = read_json(root / "training.json")
    write_json(
        root / "readback.json",
        {
            "identity": training["identity"],
            "model_id": training["model_id"],
            "run_id": "missing",
            "download_path": "downloaded",
        },
    )
    shutil.copytree(root / "model", root / "downloaded")
    assert verify_acceptance(root, project)["cases"]["A6"]["status"] != "PASS"
    assert not (root / "mlflow").exists(), "Verification must not create tracking state"


def test_missing_actual_evidence_is_not_a_complete_study(tmp_path):
    result = study.verify_acceptance(tmp_path)
    assert result["complete"] is False
    for case in ("A4", "A6", "A7"):
        assert result["cases"][case]["status"] in ("NOT_RUN", "FAIL")


def test_synthetic_fixture_metrics_never_prove_actual_quality(tmp_path):
    write_json(tmp_path / "study.json", {"origin": "software_fixture"})
    write_json(tmp_path / "metrics.json", {"encoder": {"macro_f1": 1.0}})
    result = study.verify_acceptance(tmp_path)
    assert result["complete"] is False
    assert result["cases"]["A1"]["status"] != "PASS"


def test_optimized_verifier_cannot_acquire_sources_for_a_fixture(tmp_path):
    import subprocess
    import sys
    from dataclasses import asdict

    from p24.config import StudyConfig

    write_json(tmp_path / "config.json", asdict(StudyConfig()))
    write_json(tmp_path / "study.json", {"origin": "software_fixture"})
    code = (
        "from pathlib import Path;from p24.study import verify_acceptance;"
        f"r=verify_acceptance(Path({str(tmp_path)!r}));print(r['complete'])"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code], capture_output=True, text=True, check=False, timeout=90
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "sources").exists(), (
        "Read-only verification must never begin acquisition"
    )
