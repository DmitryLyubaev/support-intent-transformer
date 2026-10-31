from p24 import study
from p24.io import write_json


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
