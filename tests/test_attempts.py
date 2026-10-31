import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import psutil
import pytest

from p24 import attempts
from p24.contracts import RunIdentity

IDENTITY = RunIdentity("source", "lock", "config", "freeze")


def test_exclusive_claim_survives_crash_and_changed_identity(tmp_path):
    first = attempts.claim_attempt(tmp_path, "measured", IDENTITY)
    assert first.exists()
    with pytest.raises(FileExistsError):
        attempts.claim_attempt(tmp_path, "measured", replace(IDENTITY, source_hash="different"))


def test_concurrent_commands_get_only_one_claim(tmp_path):
    def claim():
        try:
            attempts.claim_attempt(tmp_path, "measured", IDENTITY)
            return "claimed"
        except FileExistsError:
            return "denied"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: claim(), range(2))) == ["claimed", "denied"]


def test_supervisor_kills_only_owned_tree(tmp_path):
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        claim = attempts.claim_attempt(tmp_path, "measured", IDENTITY)
        child_pid = tmp_path / "child.pid"
        code = (
            "import subprocess,sys,time; from pathlib import Path; "
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
            f"Path({str(child_pid)!r}).write_text(str(p.pid)); time.sleep(60)"
        )
        receipt = attempts.supervise([sys.executable, "-c", code], claim, 2, 4 * 1024**3)
        assert receipt.status == "timeout"
        assert receipt.peak_rss_bytes > 0
        assert sentinel.poll() is None
        assert child_pid.exists()
        assert not psutil.pid_exists(int(child_pid.read_text()))
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=5)


def test_memory_limit_is_a_terminal_failure(tmp_path, monkeypatch):
    claim = attempts.claim_attempt(tmp_path, "measured", IDENTITY)
    monkeypatch.setattr(attempts, "_tree_rss", lambda process: 4097)
    receipt = attempts.supervise(
        [sys.executable, "-c", "import time;time.sleep(60)"], claim, 5, 4096
    )
    assert receipt.status == "rss_exceeded"
    assert receipt.output_dir is None


@pytest.mark.parametrize("code", ["import time;time.sleep(0.5)", "raise SystemExit(4)"])
def test_missing_or_failed_terminal_receipt_is_failed(tmp_path, code):
    claim = attempts.claim_attempt(tmp_path, "measured", IDENTITY)
    receipt = attempts.supervise([sys.executable, "-c", code], claim, 5, 4 * 1024**3)
    assert receipt.status == "failed"
    assert receipt.output_dir is None


def test_completion_after_deadline_cannot_be_accepted(tmp_path, monkeypatch):
    import json
    import time
    from dataclasses import asdict

    claim = attempts.claim_attempt(tmp_path, "measured", IDENTITY)
    folder = tmp_path / "measured"
    payload = json.dumps(
        {
            "complete": True,
            "identity": asdict(IDENTITY),
            "purpose": "measured",
            "completed_steps": 1,
        }
    )
    code = (
        "import time;from pathlib import Path;time.sleep(0.2);"
        f"p=Path({str(folder)!r});(p/'staged').mkdir();"
        "(p/'staged'/'model.safetensors').write_bytes(b'fixture');"
        f"(p/'worker-receipt.json').write_text({payload!r})"
    )
    original_sleep = time.sleep
    monkeypatch.setattr(attempts.time, "sleep", lambda seconds: original_sleep(1.5))
    receipt = attempts.supervise([sys.executable, "-c", code], claim, 1, 4 * 1024**3)
    assert receipt.status == "timeout"
    assert receipt.output_dir is None
