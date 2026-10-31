import os
import socket
import subprocess
import sys
import time

import httpx

from p24.bundle import load_bundle, seal_bundle
from p24.config import StudyConfig


def test_actual_process_restart_retains_identity(tiny_base, tmp_path):
    bundle = tmp_path / "bundle"
    identity = seal_bundle(tiny_base, bundle, StudyConfig())
    expected = load_bundle(bundle).predict(["hello card"])[0]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    results = []
    for index in range(2):
        with (tmp_path / f"server-{index}.log").open("wb") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "p24.api", "--bundle", str(bundle), "--port", str(port)],
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            try:
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    assert process.poll() is None, "API exited before becoming ready"
                    try:
                        response = httpx.get(base + "/health", timeout=0.5)
                        if response.status_code == 200:
                            break
                    except httpx.RequestError:
                        pass
                    time.sleep(0.1)
                else:
                    raise AssertionError("API startup deadline exceeded")
                assert response.json()["model_id"] == identity.model_id
                prediction = httpx.post(base + "/predict", json={"text": "hello card"}, timeout=5)
                assert prediction.status_code == 200
                assert prediction.json()["label"] == expected.label
                assert prediction.json()["truncated"] == expected.truncated
                results.append(prediction.json())
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=10)
    assert results[0] == results[1]
