import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from p24.api import create_app
from p24.bundle import load_bundle
from p24.io import write_json

EXAMPLES = [
    "My card has not arrived yet.",
    "My card stopped working today.",
    "I lost my card and need a replacement.",
    "A card payment is still pending.",
    "  CAFÉ card  ",
    " ".join(["card"] * 63),
]


def run_http_demo(bundle_dir: Path, evidence_dir: Path, port: int = 8024) -> dict:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    predictor = load_bundle(bundle_dir)
    expected = predictor.predict(EXAMPLES)
    model_id = predictor.identity.model_id
    app = create_app(bundle_dir)
    observed = []
    with TestClient(app) as client:
        original = app.state.predictor.predict

        def capture(texts):
            values = original(texts)
            observed.extend(values)
            return values

        app.state.predictor.predict = capture
        for text, prediction in zip(EXAMPLES, expected):
            response = client.post("/predict", json={"text": text})
            if not (response.status_code == 200 and response.json()["label"] == prediction.label):
                raise ValueError("Evidence check failed")
        if not [r.token_ids for r in observed] == [r.token_ids for r in expected]:
            raise ValueError("Evidence check failed")
    if not (expected[-1].truncated and len(expected[-1].token_ids) == 64):
        raise ValueError("Evidence check failed")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
    processes, predictions_count, rejected_count = ([], 0, 0)
    base = f"http://127.0.0.1:{port}"
    for index in range(2):
        with (evidence_dir / f"server-{index}.log").open("wb") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "p24.api", "--bundle", str(bundle_dir), "--port", str(port)],
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            try:
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(
                            "Owned API exited before readiness; inspect retained server log"
                        )
                    try:
                        health = httpx.get(base + "/health", timeout=0.5)
                        if health.status_code == 200:
                            if not health.json()["model_id"] == model_id:
                                raise ValueError("Evidence check failed")
                            break
                    except httpx.RequestError:
                        pass
                    time.sleep(0.1)
                else:
                    raise RuntimeError("Owned API readiness deadline exceeded")
                for text, prediction in zip(EXAMPLES, expected):
                    response = httpx.post(base + "/predict", json={"text": text}, timeout=5)
                    if not (
                        response.status_code == 200
                        and response.json()
                        == {
                            "label": prediction.label,
                            "model_id": model_id,
                            "truncated": prediction.truncated,
                        }
                    ):
                        raise ValueError("Evidence check failed")
                    predictions_count += 1
                for body in ({"text": 4}, {"text": " "}, {"text": "x" * 2049}):
                    if not httpx.post(base + "/predict", json=body, timeout=5).status_code in (
                        400,
                        422,
                    ):
                        raise ValueError("Evidence check failed")
                    rejected_count += 1
                if not httpx.post(
                    base + "/predict",
                    content=b"{",
                    headers={"content-type": "application/json"},
                    timeout=5,
                ).status_code in (400, 422):
                    raise ValueError("Evidence check failed")
                rejected_count += 1
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=10)
                processes.append(
                    {
                        "pid": process.pid,
                        "stopped": process.poll() is not None,
                        "exit_code": process.returncode,
                    }
                )
    result = {
        "model_id": model_id,
        "route_tokens_match": True,
        "http_parity": True,
        "truncation_boundary": True,
        "processes": processes,
        "prediction_requests": predictions_count,
        "rejected_requests": rejected_count,
        "route_token_ids": [list(r.token_ids) for r in observed],
        "examples_origin": "independently authored demonstration strings, not quality references",
    }
    write_json(evidence_dir / "receipt.json", result)
    return result
