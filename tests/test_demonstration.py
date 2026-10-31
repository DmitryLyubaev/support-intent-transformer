import socket

import psutil

from p24.bundle import seal_bundle
from p24.config import StudyConfig
from p24.demo import run_http_demo


def test_actual_demo_records_route_tokens_http_restart_and_cleanup(tiny_base, tmp_path):
    bundle = tmp_path / "model"
    identity = seal_bundle(tiny_base, bundle, StudyConfig())
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    result = run_http_demo(bundle, tmp_path / "http-evidence", port)
    assert result["model_id"] == identity.model_id
    assert result["route_tokens_match"] is True
    assert result["http_parity"] is True
    assert len(result["processes"]) == 2
    assert len({p["pid"] for p in result["processes"]}) == 2
    assert all(not psutil.pid_exists(p["pid"]) for p in result["processes"])
    assert result["truncation_boundary"] is True
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
