import os
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import psutil

from p24.contracts import AttemptReceipt, RunIdentity
from p24.io import canonical, read_json, write_json


def claim_attempt(root: Path, purpose: str, identity: RunIdentity) -> Path:
    if purpose not in ("feasibility", "measured"):
        raise ValueError("Unknown attempt purpose")
    root.mkdir(parents=True, exist_ok=True)
    claim = root / f"{purpose}.claim.json"
    with claim.open("xb") as stream:
        stream.write(
            canonical(
                {
                    "schema_version": 1,
                    "purpose": purpose,
                    "identity": asdict(identity),
                    "claimed_at": datetime.now(UTC).isoformat(),
                }
            )
        )
        stream.flush()
        os.fsync(stream.fileno())
    append_event(claim, {"event": "claimed"})
    return claim


def append_event(claim: Path, event) -> None:
    with claim.with_suffix(".events.jsonl").open("ab") as stream:
        stream.write(canonical({"time": datetime.now(UTC).isoformat(), **event}) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _stop_owned(process):
    if process is None:
        return
    try:
        targets = process.children(recursive=True) + [process]
    except psutil.NoSuchProcess:
        return
    for target in targets:
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=2)
    for target in alive:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=2)


def supervise(argv, claim: Path, timeout_seconds: int, rss_limit_bytes: int) -> AttemptReceipt:
    entry = read_json(claim)
    with claim.with_suffix(".launched").open("xb") as marker:
        marker.write(b"launch consumed\n")
        marker.flush()
        os.fsync(marker.fileno())
    started = time.monotonic()
    purpose = entry["purpose"]
    identity = RunIdentity(**entry["identity"])
    folder = claim.parent / purpose
    folder.mkdir(exist_ok=True)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    append_event(claim, {"event": "starting"})
    process = None
    peak, status, steps, output_dir = 0, "failed", 0, None
    with (folder / "worker.log").open("wb") as log:
        child = subprocess.Popen(
            argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            process = psutil.Process(child.pid)
        except psutil.NoSuchProcess:
            pass
        append_event(claim, {"event": "started", "pid": child.pid})
        while child.poll() is None:
            peak = max(peak, _tree_rss(process) if process else 0)
            if time.monotonic() - started > timeout_seconds:
                status = "timeout"
                _stop_owned(process)
                break
            if peak > rss_limit_bytes:
                status = "rss_exceeded"
                _stop_owned(process)
                break
            time.sleep(0.2)
        child.wait(timeout=5)
        if time.monotonic() - started > timeout_seconds and status == "failed":
            status = "timeout"
        worker_receipt = folder / "worker-receipt.json"
        if (
            status not in ("timeout", "rss_exceeded")
            and child.returncode == 0
            and worker_receipt.is_file()
        ):
            worker = read_json(worker_receipt)
            if (
                worker.get("complete") is True
                and worker.get("identity") == entry["identity"]
                and worker.get("purpose") == purpose
                and (folder / "staged" / "model.safetensors").is_file()
            ):
                status = "succeeded"
                steps = int(worker["completed_steps"])
                output_dir = folder / "staged"
    receipt = AttemptReceipt(
        purpose, status, steps, time.monotonic() - started, peak, output_dir, identity
    )
    payload = asdict(receipt)
    payload["output_dir"] = str(output_dir) if output_dir else None
    write_json(claim.parent / f"{purpose}.receipt.json", payload)
    append_event(
        claim,
        {
            "event": "finished",
            "status": status,
            "elapsed_seconds": receipt.elapsed_seconds,
            "peak_rss_bytes": peak,
        },
    )
    return receipt


def _tree_rss(process):
    total = 0
    try:
        for item in [process] + process.children(recursive=True):
            try:
                total += item.memory_info().rss
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    return total
