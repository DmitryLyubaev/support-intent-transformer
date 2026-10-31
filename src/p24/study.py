import math
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import psutil

from p24.config import LABELS, StudyConfig, read_config
from p24.contracts import Prediction, RunIdentity
from p24.data import prepare, read_split, read_training, write_freeze
from p24.io import canonical, digest_bytes, digest_file, read_json, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class StudyNotReady(ValueError):
    pass


class CapacityWait(StudyNotReady):
    pass


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def source_map() -> dict:
    files = [
        p for folder in ("src", "scripts", "tests") for p in (PROJECT_ROOT / folder).rglob("*.py")
    ]
    files += [PROJECT_ROOT / name for name in ("pyproject.toml", "uv.lock", "config/study.json")]
    return {p.relative_to(PROJECT_ROOT).as_posix(): digest_file(p) for p in sorted(files)}


def current_identity(config: StudyConfig, root: Path) -> RunIdentity:
    frozen = root / "freeze" / "manifest.json"
    return RunIdentity(
        digest_bytes(canonical(source_map())),
        digest_file(PROJECT_ROOT / "uv.lock"),
        digest_bytes(canonical(asdict(config))),
        read_json(frozen)["identity"] if frozen.exists() else "unfrozen",
    )


def capacity(root: Path, stage: str) -> dict:
    result = {
        "time": now(),
        "stage": stage,
        "free_ram_bytes": psutil.virtual_memory().available,
        "free_disk_bytes": shutil.disk_usage(root).free,
    }
    with (root / "capacity.jsonl").open("ab") as stream:
        stream.write(canonical(result) + b"\n")
    if result["free_ram_bytes"] < 6 * 1024**3 or result["free_disk_bytes"] < 10 * 1024**3:
        raise CapacityWait(
            f"Capacity wait: RAM {result['free_ram_bytes'] / 1024**3:.2f} GiB, disk {result['free_disk_bytes'] / 1024**3:.2f} GiB; no attempt consumed"
        )
    return result


def _exclusive(root: Path, name: str, payload: dict) -> Path:
    path = root / f"{name}.claim.json"
    with path.open("xb") as stream:
        stream.write(canonical({"schema_version": 1, "time": now(), **payload}))
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _same(record: dict, identity: RunIdentity) -> None:
    if record.get("identity") != asdict(identity):
        raise StudyNotReady("Saved study lineage differs from current source/config/lock/freeze")


def _snapshot_valid(root: Path, identity: RunIdentity, verify_inputs: bool = True) -> dict:
    metadata = read_json(root / "study.json")
    if metadata.get("origin") != "actual_public_inputs":
        raise StudyNotReady("Software fixture cannot become actual study evidence")
    if verify_inputs:
        from p24.sources import verify_source_receipts

        verify_source_receipts(read_config(root / "config.json"), root)
    for field in ("source_hash", "lock_hash", "config_hash"):
        if metadata["identity"][field] != getattr(identity, field):
            raise StudyNotReady("Acquisition source/config/lock lineage changed")
    return metadata


def _require_prepared(root: Path, identity: RunIdentity) -> dict:
    try:
        result = read_json(root / "preparation.json")
    except FileNotFoundError as exc:
        raise StudyNotReady("Dataset preparation/review/freeze is not complete") from exc
    _same(result, identity)
    return result


def require_evaluation(root: Path, identity: RunIdentity) -> dict:
    from p24.bundle import verify_bundle
    from p24.encoder import validate_adaptation

    try:
        record = read_json(root / "training.json")
    except FileNotFoundError as exc:
        raise StudyNotReady("A complete measured fit is required before final test") from exc
    _same(record, identity)
    parent = read_json(root / "attempts" / "measured.receipt.json")
    worker = read_json(root / "attempts" / "measured" / "worker-receipt.json")
    _same(parent, identity)
    _same(worker, identity)
    if (
        parent["status"] != "succeeded"
        or worker.get("complete") is not True
        or worker.get("epochs_completed") != 3
    ):
        raise StudyNotReady("A complete three-epoch measured fit is required")
    validate_adaptation(worker)
    bundle = verify_bundle(root / "model")
    if record["model_id"] != bundle.model_id:
        raise StudyNotReady("Selected model bundle identity changed")
    return record


def claim_evaluation(root: Path, identity: RunIdentity, model_id: str) -> Path:
    return _exclusive(root, "evaluation", {"identity": asdict(identity), "model_id": model_id})


def quality_report_valid(report):
    try:
        count = report["count"]
        for arm in ("baseline", "encoder"):
            metrics = report[arm]
            if (
                count <= 0
                or metrics["count"] != count
                or len(metrics["per_class"]) != 6
                or (sum(sum(row) for row in metrics["confusion_matrix"]) != count)
                or (sum(c["support"] for c in metrics["per_class"]) != count)
                or (not math.isfinite(metrics["macro_f1"]))
                or (not 0 <= metrics["macro_f1"] <= 1)
            ):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _prepare_stage(root: Path, config: StudyConfig, identity: RunIdentity) -> dict:
    dispositions_path = root / "privacy-dispositions.json"
    dispositions = read_json(dispositions_path) if dispositions_path.exists() else {}
    prepared = prepare(_source_rows(root), config, dispositions)
    sample = [
        asdict(row)
        for label in LABELS
        for row in sorted(
            (r for r in prepared.train.rows if r.label == label), key=lambda r: r.row_id
        )[:10]
    ]
    preview = {
        "candidate_identity": prepared.identity,
        "sample": sample,
        "audit": prepared.audit,
        "counts": {
            name: getattr(prepared, name).manifest["counts"]
            for name in ("train", "validation", "test")
        },
    }
    write_json(root / "previews" / f"{prepared.identity}.json", preview)
    write_json(root / "preparation-preview.json", preview)
    review_path = root / "selection-review.json"
    if not review_path.exists():
        return {
            "status": "review_required",
            "candidate_identity": prepared.identity,
            "sample_count": len(sample),
            "preview": str(root / "preparation-preview.json"),
        }
    review = read_json(review_path)
    if (
        review.get("candidate_identity") != prepared.identity
        or review.get("reviewer_kind") != "assistant"
        or set(review.get("sample_row_ids", [])) != {r["row_id"] for r in sample}
        or (review.get("accepted") is not True)
    ):
        raise StudyNotReady("Preparation sample review does not match current candidate")
    write_freeze(prepared, root)
    result = {
        "status": "frozen",
        "identity": asdict(current_identity(config, root)),
        "counts": preview["counts"],
        "review": review,
        "time": now(),
    }
    write_json(root / "preparation.json", result)
    return result


def _encoder_stage(
    root: Path, config_path: Path, config: StudyConfig, identity: RunIdentity, purpose: str
) -> dict:
    from p24.attempts import claim_attempt, supervise
    from p24.bundle import seal_bundle
    from p24.encoder import validate_adaptation
    from p24.tracking import log_run

    baseline = read_json(root / "baseline.json")
    _same(baseline, identity)
    if baseline["status"] != "succeeded":
        raise StudyNotReady("Baseline did not succeed")
    if purpose == "measured":
        feasibility = read_json(root / "feasibility.json")
        _same(feasibility, identity)
        if feasibility.get("gate_passed") is not True:
            raise StudyNotReady("Feasibility gate did not pass")
    claim = claim_attempt(root / "attempts", purpose, identity)
    parent = supervise(
        [
            sys.executable,
            "-m",
            "p24.cli",
            "_worker",
            "--purpose",
            purpose,
            "--config",
            str(config_path),
            "--root",
            str(root),
        ],
        claim,
        600 if purpose == "feasibility" else 3600,
        4 * 1024**3,
    )
    result = asdict(parent)
    result["output_dir"] = str(parent.output_dir) if parent.output_dir else None
    if parent.status != "succeeded":
        result["mlflow_run_id"] = log_run(root / "mlflow", purpose, identity, result, [])
        write_json(root / f"{purpose}-failure.json", result)
        raise RuntimeError(f"{purpose} attempt ended {parent.status}; slot remains consumed")
    worker = read_json(root / "attempts" / purpose / "worker-receipt.json")
    validate_adaptation(worker)
    result["fit"] = worker
    if purpose == "feasibility":
        train, _ = read_training(root / "freeze")
        projection = (
            3 * math.ceil(len(train.rows) / 16) * parent.elapsed_seconds / parent.completed_steps
        )
        result["projected_seconds"] = projection
        result["gate_passed"] = (
            0 < parent.completed_steps <= 50
            and projection <= 2700
            and (parent.elapsed_seconds <= 600)
            and (parent.peak_rss_bytes <= 4 * 1024**3)
        )
        result["mlflow_run_id"] = log_run(root / "mlflow", purpose, identity, result, [])
        weights = parent.output_dir / "model.safetensors"
        result["discarded_weights_sha256"] = digest_file(weights)
        weights.unlink()
        write_json(root / "feasibility.json", result)
        if not result["gate_passed"]:
            raise RuntimeError("Feasibility projection/resource gate failed; no measured launch")
    else:
        if worker["epochs_completed"] != 3 or worker["selected_epoch"] != 3:
            raise RuntimeError("Measured fit did not complete/select the fixed third epoch")
        first = read_json(root / "feasibility.json")["fit"]["encoder_changes"]
        if [c["before_sha256"] for c in first] != [
            c["before_sha256"] for c in worker["encoder_changes"]
        ]:
            raise RuntimeError("Measured initialization differs from pinned feasibility base")
        bundle = seal_bundle(parent.output_dir, root / "model", config)
        result["model_id"] = bundle.model_id
        result["mlflow_run_id"] = log_run(
            root / "mlflow", purpose, identity, result, [root / "model"]
        )
        write_json(root / "training.json", result)
    return result


def _evaluate_stage(root: Path, identity: RunIdentity, training: dict) -> dict:
    from p24.baseline import predict_baseline
    from p24.bundle import load_bundle
    from p24.metrics import compare, error_sample

    claim_evaluation(root, identity, training["model_id"])
    test = read_split(root / "freeze", "test")
    baseline = predict_baseline(root / "baseline" / "baseline.joblib", test.rows)
    predictor = load_bundle(root / "model")
    encoder, truncated = ([], 0)
    for start in range(0, len(test.rows), 16):
        batch = test.rows[start : start + 16]
        results = predictor.predict([r.text for r in batch])
        encoder.extend(
            (Prediction(row.row_id, result.label) for row, result in zip(batch, results))
        )
        truncated += sum(r.truncated for r in results)
    predictions = {
        "schema_version": 1,
        "identity": asdict(identity),
        "model_id": training["model_id"],
        "baseline": [asdict(p) for p in baseline],
        "encoder": [asdict(p) for p in encoder],
    }
    write_json(root / "predictions.json", predictions)
    metrics = compare(test.rows, baseline, encoder)
    if not quality_report_valid(metrics):
        raise RuntimeError("Evaluation metrics/population failed validation")
    write_json(root / "metrics.json", metrics)
    result = {
        "status": "succeeded",
        "identity": asdict(identity),
        "model_id": training["model_id"],
        "time": now(),
        "count": len(test.rows),
        "predictions_sha256": digest_file(root / "predictions.json"),
        "metrics_sha256": digest_file(root / "metrics.json"),
        "test_truncated": truncated,
        "error_sample": {
            "baseline": error_sample(test.rows, baseline),
            "encoder": error_sample(test.rows, encoder),
        },
    }
    write_json(root / "evaluation.json", result)
    return result


def run_stage(stage: str, config: Path, root: Path) -> dict:
    allowed = {
        "acquire",
        "prepare",
        "baseline",
        "feasibility",
        "train",
        "evaluate",
        "readback",
        "serve-check",
        "report",
    }
    if stage not in allowed:
        raise ValueError("Unknown explicit study stage")
    root = root.resolve()
    if root != (PROJECT_ROOT / ".local" / "study").resolve():
        raise ValueError(
            "Actual study root must be this project's owned .local/study; no budget-reset roots"
        )
    root.mkdir(parents=True, exist_ok=True)
    cfg = read_config(config)
    identity = current_identity(cfg, root)
    if stage == "report":
        return write_report(root)
    if (root / "stop.json").exists():
        raise StudyNotReady(
            "Study is stopped; preserve evidence and obtain a revised allowance/design"
        )
    training = require_evaluation(root, identity) if stage == "evaluate" else None
    capacity(root, stage)
    if stage == "acquire":
        from p24.sources import fetch_sources

        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if dirty:
            raise StudyNotReady("Commit/verify clean executable source before actual acquisition")
        if not (root / "study.json").exists():
            metadata = {
                "schema_version": 1,
                "origin": "actual_public_inputs",
                "identity": asdict(identity),
                "source_files": source_map(),
                "source_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip(),
                "acquisition_started": now(),
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "packages": {
                        name: version(name)
                        for name in ("torch", "transformers", "datasets", "scikit-learn", "mlflow")
                    },
                },
            }
            write_json(root / "study.json", metadata)
            write_json(root / "config.json", asdict(cfg))
        _snapshot_valid(root, identity, verify_inputs=False)
        return fetch_sources(cfg, root)
    _snapshot_valid(root, identity)
    try:
        if stage == "prepare":
            return _prepare_stage(root, cfg, identity)
        _require_prepared(root, identity)
        if stage == "baseline":
            from p24.baseline import fit_baseline
            from p24.tracking import log_run

            _exclusive(root, "baseline", {"identity": asdict(identity)})
            train, _ = read_training(root / "freeze")
            artifact = fit_baseline(train, cfg, root / "baseline" / "baseline.joblib")
            result = {
                "status": "succeeded",
                "identity": asdict(identity),
                "time": now(),
                "artifact_sha256": digest_file(artifact),
                "training_count": len(train.rows),
            }
            result["mlflow_run_id"] = log_run(
                root / "mlflow",
                "baseline",
                identity,
                result,
                [artifact, artifact.with_suffix(".json")],
            )
            write_json(root / "baseline.json", result)
            return result
        if stage in ("feasibility", "train"):
            return _encoder_stage(
                root,
                config.resolve(),
                cfg,
                identity,
                "feasibility" if stage == "feasibility" else "measured",
            )
        if stage == "evaluate":
            return _evaluate_stage(root, identity, training)
        training = require_evaluation(root, identity)
        if stage == "readback":
            from p24.tracking import download_bundle

            bundle = download_bundle(
                root / "mlflow", training["mlflow_run_id"], root / "mlflow-download"
            )
            if bundle.model_id != training["model_id"]:
                raise RuntimeError("Downloaded MLflow model identity differs")
            result = {
                "identity": asdict(identity),
                "model_id": bundle.model_id,
                "time": now(),
                "run_id": training["mlflow_run_id"],
                "download_path": "mlflow-download/model",
            }
            write_json(root / "readback.json", result)
            return result
        if stage == "serve-check":
            from p24.demo import run_http_demo

            result = {
                "identity": asdict(identity),
                **run_http_demo(root / "model", root / "http-evidence"),
            }
            write_json(root / "serving.json", result)
            return result
    except (StudyNotReady, FileExistsError):
        raise
    except Exception as exc:
        write_json(
            root / "stop.json",
            {
                "time": now(),
                "stage": stage,
                "identity": asdict(identity),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    raise ValueError("Stage did not dispatch")


def run_worker(config_path: Path, root: Path, purpose: str) -> None:
    from p24.attempts import append_event
    from p24.encoder import fit_encoder

    root = root.resolve()
    if root != (PROJECT_ROOT / ".local" / "study").resolve():
        raise ValueError("Worker root is not owned actual study")
    config = read_config(config_path)
    identity = current_identity(config, root)
    claim = root / "attempts" / f"{purpose}.claim.json"
    _same(read_json(claim), identity)
    if not claim.with_suffix(".launched").exists():
        raise StudyNotReady("Worker requires a supervising launch claim")
    with claim.with_suffix(".worker-started").open("xb") as marker:
        marker.write(b"worker consumed\n")
    train, validation = read_training(root / "freeze")
    folder = root / "attempts" / purpose
    receipt = fit_encoder(
        train,
        None if purpose == "feasibility" else validation,
        root / "sources" / "model",
        folder / "staged",
        config,
        purpose,
        lambda event: append_event(claim, event),
    )
    write_json(folder / "worker-receipt.json", {**receipt, "identity": asdict(identity)})


from p24.sources import read_source_rows as _source_rows


def verify_acceptance(root):
    from p24.acceptance import verify_acceptance as verify

    return verify(root, PROJECT_ROOT)


def write_report(root: Path) -> dict:
    from p24.reporting import write_report as render

    return render(root, PROJECT_ROOT, now())
