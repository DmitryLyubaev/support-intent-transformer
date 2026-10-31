import json
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

from p24.config import LABELS, read_config
from p24.contracts import Prediction
from p24.data import prepare, read_split, read_training
from p24.io import canonical, digest_bytes, digest_file, read_json
from p24.sources import read_source_rows


def _recorded_snapshot(root: Path, project_root: Path) -> dict:
    """Anchor historical measured bytes independently of later verifier changes."""
    metadata = read_json(root / "study.json")
    if metadata.get("origin") != "actual_public_inputs":
        raise ValueError("Synthetic fixture is not actual provenance")
    commit = metadata["source_commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Invalid measured source commit")
    names = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", commit],
        cwd=project_root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.splitlines()
    expected = {
        name
        for name in names
        if (
            name.startswith(("src/", "scripts/", "tests/")) and Path(name).suffix in (".py", ".ps1")
        )
        or name in ("pyproject.toml", "uv.lock", "config/study.json")
    }
    files = metadata["source_files"]
    if set(files) != expected or not {"pyproject.toml", "uv.lock", "config/study.json"} <= expected:
        raise ValueError("Measured source inventory differs from preserved Git snapshot")
    contents = {}
    for name in sorted(expected):
        contents[name] = subprocess.run(
            ["git", "show", f"{commit}:{name}"],
            cwd=project_root,
            capture_output=True,
            check=True,
        ).stdout
        if digest_bytes(contents[name]) != files[name]:
            raise ValueError("Measured source bytes differ from preserved Git snapshot")
    config = asdict(read_config(root / "config.json"))
    if canonical(json.loads(contents["config/study.json"])) != canonical(config):
        raise ValueError("Recorded config differs from measured source config")
    fields = {
        "source_hash": digest_bytes(canonical(files)),
        "lock_hash": files["uv.lock"],
        "config_hash": digest_bytes(canonical(config)),
    }
    if any(metadata["identity"].get(k) != v for k, v in fields.items()):
        raise ValueError("Recorded source/config/lock identity is inconsistent")
    frozen = read_json(root / "freeze/manifest.json")
    return {"source_commit": commit, "identity": {**fields, "freeze_id": frozen["identity"]}}


def verify_acceptance(root, project_root: Path):
    root = Path(root)
    cases = {}
    snapshot = None

    def identity():
        nonlocal snapshot
        if snapshot is None:
            snapshot = _recorded_snapshot(root, project_root)
        return snapshot["identity"]

    def record(relative):
        result = read_json(root / relative)
        if result.get("identity") != identity():
            raise ValueError(f"Measured study lineage differs: {relative}")
        return result

    def selected_model():
        from p24.bundle import verify_bundle

        training = record("training.json")
        if training["model_id"] != verify_bundle(root / "model").model_id:
            raise ValueError("Selected model identity differs from measured receipt")
        return training["model_id"]

    def check(case, operation):
        try:
            identity()
            detail = operation()
            cases[case] = {"status": "PASS", "detail": detail}
        except FileNotFoundError:
            cases[case] = {"status": "NOT_RUN", "detail": "Required evidence is absent"}
        except Exception as exc:  # noqa: BLE001 - report independent failed checks, never hide them
            cases[case] = {"status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"}

    def provenance():
        from p24.sources import verify_source_receipts

        cfg = read_config(root / "config.json")
        metadata = read_json(root / "study.json")
        if not metadata["origin"] == "actual_public_inputs":
            raise ValueError("Synthetic fixture is not actual provenance")
        if not (root / "sources" / "receipt.json").is_file():
            raise ValueError("No acquisition receipt")
        receipt = verify_source_receipts(cfg, root)
        if not len(receipt["files"]) == 7:
            raise ValueError("Evidence check failed")
        if (
            not "cc-by-4.0"
            in (root / "sources/dataset/README.md").read_text(encoding="utf-8").lower()
        ):
            raise ValueError("Evidence check failed")
        if (
            not "apache-2.0"
            in (root / "sources/model/README.md").read_text(encoding="utf-8").lower()
        ):
            raise ValueError("Evidence check failed")
        review = read_json(root / "selection-review.json")
        preparation = record("preparation.json")
        train, _ = read_training(root / "freeze")
        sample = [
            r.row_id
            for label in LABELS
            for r in sorted((r for r in train.rows if r.label == label), key=lambda r: r.row_id)[
                :10
            ]
        ]
        if not (
            review["accepted"] is True
            and review["reviewer_kind"] == "assistant"
            and review["candidate_identity"] == identity()["freeze_id"]
            and len(review["sample_row_ids"]) == len(sample)
            and set(review["sample_row_ids"]) == set(sample)
            and preparation["status"] == "frozen"
        ):
            raise ValueError("Evidence check failed")
        return "Seven pinned file receipts, licences and recorded assistant sample review"

    def split_check():
        cfg = read_config(root / "config.json")
        preparation = record("preparation.json")
        splits = [read_split(root / "freeze", name) for name in ("train", "validation", "test")]
        if preparation["counts"] != {
            name: s.manifest["counts"] for name, s in zip(("train", "validation", "test"), splits)
        }:
            raise ValueError("Preparation counts differ from frozen splits")
        group_sets = [set(s.manifest["groups"].values()) for s in splits]
        if (
            group_sets[0] & group_sets[1]
            | group_sets[0] & group_sets[2]
            | group_sets[1] & group_sets[2]
        ):
            raise ValueError("Evidence check failed")
        if not all(r.upstream_split == "train" for s in splits[:2] for r in s.rows):
            raise ValueError("Evidence check failed")
        if not all(r.upstream_split == "test" for r in splits[2].rows):
            raise ValueError("Evidence check failed")
        for label in LABELS:
            counts = [sum(r.label == label for r in s.rows) for s in splits]
            if not (
                counts[0] >= 40
                and counts[1] >= 10
                and (counts[2] >= 10)
                and (sum(counts[:2]) <= 160)
            ):
                raise ValueError("Evidence check failed")
        dispositions = (
            read_json(root / "privacy-dispositions.json")
            if (root / "privacy-dispositions.json").exists()
            else {}
        )
        rebuilt = prepare(read_source_rows(root), cfg, dispositions)
        if not rebuilt.identity == read_json(root / "freeze/manifest.json")["identity"]:
            raise ValueError("Evidence check failed")
        return {name: len(s.rows) for name, s in zip(("train", "validation", "test"), splits)}

    def baseline_check():
        receipt = record("baseline.json")
        record("baseline.claim.json")
        if receipt["status"] != "succeeded":
            raise ValueError("Baseline did not succeed")
        import joblib

        from p24.data import normalize

        artifact = root / "baseline/baseline.joblib"
        if not digest_file(artifact) == receipt["artifact_sha256"]:
            raise ValueError("Evidence check failed")
        details = read_json(artifact.with_suffix(".json"))
        train, _ = read_training(root / "freeze")
        if not (
            details["training_row_ids"] == [r.row_id for r in train.rows] and details["converged"]
        ):
            raise ValueError("Evidence check failed")
        pipeline = joblib.load(artifact)
        vectorizer = pipeline.named_steps["tfidf"]
        analyzer = vectorizer.build_analyzer()
        training_terms = {term for row in train.rows for term in analyzer(normalize(row.text))}
        if not set(vectorizer.vocabulary_) <= training_terms:
            raise ValueError("Evidence check failed")
        return {"training_count": len(train.rows), "vocabulary_size": details["vocabulary_size"]}

    def adaptation():
        training = record("training.json")
        selected_model()
        from safetensors.torch import load_file

        from p24.encoder import _tensor_digest, validate_adaptation

        worker = training["fit"]
        parent = record("attempts/measured.receipt.json")
        feasibility = record("feasibility.json")
        for purpose, receipt in (("feasibility", feasibility), ("measured", training)):
            claim = record(f"attempts/{purpose}.claim.json")
            supervised = record(f"attempts/{purpose}.receipt.json")
            saved_worker = record(f"attempts/{purpose}/worker-receipt.json")
            if claim["purpose"] != purpose or canonical(saved_worker) != canonical(receipt["fit"]):
                raise ValueError("Attempt claim/worker differs from measured record")
            if any(supervised[k] != receipt[k] for k in supervised):
                raise ValueError("Attempt parent differs from measured record")
            for marker in ("launched", "worker-started"):
                if not (root / f"attempts/{purpose}.claim.{marker}").is_file():
                    raise FileNotFoundError("Required durable launch marker is absent")
        if not (parent["status"] == "succeeded" and parent["elapsed_seconds"] <= 3600):
            raise ValueError("Evidence check failed")
        if not (parent["peak_rss_bytes"] <= 4 * 1024**3 and feasibility["gate_passed"]):
            raise ValueError("Evidence check failed")
        if not worker["epochs_completed"] == worker["selected_epoch"] == 3:
            raise ValueError("Evidence check failed")
        if not worker["trainable_parameters"] == worker["total_parameters"] > 0:
            raise ValueError("Evidence check failed")
        validate_adaptation(worker)
        base = load_file(root / "sources/model/model.safetensors")
        saved = load_file(root / "model/model.safetensors")
        for change in worker["encoder_changes"]:
            suffix = change["name"].removeprefix("bert.")
            matches = [name for name in base if name.endswith(suffix)]
            if not len(matches) == 1:
                raise ValueError("Evidence check failed")
            before, after = (base[matches[0]], saved[change["name"]])
            if not _tensor_digest(before) == change["before_sha256"]:
                raise ValueError("Evidence check failed")
            if not _tensor_digest(after) == change["after_sha256"]:
                raise ValueError("Evidence check failed")
            if not float((before - after).abs().max()) == change["max_abs_delta"] > 0:
                raise ValueError("Evidence check failed")
        if not len(list((root / "attempts").glob("measured.claim.json"))) == 1:
            raise ValueError("Evidence check failed")
        return {
            "steps": worker["completed_steps"],
            "epochs": 3,
            "seconds": parent["elapsed_seconds"],
            "peak_rss_bytes": parent["peak_rss_bytes"],
        }

    def evaluation():
        result = record("evaluation.json")
        claim = record("evaluation.claim.json")
        from p24.metrics import compare
        from p24.study import quality_report_valid

        if not digest_file(root / "predictions.json") == result["predictions_sha256"]:
            raise ValueError("Evidence check failed")
        if not digest_file(root / "metrics.json") == result["metrics_sha256"]:
            raise ValueError("Evidence check failed")
        predictions = record("predictions.json")
        model_id = selected_model()
        if any(r.get("model_id") != model_id for r in (result, claim, predictions)):
            raise ValueError("Evaluation claim/predictions/model identities differ")
        rows = read_split(root / "freeze", "test").rows
        rebuilt = compare(
            rows,
            [Prediction(**p) for p in predictions["baseline"]],
            [Prediction(**p) for p in predictions["encoder"]],
        )
        if not canonical(rebuilt) == canonical(read_json(root / "metrics.json")):
            raise ValueError("Evidence check failed")
        if not quality_report_valid(rebuilt):
            raise ValueError("Evidence check failed")
        return {
            "count": len(rows),
            "baseline_macro_f1": rebuilt["baseline"]["macro_f1"],
            "encoder_macro_f1": rebuilt["encoder"]["macro_f1"],
        }

    def lineage():
        receipt = record("readback.json")
        if receipt["model_id"] != selected_model():
            raise ValueError("Readback refers to a different measured model")
        from p24.bundle import verify_bundle
        from p24.tracking import read_run

        original = verify_bundle(root / "model")
        downloaded = verify_bundle(root / receipt["download_path"])
        if not original.model_id == downloaded.model_id == receipt["model_id"]:
            raise ValueError("Evidence check failed")
        if receipt["run_id"] != record("training.json")["mlflow_run_id"]:
            raise ValueError("Readback refers to a different measured MLflow run")
        for purpose, name in (
            ("baseline", "baseline"),
            ("feasibility", "feasibility"),
            ("measured", "training"),
        ):
            saved = record(f"{name}.json")
            run = read_run(root / "mlflow", saved["mlflow_run_id"])
            if not (run["status"] == "FINISHED" and run["tags"]["purpose"] == purpose):
                raise ValueError("MLflow run purpose/status differs from measured study")
            if any(run["tags"].get(k) != v for k, v in identity().items()):
                raise ValueError("MLflow run lineage differs from measured study")
        if not receipt["run_id"]:
            raise ValueError("Evidence check failed")
        return {"model_id": original.model_id, "mlflow_run_id": receipt["run_id"]}

    def serving():
        receipt = record("serving.json")
        from p24.bundle import verify_bundle

        if not receipt["model_id"] == verify_bundle(root / "model").model_id == selected_model():
            raise ValueError("Evidence check failed")
        if not (
            receipt["route_tokens_match"]
            and receipt["http_parity"]
            and receipt["truncation_boundary"]
        ):
            raise ValueError("Evidence check failed")
        if not (len(receipt["processes"]) == 2 and all(p["stopped"] for p in receipt["processes"])):
            raise ValueError("Evidence check failed")
        return {
            "prediction_requests": receipt["prediction_requests"],
            "rejected_requests": receipt["rejected_requests"],
            "owned_processes_stopped": True,
        }

    def finish():
        receipt = read_json(root / "report.json")
        if receipt["model_id"] != selected_model():
            raise ValueError("Report refers to a different measured model")
        if not receipt["metrics_sha256"] == digest_file(root / "metrics.json"):
            raise ValueError("Evidence check failed")
        if set(receipt["documents"]) != {"docs/study-report.md", "docs/model-card.md"}:
            raise ValueError("Report document inventory differs")
        for path, expected in receipt["documents"].items():
            if not digest_file(project_root / path) == expected:
                raise ValueError("Evidence check failed")
        if not record("serving.json")["processes"][-1]["stopped"]:
            raise ValueError("Evidence check failed")
        return "Source-bound report/model card and preserved artifacts; owned API stopped"

    for case, operation in zip(
        (f"A{i}" for i in range(1, 9)),
        (provenance, split_check, baseline_check, adaptation, evaluation, lineage, serving, finish),
    ):
        check(case, operation)
    from p24.study import source_map

    return {
        "schema_version": 1,
        "measured_snapshot": snapshot,
        "verifier_source_hash": digest_bytes(canonical(source_map())),
        "cases": cases,
        "complete": all(c["status"] == "PASS" for c in cases.values()),
    }
