from pathlib import Path

from p24.config import LABELS, read_config
from p24.contracts import Prediction
from p24.data import prepare, read_split, read_training
from p24.io import canonical, digest_file, read_json
from p24.sources import read_source_rows


def verify_acceptance(root, project_root: Path):
    root = Path(root)
    cases = {}

    def check(case, operation):
        try:
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
        if not (review["accepted"] and review["reviewer_kind"] == "assistant"):
            raise ValueError("Evidence check failed")
        return "Seven pinned file receipts, licences and recorded assistant sample review"

    def split_check():
        cfg = read_config(root / "config.json")
        splits = [read_split(root / "freeze", name) for name in ("train", "validation", "test")]
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
        receipt = read_json(root / "baseline.json")
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
        training = read_json(root / "training.json")
        from safetensors.torch import load_file

        from p24.encoder import _tensor_digest, validate_adaptation

        worker = training["fit"]
        parent = read_json(root / "attempts/measured.receipt.json")
        feasibility = read_json(root / "feasibility.json")
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
        result = read_json(root / "evaluation.json")
        from p24.metrics import compare
        from p24.study import quality_report_valid

        if not digest_file(root / "predictions.json") == result["predictions_sha256"]:
            raise ValueError("Evidence check failed")
        if not digest_file(root / "metrics.json") == result["metrics_sha256"]:
            raise ValueError("Evidence check failed")
        predictions = read_json(root / "predictions.json")
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
        receipt = read_json(root / "readback.json")
        from p24.bundle import verify_bundle
        from p24.tracking import _client

        original = verify_bundle(root / "model")
        downloaded = verify_bundle(root / receipt["download_path"])
        if not original.model_id == downloaded.model_id == receipt["model_id"]:
            raise ValueError("Evidence check failed")
        run = _client(root / "mlflow").get_run(receipt["run_id"])
        if not (run.info.status == "FINISHED" and run.data.tags["purpose"] == "measured"):
            raise ValueError("Evidence check failed")
        if not all(
            (
                run.data.tags[k] == v
                for k, v in read_json(root / "training.json")["identity"].items()
            )
        ):
            raise ValueError("Evidence check failed")
        return {"model_id": original.model_id, "mlflow_run_id": receipt["run_id"]}

    def serving():
        receipt = read_json(root / "serving.json")
        from p24.bundle import verify_bundle

        if not receipt["model_id"] == verify_bundle(root / "model").model_id:
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
        if not receipt["metrics_sha256"] == digest_file(root / "metrics.json"):
            raise ValueError("Evidence check failed")
        for path, expected in receipt["documents"].items():
            if not digest_file(project_root / path) == expected:
                raise ValueError("Evidence check failed")
        if not read_json(root / "serving.json")["processes"][-1]["stopped"]:
            raise ValueError("Evidence check failed")
        return "Source-bound report/model card and preserved artifacts; owned API stopped"

    for case, operation in zip(
        (f"A{i}" for i in range(1, 9)),
        (provenance, split_check, baseline_check, adaptation, evaluation, lineage, serving, finish),
    ):
        check(case, operation)
    return {
        "schema_version": 1,
        "cases": cases,
        "complete": all(c["status"] == "PASS" for c in cases.values()),
    }
