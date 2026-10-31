"""Explicit metadata-only enrichment of existing, historical local MLflow runs."""

from pathlib import Path

from p24.acceptance import _recorded_snapshot
from p24.io import read_json
from p24.tracking import _client, read_run


def enrich_runs(root: Path, project_root: Path) -> dict:
    snapshot = _recorded_snapshot(root, project_root)
    identity = snapshot["identity"]

    def record(name):
        value = read_json(root / name)
        if value.get("identity") != identity:
            raise ValueError(f"Enrichment lineage differs: {name}")
        return value

    metadata = read_json(root / "study.json")
    config = read_json(root / "config.json")
    sources = read_json(root / "sources/receipt.json")
    preparation = record("preparation.json")
    evaluation = record("evaluation.json")
    metrics = read_json(root / "metrics.json")
    report = read_json(root / "report.json")
    receipts = {
        purpose: record(f"{name}.json")
        for purpose, name in (
            ("baseline", "baseline"),
            ("feasibility", "feasibility"),
            ("measured", "training"),
        )
    }
    if (
        evaluation["model_id"] != receipts["measured"]["model_id"]
        or report["model_id"] != evaluation["model_id"]
    ):
        raise ValueError("Enrichment model lineage differs")
    # Complete read-only preflight before initializing any writing tracking client.
    for purpose, receipt in receipts.items():
        run = read_run(root / "mlflow", receipt["mlflow_run_id"])
        if run["status"] != "FINISHED" or run["tags"].get("purpose") != purpose:
            raise ValueError("Existing run status/purpose differs from enrichment lineage")
        if any(run["tags"].get(k) != v for k, v in identity.items()):
            raise ValueError("Existing MLflow run lineage differs")

    client = _client(root / "mlflow")
    for purpose, receipt in receipts.items():
        run_id = receipt["mlflow_run_id"]
        client.log_dict(run_id, metadata, "lineage/study.json")
        client.log_dict(run_id, config, "lineage/config.json")
        client.log_dict(run_id, sources, "lineage/source-receipts.json")
        client.log_dict(
            run_id,
            {"identity": identity, "counts": preparation["counts"]},
            "lineage/split-summary.json",
        )
        for name, value in config.items():
            client.log_param(run_id, name, value)
        if purpose == "measured":
            for name in ("validation_loss", "validation_macro_f1"):
                existing = {(p.step, p.value) for p in client.get_metric_history(run_id, name)}
                for epoch in receipt["fit"]["validation_history"]:
                    point = (epoch["epoch"], epoch[name])
                    if point not in existing:
                        client.log_metric(run_id, name, point[1], step=point[0])
                        existing.add(point)
            client.log_dict(run_id, evaluation, "evaluation/receipt.json")
            client.log_dict(run_id, metrics, "evaluation/metrics.json")
            client.log_dict(run_id, report, "evaluation/report-receipt.json")
    return {
        "status": "enriched",
        "source_commit": snapshot["source_commit"],
        "run_ids": {purpose: receipt["mlflow_run_id"] for purpose, receipt in receipts.items()},
        "method": "existing-run metadata only; no fit, inference or receipt rewrite",
    }
