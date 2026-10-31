import sqlite3
from dataclasses import asdict
from pathlib import Path

from mlflow import MlflowClient

from p24.bundle import verify_bundle
from p24.contracts import BundleIdentity, RunIdentity


def _client(root: Path) -> MlflowClient:
    root.mkdir(parents=True, exist_ok=True)
    return MlflowClient(tracking_uri=f"sqlite:///{(root / 'tracking.sqlite').resolve().as_posix()}")


def read_run(root: Path, run_id: str) -> dict:
    """Inspect a quiescent existing store without initializing or migrating it."""
    database = (root / "tracking.sqlite").resolve()
    if not database.is_file():
        raise FileNotFoundError("Required retained MLflow database is absent")
    for suffix in ("-wal", "-journal"):
        sidecar = database.with_name(database.name + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise ValueError("MLflow verification requires a quiescent checkpointed database")
    with sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
        row = connection.execute("SELECT status FROM runs WHERE run_uuid = ?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("Recorded MLflow run is absent")
        tags = dict(connection.execute("SELECT key,value FROM tags WHERE run_uuid = ?", (run_id,)))
    return {"status": row[0], "tags": tags}


def log_run(root: Path, purpose: str, identity: RunIdentity, receipt: dict, artifacts) -> str:
    client = _client(root)
    name = "p24-support-intent"
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = client.create_experiment(
            name, artifact_location=(root / "artifacts").resolve().as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    run = client.create_run(experiment_id, tags={"purpose": purpose, **asdict(identity)})
    run_id = run.info.run_id
    client.log_param(run_id, "seed", 42)
    client.log_param(run_id, "purpose", purpose)
    client.log_dict(run_id, {"identity": asdict(identity), "receipt": receipt}, "receipt.json")
    for name, value in receipt.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            client.log_metric(run_id, name, float(value))
    for artifact in artifacts:
        if artifact.is_dir():
            verify_bundle(artifact)
            client.log_artifacts(run_id, str(artifact), artifact_path="model")
        else:
            client.log_artifact(run_id, str(artifact), artifact_path="files")
    client.set_terminated(
        run_id,
        status="FAILED"
        if receipt.get("status") in ("failed", "timeout", "rss_exceeded")
        else "FINISHED",
    )
    return run_id


def download_bundle(root: Path, run_id: str, destination: Path) -> BundleIdentity:
    if destination.exists():
        raise ValueError("Artifact download destination must be new")
    destination.mkdir(parents=True)
    downloaded = Path(_client(root).download_artifacts(run_id, "model", str(destination)))
    return verify_bundle(downloaded)
