from p24.bundle import seal_bundle
from p24.config import StudyConfig
from p24.contracts import RunIdentity
from p24.tracking import download_bundle, log_run


def test_mlflow_download_matches_original(tiny_base, tmp_path):
    original = tmp_path / "bundle"
    identity = seal_bundle(tiny_base, original, StudyConfig())
    root = tmp_path / "tracking"
    run_id = log_run(
        root,
        "fixture-only",
        RunIdentity("src", "lock", "cfg", "freeze"),
        {"status": "succeeded", "fixture_only": True},
        [original],
    )
    destination = tmp_path / "downloaded"
    downloaded = download_bundle(root, run_id, destination)
    assert downloaded.model_id == identity.model_id
    assert (destination / "model" / "model.safetensors").read_bytes() == (
        original / "model.safetensors"
    ).read_bytes()
    assert (destination / "model" / "vocab.txt").read_bytes() == (
        original / "vocab.txt"
    ).read_bytes()
