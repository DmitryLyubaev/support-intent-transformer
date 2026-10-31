import shutil

import pytest

from p24.bundle import load_bundle, seal_bundle, verify_bundle
from p24.config import LABELS, StudyConfig


def test_bundle_relocation_keeps_identity_and_actual_inference(tiny_base, tmp_path):
    first = tmp_path / "bundle"
    identity = seal_bundle(tiny_base, first, StudyConfig())
    second = tmp_path / "relocated with spaces"
    shutil.copytree(first, second)
    assert verify_bundle(second).model_id == identity.model_id
    original = load_bundle(first).predict(["hello card"])
    relocated = load_bundle(second).predict(["hello card"])
    assert original == relocated
    assert original[0].token_ids == (2, 5, 6, 3)
    assert original[0].label in LABELS
    assert original[0].truncated is False


@pytest.mark.parametrize(
    "filename", ["vocab.txt", "model.safetensors", "labels.json", "inference.json"]
)
def test_substituted_bundle_files_fail_before_loading(tiny_base, tmp_path, filename):
    destination = tmp_path / "bundle"
    seal_bundle(tiny_base, destination, StudyConfig())
    with (destination / filename).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="digest|size"):
        load_bundle(destination)


def test_missing_or_extra_bundle_file_is_rejected(tiny_base, tmp_path):
    destination = tmp_path / "bundle"
    seal_bundle(tiny_base, destination, StudyConfig())
    (destination / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        verify_bundle(destination)
    (destination / "unexpected.txt").unlink()
    (destination / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="inventory"):
        verify_bundle(destination)
