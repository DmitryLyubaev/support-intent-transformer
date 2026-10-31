from datetime import UTC, datetime
from pathlib import Path

from p24.config import StudyConfig
from p24.io import digest_file, read_json, write_json

INPUTS = (
    (
        "dataset",
        "PolyAI/banking77",
        ("README.md", "data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"),
    ),
    (
        "model",
        "google/bert_uncased_L-2_H-128_A-2",
        ("README.md", "config.json", "vocab.txt", "model.safetensors"),
    ),
)


class SourceError(ValueError):
    pass


def fetch_sources(config: StudyConfig, root: Path, transport=None) -> dict:
    cache = root / "sources"
    receipt_path = cache / "receipt.json"
    expected = [
        (kind, repo, config.dataset_revision if kind == "dataset" else config.model_revision, name)
        for kind, repo, names in INPUTS
        for name in names
    ]
    if cache.exists():
        if not receipt_path.is_file():
            raise SourceError("Existing source cache lacks complete receipt; retain it")
        receipt = read_json(receipt_path)
        if [
            (f["kind"], f["repo"], f["revision"], f["filename"]) for f in receipt["files"]
        ] != expected:
            raise SourceError("Source receipt identity mismatch")
        for entry in receipt["files"]:
            path = cache / entry["path"]
            if (
                not path.is_file()
                or digest_file(path) != entry["sha256"]
                or path.stat().st_size != entry["size"]
            ):
                raise SourceError("Source digest/size mismatch")
        return receipt
    cache.mkdir(parents=True)
    receipt = {"schema_version": 1, "files": []}
    for kind, repo, revision, filename in expected:
        destination = cache / kind / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if transport is None:
            from huggingface_hub import hf_hub_download

            source = Path(
                hf_hub_download(
                    repo,
                    filename,
                    repo_type="dataset" if kind == "dataset" else "model",
                    revision=revision,
                    cache_dir=str(root / "hub-cache"),
                )
            )
            import shutil

            shutil.copyfile(source, destination.with_suffix(destination.suffix + ".tmp"))
        else:
            destination.with_suffix(destination.suffix + ".tmp").write_bytes(
                transport(repo, revision, filename, kind)
            )
        destination.with_suffix(destination.suffix + ".tmp").replace(destination)
        receipt["files"].append(
            {
                "kind": kind,
                "repo": repo,
                "revision": revision,
                "filename": filename,
                "path": destination.relative_to(cache).as_posix(),
                "url": f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}/resolve/{revision}/{filename}",
                "retrieved_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "sha256": digest_file(destination),
                "size": destination.stat().st_size,
            }
        )
    write_json(receipt_path, receipt)
    return receipt
