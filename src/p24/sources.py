from datetime import UTC, datetime
from pathlib import Path

from p24.config import StudyConfig
from p24.contracts import Row
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


def verify_source_receipts(config: StudyConfig, root: Path) -> dict:
    cache = root / "sources"
    receipt_path = cache / "receipt.json"
    expected = [
        (kind, repo, config.dataset_revision if kind == "dataset" else config.model_revision, name)
        for kind, repo, names in INPUTS
        for name in names
    ]
    if not receipt_path.is_file():
        raise SourceError("Existing source cache lacks complete receipt; retain it")
    receipt = read_json(receipt_path)
    if [(f["kind"], f["repo"], f["revision"], f["filename"]) for f in receipt["files"]] != expected:
        raise SourceError("Source receipt identity mismatch")
    for entry in receipt["files"]:
        path = cache / entry["path"]
        if (
            entry["path"] != f"{entry['kind']}/{entry['filename']}"
            or path.is_symlink()
            or not path.resolve().is_relative_to(cache.resolve())
        ):
            raise SourceError("Source receipt path escaped expected inventory")
        if (
            not path.is_file()
            or digest_file(path) != entry["sha256"]
            or path.stat().st_size != entry["size"]
        ):
            raise SourceError("Source digest/size mismatch")
    return receipt


def fetch_sources(config: StudyConfig, root: Path, transport=None) -> dict:
    cache = root / "sources"
    receipt_path = cache / "receipt.json"
    if cache.exists():
        return verify_source_receipts(config, root)
    expected = [
        (kind, repo, config.dataset_revision if kind == "dataset" else config.model_revision, name)
        for kind, repo, names in INPUTS
        for name in names
    ]
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
                    token=False,
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


def read_source_rows(root: Path) -> list[Row]:
    from datasets import load_dataset

    result = []
    for split in ("train", "test"):
        path = root / "sources" / "dataset" / f"data/{split}-00000-of-00001.parquet"
        dataset = load_dataset(
            "parquet", data_files=str(path), split="train", cache_dir=str(root / "dataset-cache")
        )
        labels = dataset.features["label"].names
        result.extend(
            (
                Row(f"{split}:{i:06}", split, labels[int(row["label"])], row["text"])
                for i, row in enumerate(dataset)
            )
        )
    return result
