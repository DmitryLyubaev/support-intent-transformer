import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

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


NATIVE_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "download_source.ps1"


def native_download(url: str, destination: Path) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "huggingface.co" or parsed.username:
        raise SourceError("Download requires the approved HTTPS origin")
    runner = shutil.which("pwsh")
    if runner is None:
        raise SourceError("Native verified HTTPS transport requires existing PowerShell 7")
    subprocess.run(
        [
            runner,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(NATIVE_SCRIPT),
            "-Uri",
            url,
            "-OutputPath",
            str(destination.resolve()),
        ],
        check=True,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


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
    if type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1:
        raise SourceError("Unsupported source receipt schema")
    required = {
        "kind",
        "repo",
        "revision",
        "filename",
        "path",
        "url",
        "retrieved_at",
        "sha256",
        "size",
    }
    if not isinstance(receipt.get("files"), list) or any(
        not isinstance(entry, dict) or set(entry) != required for entry in receipt["files"]
    ):
        raise SourceError("Source provenance fields are incomplete")
    if [(f["kind"], f["repo"], f["revision"], f["filename"]) for f in receipt["files"]] != expected:
        raise SourceError("Source receipt identity mismatch")
    for entry in receipt["files"]:
        url = f"https://huggingface.co/{'datasets/' if entry['kind'] == 'dataset' else ''}{entry['repo']}/resolve/{entry['revision']}/{entry['filename']}"
        if entry["url"] != url:
            raise SourceError("Source provenance URL differs from pinned input")
        try:
            timestamp = datetime.fromisoformat(entry["retrieved_at"])
            if timestamp.tzinfo is None or timestamp.utcoffset().total_seconds() != 0:
                raise ValueError("Not UTC")
        except (TypeError, ValueError, AttributeError) as exc:
            raise SourceError("Source retrieval timestamp must be parseable UTC") from exc
        if (
            not isinstance(entry["path"], str)
            or type(entry["size"]) is not int
            or entry["size"] <= 0
            or not isinstance(entry["sha256"], str)
            or len(entry["sha256"]) != 64
        ):
            raise SourceError("Source provenance size/hash/path is invalid")
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
            native_download(
                f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}/resolve/{revision}/{filename}",
                destination.with_suffix(destination.suffix + ".tmp"),
            )
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
    import json

    from pyarrow import parquet

    result = []
    for split in ("train", "test"):
        path = root / "sources" / "dataset" / f"data/{split}-00000-of-00001.parquet"
        source = parquet.ParquetFile(path)
        metadata = json.loads(source.schema_arrow.metadata[b"huggingface"])
        labels = metadata["info"]["features"]["label"]["names"]
        dataset = source.read(columns=["text", "label"]).to_pylist()
        result.extend(
            (
                Row(f"{split}:{i:06}", split, labels[int(row["label"])], row["text"])
                for i, row in enumerate(dataset)
            )
        )
    return result
