import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

from p24.config import LABELS, StudyConfig
from p24.contracts import BundleIdentity, Inference
from p24.data import normalize
from p24.io import canonical, digest_bytes, digest_file, read_json, write_json

POLICY = {
    "normalization": "NFC-strip",
    "max_length": 64,
    "do_lower_case": True,
    "truncation": True,
    "padding": "dynamic",
    "tokenizer_class": "BertTokenizerFast",
}


def seal_bundle(staged: Path, destination: Path, config: StudyConfig) -> BundleIdentity:
    if destination.exists():
        raise ValueError("Bundle destination exists; retain original")
    shutil.copytree(staged, destination)
    model_config = read_json(destination / "config.json")
    model_config["id2label"] = {str(i): label for i, label in enumerate(config.labels)}
    model_config["label2id"] = {label: i for i, label in enumerate(config.labels)}
    write_json(destination / "config.json", model_config)
    write_json(destination / "labels.json", list(config.labels))
    write_json(destination / "inference.json", POLICY)
    files = {
        p.relative_to(destination).as_posix(): {"sha256": digest_file(p), "size": p.stat().st_size}
        for p in sorted(destination.rglob("*"))
        if p.is_file()
    }
    payload = {"files": files, "labels": list(config.labels), "inference": POLICY}
    identity = digest_bytes(canonical(payload))
    write_json(
        destination / "manifest.json", {"schema_version": 1, **payload, "model_id": identity}
    )
    return verify_bundle(destination)


def verify_bundle(path: Path) -> BundleIdentity:
    path = path.resolve()
    manifest = read_json(path / "manifest.json")
    payload = {k: manifest[k] for k in ("files", "labels", "inference")}
    if digest_bytes(canonical(payload)) != manifest["model_id"]:
        raise ValueError("Bundle identity mismatch")
    expected = set(manifest["files"])
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
    if actual != expected | {"manifest.json"}:
        raise ValueError("Bundle inventory mismatch")
    required = {
        "model.safetensors",
        "config.json",
        "vocab.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "labels.json",
        "inference.json",
    }
    if not required <= expected:
        raise ValueError("Bundle inventory lacks required model/tokenizer files")
    for name, entry in manifest["files"].items():
        relative = PurePosixPath(name)
        candidate = path / name
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or ":" in name
            or candidate.is_symlink()
            or not candidate.resolve().is_relative_to(path)
        ):
            raise ValueError("Bundle inventory path is unsafe")
        if candidate.stat().st_size != entry["size"] or digest_file(candidate) != entry["sha256"]:
            raise ValueError("Bundle digest/size mismatch")
    if manifest["labels"] != list(LABELS) or read_json(path / "labels.json") != list(LABELS):
        raise ValueError("Bundle label mapping mismatch")
    if manifest["inference"] != POLICY or read_json(path / "inference.json") != POLICY:
        raise ValueError("Bundle inference policy mismatch")
    return BundleIdentity(manifest["model_id"], manifest)


@dataclass
class BundlePredictor:
    identity: BundleIdentity
    model: BertForSequenceClassification
    tokenizer: BertTokenizerFast

    def predict(self, texts) -> tuple[Inference, ...]:
        if not texts or any(not isinstance(t, str) or not normalize(t) for t in texts):
            raise ValueError("Inference requires nonempty strings")
        normalized = [normalize(t) for t in texts]
        untruncated = self.tokenizer(normalized, truncation=False, padding=False)["input_ids"]
        encoded = self.tokenizer(
            normalized, truncation=True, max_length=64, padding=True, return_tensors="pt"
        )
        with torch.inference_mode():
            labels = self.model(**encoded).logits.argmax(dim=-1).tolist()
        return tuple(
            Inference(
                LABELS[label], tuple(int(v) for v, keep in zip(ids, mask) if keep), len(raw) > 64
            )
            for label, ids, mask, raw in zip(
                labels,
                encoded["input_ids"].tolist(),
                encoded["attention_mask"].tolist(),
                untruncated,
            )
        )


def load_bundle(path: Path) -> BundlePredictor:
    identity = verify_bundle(path)
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    model = BertForSequenceClassification.from_pretrained(
        path, local_files_only=True, use_safetensors=True
    )
    model.eval()
    model.requires_grad_(False)
    tokenizer = BertTokenizerFast.from_pretrained(path, local_files_only=True)
    return BundlePredictor(identity, model, tokenizer)
