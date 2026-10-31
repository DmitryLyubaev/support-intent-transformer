from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Row:
    row_id: str
    upstream_split: str
    label: str
    text: str


@dataclass(frozen=True)
class PreparedSplit:
    rows: tuple[Row, ...]
    manifest: dict


@dataclass(frozen=True)
class FrozenStudy:
    train: PreparedSplit
    validation: PreparedSplit
    test: PreparedSplit
    identity: str
    audit: dict


@dataclass(frozen=True)
class Prediction:
    row_id: str
    label: str


@dataclass(frozen=True)
class Inference:
    label: str
    token_ids: tuple[int, ...]
    truncated: bool


@dataclass(frozen=True)
class RunIdentity:
    source_hash: str
    lock_hash: str
    config_hash: str
    freeze_id: str


@dataclass(frozen=True)
class AttemptReceipt:
    purpose: str
    status: str
    completed_steps: int
    elapsed_seconds: float
    peak_rss_bytes: int
    output_dir: Path | None
    identity: RunIdentity


@dataclass(frozen=True)
class BundleIdentity:
    model_id: str
    manifest: dict
