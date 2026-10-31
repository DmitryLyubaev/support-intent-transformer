from dataclasses import asdict, dataclass
from pathlib import Path

from p24.io import canonical, read_json

LABELS = (
    "card_arrival",
    "card_delivery_estimate",
    "card_not_working",
    "lost_or_stolen_card",
    "declined_card_payment",
    "pending_card_payment",
)


@dataclass(frozen=True)
class StudyConfig:
    labels: tuple[str, ...] = LABELS
    dataset_revision: str = "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6"
    model_revision: str = "30b0a37ccaaa32f332884b96992754e246e48c5f"
    seed: int = 42
    cap: int = 160
    min_train: int = 40
    min_validation: int = 10
    min_test: int = 10
    batch_size: int = 16
    length: int = 64
    epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01


def read_config(path: Path) -> StudyConfig:
    if canonical(read_json(path)) != canonical(asdict(StudyConfig())):
        raise ValueError("Actual study config differs from approved recipe")
    return StudyConfig()
