import warnings
from pathlib import Path

import joblib
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from p24.config import StudyConfig
from p24.contracts import Prediction, PreparedSplit, Row
from p24.data import normalize
from p24.io import digest_file, read_json, write_json


def fit_baseline(train: PreparedSplit, config: StudyConfig, destination: Path) -> Path:
    if destination.exists():
        raise ValueError("Baseline artifact exists; a new fit requires a new allowance")
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
            ("classifier", LogisticRegression(C=1, max_iter=1000, random_state=config.seed)),
        ]
    )
    with warnings.catch_warnings(), threadpool_limits(limits=4):
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit([normalize(r.text) for r in train.rows], [r.label for r in train.rows])
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination)
    write_json(
        destination.with_suffix(".json"),
        {
            "schema_version": 1,
            "sha256": digest_file(destination),
            "training_row_ids": [r.row_id for r in train.rows],
            "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
            "converged": True,
        },
    )
    return destination


def predict_baseline(artifact: Path, rows: list[Row]) -> tuple[Prediction, ...]:
    receipt = read_json(artifact.with_suffix(".json"))
    if digest_file(artifact) != receipt["sha256"]:
        raise ValueError("Baseline digest mismatch")
    pipeline = joblib.load(artifact)  # Only the project-generated, trusted local artifact.
    with threadpool_limits(limits=4):
        labels = pipeline.predict([normalize(r.text) for r in rows])
    return tuple(Prediction(r.row_id, str(label)) for r, label in zip(rows, labels))
