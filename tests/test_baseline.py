import joblib

from p24.baseline import fit_baseline, predict_baseline
from p24.config import LABELS, StudyConfig
from p24.contracts import PreparedSplit, Row


def test_baseline_vocabulary_uses_only_training(tmp_path):
    rows = tuple(
        Row(f"train:{i:06}", "train", label, f"topic{i} original request")
        for i, label in enumerate(LABELS)
    )
    artifact = fit_baseline(PreparedSplit(rows, {}), StudyConfig(), tmp_path / "baseline.joblib")
    validation = [Row("test:000001", "test", LABELS[0], "validationonly request")]
    prediction = predict_baseline(artifact, validation)
    pipeline = joblib.load(artifact)
    assert "validationonly" not in pipeline.named_steps["tfidf"].vocabulary_
    assert "original" in pipeline.named_steps["tfidf"].vocabulary_
    assert prediction[0].row_id == "test:000001"
    assert prediction[0].label in LABELS
