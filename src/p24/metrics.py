from collections import defaultdict

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from p24.config import LABELS
from p24.contracts import Prediction, Row


def _paired(rows, predictions, labels):
    ids = [r.row_id for r in rows]
    mapping = {p.row_id: p.label for p in predictions}
    if len(set(ids)) != len(ids) or len(mapping) != len(predictions) or set(ids) != set(mapping):
        raise ValueError("Comparison row identities are duplicated, missing or extra")
    if (
        not rows
        or any(r.label not in labels for r in rows)
        or any(p.label not in labels for p in predictions)
    ):
        raise ValueError("Comparison labels or population are invalid")
    return [r.label for r in rows], [mapping[r.row_id] for r in rows]


def score(rows: list[Row], predictions: list[Prediction], labels) -> dict:
    reference, predicted = _paired(rows, predictions, labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        reference, predicted, labels=list(labels), zero_division=0
    )
    return {
        "count": len(rows),
        "macro_f1": float(f1.mean()),
        "accuracy": float(accuracy_score(reference, predicted)),
        "per_class": [
            {
                "label": label,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(n),
            }
            for label, p, r, f, n in zip(labels, precision, recall, f1, support)
        ],
        "confusion_matrix": confusion_matrix(reference, predicted, labels=list(labels)).tolist(),
        "undefined_precision_recall": 0,
    }


def compare(rows, baseline, encoder) -> dict:
    first, second = score(rows, baseline, LABELS), score(rows, encoder, LABELS)
    return {
        "schema_version": 1,
        "count": len(rows),
        "baseline": first,
        "encoder": second,
        "delta_macro_f1": second["macro_f1"] - first["macro_f1"],
    }


def error_sample(rows, predictions) -> tuple[str, ...]:
    reference, predicted = _paired(rows, predictions, LABELS)
    errors = defaultdict(list)
    for row, actual, prediction in zip(rows, reference, predicted):
        if actual != prediction:
            errors[actual].append(row.row_id)
    return tuple(row_id for label in LABELS for row_id in sorted(errors[label])[:2])
