import pytest

from p24.config import LABELS
from p24.contracts import Prediction, Row
from p24.metrics import compare, error_sample, score


def reference_rows():
    return [
        Row(f"test:{i:06}", "test", label, "independent fixture") for i, label in enumerate(LABELS)
    ]


def test_metrics_include_all_six_labels():
    rows = reference_rows()
    predictions = [Prediction(r.row_id, LABELS[i]) for r, i in zip(rows, [0, 0, 2, 3, 4, 5])]
    result = score(rows, predictions, LABELS)
    assert result["macro_f1"] == pytest.approx(7 / 9)
    assert result["accuracy"] == pytest.approx(5 / 6)
    assert [c["support"] for c in result["per_class"]] == [1] * 6
    assert result["per_class"][1]["recall"] == 0
    assert result["confusion_matrix"][1] == [1, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("case", ["duplicate", "missing", "extra"])
def test_pairing_rejects_duplicate_missing_or_extra_ids(case):
    rows = reference_rows()
    predictions = [Prediction(r.row_id, r.label) for r in rows]
    if case == "duplicate":
        predictions[-1] = predictions[0]
    elif case == "missing":
        predictions.pop()
    else:
        predictions.append(Prediction("test:999999", LABELS[0]))
    with pytest.raises(ValueError, match="identit"):
        score(rows, predictions, LABELS)


def test_shuffled_predictions_pair_by_id_and_inferior_encoder_is_valid():
    rows = reference_rows()
    baseline = [Prediction(r.row_id, r.label) for r in reversed(rows)]
    encoder = [Prediction(r.row_id, LABELS[0]) for r in rows]
    result = compare(rows, baseline, encoder)
    assert result["baseline"]["macro_f1"] == 1
    assert result["delta_macro_f1"] < 0
    assert result["count"] == 6


def test_error_sample_is_lowest_two_ids_per_class():
    rows = [Row(f"test:{i:06}", "test", LABELS[0], "fixture") for i in [9, 3, 7, 1]]
    predictions = [Prediction(r.row_id, LABELS[1]) for r in rows]
    assert error_sample(rows, predictions) == ("test:000001", "test:000003")
