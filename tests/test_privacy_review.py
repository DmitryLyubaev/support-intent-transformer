from p24.config import LABELS
from p24.contracts import Row
from p24.data import prepare


def test_manual_identifying_disposition_excludes_unflagged_record(rows, config):
    candidate = Row("train:000001", "train", LABELS[0], "Contact fictional John Smith at home")
    result = prepare(rows + [candidate], config, {"train:000001": "exclude_identifying"})
    assert result.audit["exclusions"].get("privacy", []) == ["train:000001"]
