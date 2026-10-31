import json

import pytest

from p24.io import digest_file


def inventory(root):
    return {
        p.relative_to(root).as_posix(): (digest_file(p), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def test_source_row_readback_never_builds_a_dataset_cache(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from p24.sources import read_source_rows

    source = tmp_path / "sources/dataset/data"
    source.mkdir(parents=True)
    features = {
        "info": {
            "features": {
                "text": {"dtype": "string", "_type": "Value"},
                "label": {"names": ["first", "second"], "_type": "ClassLabel"},
            }
        }
    }
    for split in ("train", "test"):
        table = pa.table({"text": ["independent fixture"], "label": [1]})
        table = table.replace_schema_metadata({b"huggingface": json.dumps(features).encode()})
        pq.write_table(table, source / f"{split}-00000-of-00001.parquet")
    before = inventory(tmp_path)
    rows = read_source_rows(tmp_path)
    assert [(r.row_id, r.label) for r in rows] == [
        ("train:000000", "second"),
        ("test:000000", "second"),
    ]
    assert inventory(tmp_path) == before, "Readback must not create or modify evidence/cache files"


def test_mlflow_readback_does_not_create_an_absent_store(tmp_path):
    from p24 import tracking

    root = tmp_path / "absent-tracking"
    with pytest.raises(FileNotFoundError):
        tracking.read_run(root, "no-such-run")
    assert not root.exists()


def test_mlflow_existing_run_is_read_without_any_evidence_writes(tmp_path):
    from p24 import tracking
    from p24.contracts import RunIdentity

    root = tmp_path / "tracking"
    run_id = tracking.log_run(
        root,
        "fixture-only",
        RunIdentity("source", "lock", "config", "freeze"),
        {"status": "succeeded"},
        [],
    )
    before = inventory(root)
    result = tracking.read_run(root, run_id)
    assert result["status"] == "FINISHED"
    assert result["tags"]["purpose"] == "fixture-only"
    assert result["tags"]["source_hash"] == "source"
    assert inventory(root) == before
