import pytest

from p24.sources import SourceError, fetch_sources


def test_partial_or_different_source_receipt_is_rejected(tmp_path, config):
    cache = tmp_path / "sources"
    cache.mkdir()
    (cache / "partial").write_bytes(b"partial download")
    with pytest.raises(SourceError, match="receipt"):
        fetch_sources(config, tmp_path)


def test_complete_acquisition_receipts_include_hash_size_and_time(tmp_path, config):
    def transport(repo, revision, filename, kind):
        assert revision in (config.model_revision, config.dataset_revision)
        return f"{repo}|{filename}".encode()

    receipt = fetch_sources(config, tmp_path, transport)
    assert len(receipt["files"]) == 7
    assert all(
        len(f["sha256"]) == 64 and f["size"] > 0 and f["retrieved_at"].endswith("Z")
        for f in receipt["files"]
    )
    target = tmp_path / "sources" / receipt["files"][0]["path"]
    target.write_bytes(b"changed")
    with pytest.raises(SourceError, match="digest"):
        fetch_sources(config, tmp_path, transport)


def test_source_receipt_cannot_reference_a_file_outside_its_cache(tmp_path, config):
    import hashlib

    from p24.io import write_json

    receipt = fetch_sources(config, tmp_path, lambda *args: b"fixture input")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"not a pinned source")
    entry = receipt["files"][0]
    entry.update(
        path="../outside.txt",
        sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
        size=outside.stat().st_size,
    )
    write_json(tmp_path / "sources/receipt.json", receipt)
    with pytest.raises(SourceError, match="path|inventory"):
        fetch_sources(config, tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("url", None),
        ("url", "https://example.org/changed-source"),
        ("retrieved_at", None),
        ("retrieved_at", "not-a-timestamp"),
        ("retrieved_at", "2026-10-31T08:00:00"),
        ("retrieved_at", "2026-10-31T08:00:00+10:00"),
    ],
)
def test_incomplete_or_invalid_provenance_cannot_pass_cached_source_gate(
    tmp_path, config, field, value
):
    from p24.io import write_json

    receipt = fetch_sources(config, tmp_path, lambda *args: b"fixture input")
    if value is None:
        del receipt["files"][0][field]
    else:
        receipt["files"][0][field] = value
    write_json(tmp_path / "sources/receipt.json", receipt)
    with pytest.raises(SourceError, match="provenance|timestamp|URL"):
        fetch_sources(config, tmp_path)


def test_unknown_source_receipt_schema_is_rejected(tmp_path, config):
    from p24.io import write_json

    receipt = fetch_sources(config, tmp_path, lambda *args: b"fixture input")
    receipt["schema_version"] = 999
    write_json(tmp_path / "sources/receipt.json", receipt)
    with pytest.raises(SourceError, match="schema"):
        fetch_sources(config, tmp_path)
