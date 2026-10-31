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
