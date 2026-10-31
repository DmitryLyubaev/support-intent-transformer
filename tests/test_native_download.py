import pytest

from p24 import sources


@pytest.mark.parametrize("url", ["http://huggingface.co/file", "https://elsewhere.example/file"])
def test_native_downloader_refuses_unapproved_origin(tmp_path, url):
    with pytest.raises(sources.SourceError, match="HTTPS|origin"):
        sources.native_download(url, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_native_downloader_preserves_structured_uri_and_output_path(tmp_path, monkeypatch):
    script = tmp_path / "transport double.ps1"
    script.write_text(
        """param([string]$Uri, [string]$OutputPath)
if ($Uri -ne 'https://huggingface.co/datasets/PolyAI/banking77/resolve/revision/README.md') { exit 13 }
[System.IO.File]::WriteAllBytes($OutputPath, [System.Text.Encoding]::UTF8.GetBytes('fixture payload'))
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(sources, "NATIVE_SCRIPT", script)
    destination = tmp_path / "output with spaces.txt"
    sources.native_download(
        "https://huggingface.co/datasets/PolyAI/banking77/resolve/revision/README.md", destination
    )
    assert destination.read_bytes() == b"fixture payload"
