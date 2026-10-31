import hashlib
import os
import socket
from dataclasses import replace

import pytest

from p24.config import LABELS, StudyConfig
from p24.contracts import Row

os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    real = socket.socket.connect

    def connect(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError("Software tests cannot use external network")
        return real(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)


@pytest.fixture
def config():
    return replace(StudyConfig(), cap=8, min_train=1, min_validation=1, min_test=1)


@pytest.fixture
def rows():
    result = []
    for split, count in (("train", 12), ("test", 3)):
        for label_index, label in enumerate(LABELS):
            for i in range(count):
                index = label_index * count + i + 100
                text = hashlib.sha256(f"{split}-{label}-{i}".encode()).hexdigest()[:32]
                result.append(Row(f"{split}:{index:06}", split, label, text))
    return result


@pytest.fixture
def tiny_base(tmp_path):
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    root = tmp_path / "tiny-base"
    root.mkdir()
    words = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "hello",
        "card",
        "lost",
        "broken",
        "my",
        "where",
        "is",
        "the",
        "payment",
        "arrived",
        "cafe",
        "##s",
    ]
    (root / "vocab.txt").write_text("\n".join(words), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(root / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(root)
    torch.manual_seed(123)
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=len(words),
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=128,
            num_labels=6,
        )
    )
    model.save_pretrained(root, safe_serialization=True)
    return root
