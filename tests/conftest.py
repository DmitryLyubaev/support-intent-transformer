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
