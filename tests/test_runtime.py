"""An actual child-process import gate catches fatal native aborts that pytest cannot catch."""

import subprocess
import sys


def test_required_native_runtime_can_import_and_use_cpu():
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            "-c",
            (
                "import torch; from transformers import BertConfig, BertForSequenceClassification; "
                "import datasets, sklearn, mlflow, fastapi; "
                "assert torch.ones(2).sum().item() == 2.0; "
                "assert not torch.cuda.is_available(); print('CPU_RUNTIME_OK')"
            ),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CPU_RUNTIME_OK" in result.stdout
