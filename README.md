# Support intent transformer

Bounded six-intent CPU pretrained-encoder adaptation study. Implementation is in progress.
The approved design and plan are in docs/superpowers/. No model-quality results yet.

Task 1 source/split software and the native import gate now pass in the approved
project-local Python 3.12.16 environment. The original failing Python 3.12.11 environment
and its evidence remain retained. Tasks 2-6 and actual data/model acquisition are unstarted.

Use the recovered environment explicitly in PowerShell before running project commands:

```powershell
$env:UV_PROJECT_ENVIRONMENT = '.venv-31216'
uv run --frozen --no-sync python scripts/check.py
```

This selects a project-owned environment for the current shell; it does not change PATH
or global Python/TLS/security settings. Full runtime provenance is in
docs/runtime-recovery-proposal.md. Software checks do not train/evaluate pretrained weights.
