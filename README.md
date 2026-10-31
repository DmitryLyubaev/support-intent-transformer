# Support intent transformer

A bounded six-intent CPU fine-tuning study: TF-IDF/logistic regression versus full
adaptation of a small pretrained BERT encoder, local MLflow lineage and saved-model
FastAPI serving. This is a personal portfolio demonstration, not production support.

Measured 31 October 2026: baseline macro-F1 **0.8960**, BERT-Tiny **0.4668**, on the
same **239 frozen test rows**. One fixed CPU recipe fully updated both encoder layers;
the baseline performed better. Training completed three epochs in **25.968 seconds**
including worker initialization/saving, with **553.5 MiB** sampled process-tree peak RSS.
The small six-intent, one-seed study supports no general quality claim. See the approved design/plan under
`docs/superpowers/`, [study report](docs/study-report.md), [model card](docs/model-card.md)
and [implementation record](docs/implementation-notes.md) for the current state.
The [review closeout](docs/review-closeout.md) records verifier fixes, the preserved
measured source and two deferred limitations. Original and post-review acceptance
outputs are retained separately under `docs/evidence/`.

## Local setup

Native Windows, Python **3.12.16**, uv **0.10.7**, CPU PyTorch and the committed `uv.lock`.
The original uv-managed Python 3.12.11 failed its OpenSSL import gate. A separately
approved project-local 3.12.16 environment passed the same dependency lock; the original
failure and environment remain retained. No PATH/global Python/TLS/security changes.

For a fresh checkout, obtain the [pinned official Python archive](https://github.com/astral-sh/python-build-standalone/releases/download/20261026/cpython-3.12.16%2B20261026-x86_64-pc-windows-msvc-install_only.tar.gz)
under `.local/runtime-download/`. Verify **47,113,364 bytes** and SHA-256
`590dfcbe3a34c07301e6332e13e2188b98655cd072b08afa45a7998dc4c22bc5` before extracting.
Inspect archive paths/types for containment, then extract only into the new owned
`.local/runtime/cpython-3.12.16/` directory. The validated archive had 3,388 regular
files/directories under its `python/` root. Never reuse a different archive silently.

```powershell
uv venv .venv-31216 --python .local/runtime/cpython-3.12.16/python/python.exe
$env:UV_PROJECT_ENVIRONMENT = '.venv-31216'
uv sync --locked --native-tls --link-mode=copy
uv run --frozen --no-sync python scripts/check.py
```

The environment variable selects this project environment for the current shell.
`check.py` verifies lock/lint/format and offline fixture tests, including actual loopback
process tests. It never acquires public data/weights, claims actual study attempts or
scores the final test. Two upstream deprecation warnings remain visible. Setup/native
runtime details are in [the recovery record](docs/runtime-recovery-proposal.md).

## Explicit study stages

Use at least **6 GiB available RAM**, **10 GiB free disk**, and coordinate heavy host use.
Each stage is separate. These commands describe the approved recipe, not permission to
repeat an already consumed fit or test inference. Original study claims/results remain
immutable under this checkout's `.local/study/`; no alternate-root budget reset exists.

```powershell
uv run --frozen --no-sync p24 acquire
uv run --frozen --no-sync p24 prepare
```

Preparation first writes a private preview and exits with `review_required`. Inspect
the fixed training sample and record an assistant review bound to that candidate hash
in `.local/study/selection-review.json`; unresolved regex flags are excluded. No review
is described as blinded human validation or an exhaustive PII audit. Then freeze with
`p24 prepare`. Source licences, hashes, exclusions and actual class counts must pass.

After source/preparation gates, execute each allowed stage once and inspect its receipt:

```powershell
uv run --frozen --no-sync p24 baseline
uv run --frozen --no-sync p24 feasibility
uv run --frozen --no-sync p24 train
uv run --frozen --no-sync p24 evaluate
uv run --frozen --no-sync p24 readback
uv run --frozen --no-sync p24 serve-check
uv run --frozen --no-sync p24 report
uv run --frozen --no-sync python scripts/acceptance.py --root .local/study
```

Feasibility is at most 50 steps/10 minutes; measured training starts afresh and is at
most three epochs/60 minutes, with a 4 GiB process-tree RSS ceiling. Seed42, batch16,
length64, full encoder/head updates and final-epoch selection are fixed. No automatic
retry, search, paid fallback or score-based continuation. Test inference happens once
for both arms on identical frozen rows; a worse encoder is a valid outcome. Recompute
metrics/report from retained predictions without refitting or rerunning test inference.

## Saved-model API

```powershell
uv run --frozen --no-sync python -m p24.api --bundle .local/study/model --port 8024
```

Only loopback `127.0.0.1:8024`. `GET /health` returns verified model identity.
`POST /predict` accepts exactly `{"text":"My card has not arrived yet."}` and returns
`label`, `model_id` and `truncated`. Raw text must contain 1-2,048 characters and remain
nonempty after NFC/strip. Saved uncased WordPiece truncates to 64 tokens including
special tokens. No confidence score or unknown-intent detector: unrelated valid strings
are still forced into one of the six labels. No request-time network/training/MLflow
service, model upload, production authentication or SLA is claimed. Stop the owned
process after use. `serve-check` demonstrates and stops two actual server processes.

Raw/prepared messages, weights, baseline binary, MLflow and original receipts stay
ignored under `.local/`. Source/aggregate evidence only; no public model/dataset release,
hosted CI, paid provider, cloud deployment or CV/profile editing is part of this scope.
Input attribution and licence limits are in [ATTRIBUTION.md](ATTRIBUTION.md).

Acquisition uses existing PowerShell 7 with normal Windows HTTPS certificate verification
for the exact pinned Hugging Face URLs. Python Requests failed its certificate-chain
check here; the failed pre-input snapshot is retained privately. No TLS verification,
certificate store, account credentials or global security configuration was changed.
