# Implementation record

Approved bounded Project 24 study, Native execution. See the isolated execution ledger
for task commits, RED/GREEN checks, operational receipts, rulings and eventual review.
Software fixtures are independent of actual classification-quality measurements.

## Initial Task 1 native runtime gate (resolved)

On 31 October 2026, nine source/split tests passed after nine expected unimplemented
failures. CPU torch 2.15.0+cpu imported and computed 2.0 from a two-element tensor;
Datasets 4.8.5 reproducibly aborted native CPython 3.12.11 with an OpenSSL Applink error.
Independent sklearn/MLflow/FastAPI/PyArrow/ssl imports passed. The full-runtime regression
failed at that checkpoint, so Task 1 was then incomplete. No dataset rows,
weights, training, API service or whole-branch review had occurred at that checkpoint.

The setup stop is required by the approved design. A concrete project-local 3.12.16
recovery proposal preserves the dependency lock, native CPU scope, global interpreter
and TLS/security settings. Dmitry approved it and it subsequently passed. Raw receipts remain ignored under
.local/evidence/. Preserve the worktree and ledger for resumption.

Recovery: approved archive SHA/size and 3,388 extraction entries verified; new owned
.venv-31216 uses Python 3.12.16. The uv.lock Git blob stayed
18b37f8f4bae1e3019bd188c767e3702266cc6ab before/after. Canonical runtime plus source/split
checks passed 12 tests in 72.40s. Two additional frozen-metadata/digest substitution
regressions were observed RED then GREEN, closing the manifest readback boundary.

## Implementation before the actual study

Task2 `c6ce900`: training-only baseline and paired metrics; canonical19 passed.
Task3 `d860630`: full encoder updates and owned watchdog; canonical30 passed, including
the independently reproduced late-completion deadline race correction.
Task4 `437f040`: verified bundles and actual SQLite-backed MLflow download; canonical37.
Task5 `1867339`: strict API and actual process restart; canonical50.
Task6 software: canonical62 passed in76.82s, zero failures/skips. The two visible upstream
deprecations are Starlette/httpx TestClient and MLflow/SQLAlchemy noload. Neither caused
a failed artifact or HTTP readback in this environment; they remain upgrade limitations.

Task6 RED/GREEN regressions additionally establish: no final-test access before eligibility,
no repeated evaluation/lineage reset, no false actual-quality claim from fixtures,
manual identifying-record exclusions, separate train/validation truncation counts,
receipt-path containment, immutable pinned-input checks before fitting, and read-only
verification even under Python optimization. Read-only checks cannot invoke downloading.
At that pre-study checkpoint all exercised models were tiny local random fixtures.

## Actual study

Executable source `4dc18f9b31bb39ff1cd96fd967008a0ecc541284` passed 65 software
tests before acquisition. All seven downloaded files matched official pinned byte
hashes. The freeze retained 610 training, 155 validation and 239 test rows. One baseline
fit, one train-only feasibility attempt and one fresh measured encoder fit succeeded.
The measured fit completed 117 steps/three epochs in 25.968 seconds including process
initialization/saving, with 580,382,720 bytes sampled process-tree peak RSS.

One final-test inference per arm produced macro-F1 0.896042 baseline and 0.466797
encoder. No tuning, relabeling or favorable rerun followed. Both pretrained encoder
layers changed, while the lost/stolen class had zero encoder recall. See study-report.md
for full per-class metrics, confusion matrices, fixed-sample error analysis and limits.
MLflow artifact download verified the identical bundle. Actual HTTP parity/restart
handled 12 predictions/eight rejections and stopped both owned processes. Raw receipts,
messages, models and local MLflow remain ignored; safe aggregate evidence is in docs/evidence/.

## Rulings retained

1. App worktree tooling rejected the new product source root. Used an ignored Git
   worktree without app-permission changes. Cost: app cannot manage its lifecycle; retain
   explicit paths and use Git for later integration.
2. Shared canonical JSON/hash/atomic-write helpers and a test-only inherited offline
   socket guard prevent identity drift and external calls from Python test children.
   Cost: serialization changes affect all identities; end-to-end digest/readback checks
   must continue to cover it.
3. Split demonstration, read-only acceptance and report rendering into focused modules,
   retaining the stage interfaces and source-row reader contract. Cost: cross-module
   drift could overstate evidence; canonical checks, A1-A8 and final review must check it.

4. Python Requests rejected the public source certificate while Windows' normal HTTPS
   client successfully fetched the same pinned README (13,049 bytes, SHA-256
   f3786a5327fd6533c7c0ff3ff654e20ee3d1261d2fa41667381f59a2e42364e1).
   Use the existing PowerShell 7 verified HTTPS client for identical pinned URLs, with
   no TLS/security/global trust changes or account credentials. Cost: transport differences
   could alter bytes; record hashes and check against official pinned file metadata.
   Preserve/archive the failed acquisition snapshot; it contains no data files or actual
   experiment claims, so no fit/evaluation allowance is reset.
