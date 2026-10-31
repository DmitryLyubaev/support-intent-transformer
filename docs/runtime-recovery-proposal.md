# Native runtime recovery proposal - 31 October 2026

Status: proposed, not approved or executed. Task 1 is incomplete at its native import gate.

## Observation

The project environment uses existing uv-managed CPython 3.12.11. CPU PyTorch 2.15.0
imports and computes a tensor successfully. Datasets 4.8.5 terminates the interpreter
with exit 1 and `OPENSSL_Uplink: no OPENSSL_Applink`; independent sklearn, MLflow,
FastAPI, PyArrow and ssl imports pass. The inherited SSLKEYLOGFILE variable is present;
its value or contents were not collected. No TLS/security setting has been changed.

Astral's [upstream fix](https://github.com/astral-sh/python-build-standalone/pull/1132),
merged 21 May 2026, addresses creating SSLContext with SSLKEYLOGFILE set on Windows.
The observed failure strongly matches that issue; an independent before/after test of
an updated runtime has not been performed. This is not evidence of a model/data defect.

## Smallest proposed recovery

Keep native Windows, Python 3.12, CPU operation, the exact dependency lock and all model/
split/experiment limits. Download the official project-local standalone build:

- Release: `20261026`, CPython `3.12.16`, x86_64 Windows MSVC install-only archive.
- [Official asset](https://github.com/astral-sh/python-build-standalone/releases/download/20261026/cpython-3.12.16%2B20261026-x86_64-pc-windows-msvc-install_only.tar.gz).
- Metadata size: 47,113,364 bytes.
- Published SHA-256: `590dfcbe3a34c07301e6332e13e2188b98655cd072b08afa45a7998dc4c22bc5`.

After approval, download under ignored .local/, verify size/digest, inspect archive paths
for containment, and extract only to a new owned .local/runtime directory. Do not replace
the shared uv interpreter, install globally, update PATH, change SSLKEYLOGFILE, TLS
verification/security software or the existing environment. Create a separate owned
.venv-31216 environment with this interpreter and sync the unchanged lock into it.
Keep the original failing environment and evidence. Re-run the native import/tensor
regression and canonical checks using the new environment explicitly; proceed with the
approved plan only if both pass. Stop/report a second native failure rather than trying
another runtime/platform automatically. No dataset/weights or actual study slot are used
to test recovery. No paid/cloud/public action is involved.

The approved design says: "If compatible CPU wheels or imports fail, record the failure
and stop rather than changing execution platforms silently." This proposal explicitly
reopens that setup gate while retaining the approved native platform. The old uv catalog
lists 3.12.12 as its newest 3.12 download, so it must not silently select a pre-fix build.
