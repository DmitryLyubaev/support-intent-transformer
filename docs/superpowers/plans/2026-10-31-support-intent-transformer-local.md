# Support Intent Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete one bounded six-intent CPU transformer fine-tuning study, compare a training-only text baseline on frozen test rows, and serve the actual saved model locally.

**Architecture:** One native Windows Python project separates provenance/preparation, baseline/metrics, bounded encoder training, artifact lineage and FastAPI serving. Tiny offline fixtures verify software before an explicit one-time real-data study. A parent supervisor records attempts and bounds the training child; saved bundles load without network or MLflow services.

**Tech Stack:** Python 3.12/uv, CPU PyTorch, Transformers/Datasets, safetensors, scikit-learn, local SQLite-backed MLflow, FastAPI/Pydantic/Uvicorn, psutil, pytest/httpx and Ruff.

**Spec:** [Approved design](../specs/2026-10-31-support-intent-transformer-design.md), approved by Dmitry on 31 October 2026. Read the design and this plan together.

Status: approved by Dmitry on 31 October 2026; recommended Native execution accepted.
All six tasks and one independent whole-branch review are complete. Three Important verifier gaps were fixed in one RED/GREEN pass; two Minors are deferred in docs/review-closeout.md. Final canonical87 PASS and corrected actual A1-A8 PASS without changing155 evidence entries. Measured executable remains4dc18f9; verifier changes do not repeat training/test inference. Dmitry selected local integration; main was fast-forwarded and independently passed87 tests/corrected A1-A8. Private files and environment snapshots are preserved; merged branch/worktree removed. The bounded local plan is complete, with no public publication. See docs/local-integration.md.

## Global Constraints

- Bounded personal CV/portfolio demonstration, not a production system. One baseline and one fully adapted BERT-Tiny encoder; no model-quality minimum.
- Product root `E:/Projects/support-intent-transformer`; paths below are relative to it. Planning root `E:/Projects/portfolio-plan`. Check absence/ownership before repository creation; use using-git-worktrees at execution time for isolated product work.
- Native Windows, project-owned Python 3.12 environment and CPU wheels. No global runtime changes, CUDA, WSL/Docker restart, other-project environment or silent platform fallback. Docker and hosted CI are outside this local plan.
- Dataset `PolyAI/banking77` revision `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`; model `google/bert_uncased_L-2_H-128_A-2` revision `30b0a37ccaaa32f332884b96992754e246e48c5f`. Fetch only the files named in the design, with actual UTC retrieval time, size and SHA-256. No remote scripts/code or moving-main fallback.
- Label order: `card_arrival`, `card_delivery_estimate`, `card_not_working`, `lost_or_stolen_card`, `declined_card_payment`, `pending_card_payment`. Original labels only; no seventh/OOS class or confidence claim.
- Model normalization NFC/strip; duplicate normalization NFKC/casefold/collapsed whitespace. Equality or character five-gram Jaccard >=0.85 for strings both >=20 characters; connected components, conflicting-label exclusion, test precedence and complete-group development selection exactly as designed.
- Development cap 160 rows/class, 960 total; grouped validation >=20%/class. Minimum 40 train, 10 validation and 10 retained test rows/class. Fail rather than silently change labels, grouping or split thresholds.
- Baseline word TF-IDF (1,2), maximum 10,000 features; logistic regression C=1, max_iter=1000, no class weighting, seed 42. Fit training only; no search/refit/retry.
- Encoder/head all trainable; seed 42, three epochs, batch 16, length 64 including specials, dynamic padding/truncation, AdamW 5e-5/weight decay 0.01, no scheduler/warmup, cross-entropy, clip norm 1.0, seeded shuffle, zero loader workers. Intra-op/BLAS <=4 threads, inter-op 1. Final epoch selected, no best-checkpoint or early-stop search.
- One train-only feasibility attempt <=50 steps/10 minutes; one fresh measured run <=three epochs/60 minutes. Actual changes in both pretrained encoder layers, finite loss, <=4 GiB sampled process-tree RSS and projected full run <=45 minutes are feasibility gates. Preserve failed attempts; no automatic actual-run retry.
- Recheck >=6 GiB free RAM and >=10 GiB free disk before setup/runs and coordinate heavy work with Projects 22/23. Wait/report when unavailable; never stop another project's processes. Sample RSS at least every five seconds; watchdog includes validation/saving.
- Freeze before final test; score both arms once on the same rows. Preserve inferior/failed results, original references and retest history. Recompute metrics from saved predictions freely; no implicit refit or test-inference rerun.
- `.local/` holds ignored raw/prepared text, weights, baseline pickle, MLflow SQLite/artifacts and raw evidence. Safe Git outputs contain no raw messages/identifiers/binaries. No public push, paid/provider call, cloud provisioning, CV/profile editing or other-application changes.
- Local MLflow experiment `p24-support-intent`; direct tracking API, no server/registry. API only `127.0.0.1:8024` after absence check. Stop only owned processes and retain evidence.
- Personal Git identity `Dmitry Lyubaev <Dmitry.Lyubaev@gmail.com>`. No Unicode U+2014 in prose.

## Review Focus

- Transitive duplicate chains and label conflicts can leak even when adjacent rows look harmless: exclude conflicting components, retain test precedence and keep every development group within one split. Task 1.
- Unresolved privacy flags or a partially recorded download must not become accepted labeled rows: require review disposition and complete source receipts; fail closed without overwriting evidence. Task 1.
- Abrupt parent/child exit, changed identities or simultaneous commands must not reset a consumed actual attempt: persist an exclusive claim before launch and refuse replay; incomplete weights remain unselected. Task 3.
- A relocated bundle or substituted tokenizer must retain identity or fail verification before loading: identity is path-independent and covers settings/vocabulary as well as weights. Task 4.
- Unicode normalization, exact token limits and strict JSON types must behave the same in offline inference and HTTP: return identical labels/token IDs and correct truncation; reject whitespace, extra keys, numeric text and >2,048 raw characters. Task 5.

## Product file map and contracts

Use `src/p24/` with `__init__.py`, `contracts.py` (dataclasses/JSON schemas), `config.py`
(approved settings), `sources.py` (allowlisted acquisition/receipts), `data.py`
(normalization/privacy dispositions/grouping/freeze), `baseline.py`, `metrics.py`,
`encoder.py` (training worker), `attempts.py` (claim/events/supervisor), `bundle.py`,
`tracking.py`, `api.py`, `study.py` (explicit orchestration) and `cli.py`.
Create `tests/conftest.py`, the task-specific tests below, `scripts/check.py`,
`scripts/acceptance.py`, `config/study.json`, `pyproject.toml`, `uv.lock`, `.gitignore`,
`README.md`, `ATTRIBUTION.md`, `docs/model-card.md`, `docs/study-report.md` and
`docs/implementation-notes.md`. No workflow/container/frontend files in this plan.

All JSON has an explicit schema version. Define these contracts in Task 1, not separate
incompatible dictionaries in later tasks:

- `Row(row_id: str, upstream_split: str, label: str, text: str)`; IDs are `train:000000`
  or `test:000000` with the actual zero-based index padded to six digits.
- `PreparedSplit(rows: tuple[Row, ...], manifest: dict)` and
  `FrozenStudy(train: PreparedSplit, validation: PreparedSplit, test: PreparedSplit, identity: str)`.
- `Prediction(row_id: str, label: str)` and `Inference(label: str, token_ids: tuple[int, ...], truncated: bool)`.
- `RunIdentity(source_hash: str, lock_hash: str, config_hash: str, freeze_id: str)`;
  `AttemptReceipt(purpose: str, status: str, completed_steps: int, elapsed_seconds: float, peak_rss_bytes: int, output_dir: Path | None, identity: RunIdentity)`.
- `BundleIdentity(model_id: str, manifest: dict)`; no API response exposes token IDs.

Canonical verification is `uv run --frozen --no-sync python scripts/check.py`: run
`uv lock --check --offline`, Ruff check/format-check and pytest, with networking disabled
for test subprocesses. It must never acquire data, claim actual study slots or train
pretrained weights. Package setup/install happens explicitly before this command.

### Task 1: Reproducible source preparation and leakage-aware freeze

**Files:** Create project setup/config/contracts/source/data files above,
`tests/conftest.py`, `tests/test_sources.py`, `tests/test_data.py`, `scripts/check.py`,
`.gitignore`, `ATTRIBUTION.md` and initial README/implementation notes.

**Interfaces:** `read_config(path: Path) -> StudyConfig` returns immutable validated
settings; `fetch_sources(config: StudyConfig, root: Path) -> dict` returns receipts;
`prepare(rows: Sequence[Row], config: StudyConfig, dispositions: Mapping[str, str]) -> FrozenStudy`;
`write_freeze(study: FrozenStudy, root: Path) -> Path` refuses a conflicting existing freeze;
`read_training(freeze: Path) -> tuple[PreparedSplit, PreparedSplit]` never loads test rows.
`StudyConfig` contains exactly the design settings and input revisions, with actual-run
validation rejecting fixture-only overrides.

- [x] **Step 1 - establish the project-owned environment.** At execution time, confirm the
  target absent, resources available and personal identity; initialize the repository,
  baseline commit and isolated branch/worktree through using-git-worktrees. Resolve
  compatible stable versions into `uv.lock`, map torch explicitly to the official CPU
  wheel index and install only in the owned environment. Include psutil for watchdog RSS,
  Uvicorn for real HTTP and httpx for tests. Verify Python 3.12, imports and an actual CPU
  tensor operation. Preserve failed resolution/install receipts; stop on incompatibility.
- [x] **Step 2 - write failing source/split tests.** Define
  `test_duplicate_chain_with_conflicting_labels_is_excluded`,
  `test_test_component_never_enters_train_or_validation`,
  `test_whole_groups_caps_and_minimums`,
  `test_unresolved_privacy_flag_is_excluded_without_relabel`,
  `test_partial_or_different_source_receipt_is_rejected` and
  `test_training_loader_never_opens_test`. Pin behavior with assertions such as:

  ```python
  assert not (train_group_ids & validation_group_ids)
  assert not (train_group_ids & test_group_ids | validation_group_ids & test_group_ids)
  assert all(count <= 160 for count in development_counts.values())
  assert excluded_ids == {"train:000001", "train:000002", "test:000003"}
  assert retained_labels == original_labels_for_retained_ids
  with pytest.raises(PreparationError, match="minimum"):
      prepare(too_few_rows, approved_config, {})
  ```

  Use tiny synthetic cases for grouping/receipt failures; separately generate enough
  independent fixture rows to exercise approved minima/caps. Stub acquisition transport,
  never call the actual Hub in software tests. Run
  `uv run --frozen --no-sync pytest tests/test_sources.py tests/test_data.py -q` and
  observe the missing implementation/behavior failures before implementation.
- [x] **Step 3 - implement preparation.** Fetch only the design's pinned files through
  ordinary Hub downloads into temporary owned paths, atomically finalize complete
  size/hash/time receipts, and refuse an unreceipted/conflicting cache. Implement the
  exact connected-component/split algorithm. Privacy scan flags plus local dispositions
  cannot alter labels; missing/uncertain dispositions exclude flagged rows. Keep raw
  text private and safe manifests text-free. Training loader receives train/validation
  manifests only; detect modification of a frozen file/config on read.
- [x] **Step 4 - verify.** Run the Task 1 tests and canonical command; expected all pass,
  no network/data download, no study claim and no generated text under tracked paths.
  Check `.local/` is ignored and record resolved versions/import/resource outcomes.
- [x] **Step 5 - commit.** Stage only Task 1's named product files and commit
  `feat: freeze pinned support-intent inputs and grouped splits`. Update planner tracker/
  handover with actual evidence; do not claim real acquisition or dataset acceptance yet.

### Task 2: Training-only baseline and exact paired metrics

**Files:** Create `baseline.py`, `metrics.py`, `tests/test_baseline.py`, `tests/test_metrics.py`.

**Interfaces:** `fit_baseline(train: PreparedSplit, config: StudyConfig, destination: Path) -> Path`;
`predict_baseline(artifact: Path, rows: Sequence[Row]) -> tuple[Prediction, ...]`;
`score(rows: Sequence[Row], predictions: Sequence[Prediction], labels: Sequence[str]) -> dict`;
`compare(rows: Sequence[Row], baseline: Sequence[Prediction], encoder: Sequence[Prediction]) -> dict`;
`error_sample(rows: Sequence[Row], predictions: Sequence[Prediction]) -> tuple[str, ...]`.

- [x] **Step 1 - write failing tests.** `test_baseline_vocabulary_uses_only_training`
  asserts an independent `validationonly` token never enters the fitted vocabulary.
  `test_metrics_include_all_six_labels` uses one reference per label and predictions
  `[0, 0, 2, 3, 4, 5]`, asserting macro-F1 `pytest.approx(7 / 9)`, accuracy
  `pytest.approx(5 / 6)`, support `[1] * 6` and zero recall for class 1.
  `test_pairing_rejects_duplicate_missing_or_extra_ids` requires rejection before
  comparison; shuffled valid predictions must pair by immutable ID.
  `test_error_sample_is_lowest_two_ids_per_class` pins deterministic selection.
- [x] **Step 2 - observe RED.** Run
  `uv run --frozen --no-sync pytest tests/test_baseline.py tests/test_metrics.py -q`;
  expected failure for missing behavior, not a broken fixture/environment.
- [x] **Step 3 - implement the fixed pipeline/evaluator.** Preserve label order, use
  zero_division=0 and exact ID-set checks. Treat convergence warnings as an unsuccessful
  baseline run, preserve receipt and stop the real study; do not increase iterations.
  Persist only trusted project-generated baseline data locally, with digest/versions.
- [x] **Step 4 - verify GREEN.** Run Task 2 tests and canonical checks; confirm a deliberately
  worse encoder comparison is reported numerically and is not an acceptance failure.
- [x] **Step 5 - commit.** Commit named files as
  `feat: compare training-only baseline with paired six-label metrics`.

### Task 3: Actual encoder adaptation with durable attempt and resource bounds

**Files:** Create `encoder.py`, `attempts.py`, `tests/test_encoder.py`, `tests/test_attempts.py`.

**Interfaces:** `claim_attempt(root: Path, purpose: Literal["feasibility", "measured"], identity: RunIdentity) -> Path`;
`append_event(claim: Path, event: Mapping[str, object]) -> None`;
`supervise(argv: Sequence[str], claim: Path, timeout_seconds: int, rss_limit_bytes: int) -> AttemptReceipt`;
`fit_encoder(train: PreparedSplit, validation: PreparedSplit | None, base_dir: Path, output: Path, config: StudyConfig, purpose: str, emit: Callable[[dict], None]) -> dict`.
The worker saves a staged model/tokenizer and tensor-change receipt; Task 4 seals the
bundle. No MLflow service or final-test path is needed by the worker.

- [x] **Step 1 - write failing tests.** A tiny locally initialized two-layer BERT/vocabulary
  fixture trains one optimizer step and asserts a nonzero max-absolute delta for one
  attention/feed-forward tensor in each layer, all parameters trainable and finite loss.
  `test_fresh_run_reloads_base_not_feasibility_weights` checks initialization digests.
  `test_exclusive_claim_survives_crash_and_changed_identity` asserts second/concurrent
  claims fail, even with different config hashes. `test_supervisor_kills_only_owned_tree`
  launches owned disposable children, uses a short fixture deadline, asserts terminal
  timeout receipt plus no owned children, and leaves an unrelated sentinel alive.
  An injected sampler tests RSS failure; a real sampler test observes nonzero process RSS.
- [x] **Step 2 - observe RED.** Run
  `uv run --frozen --no-sync pytest tests/test_encoder.py tests/test_attempts.py -q`.
  No real source downloads/pretrained training or actual study claims are allowed here.
- [x] **Step 3 - implement worker and supervisor.** Use the exact approved recipe, final
  epoch selection, logged per-epoch validation and canonical contiguous CPU tensor
  digests/deltas. Exclusive fixed-purpose claim markers under `.local/study/attempts/`
  are written/fsynced before spawning; an existing marker is never cleared for resume.
  Keep append-only claim/start/progress/end events and mark unknown/crashed attempts
  incomplete on read, never reusable. Parent sets thread limits before process startup,
  polls RSS at <=5 seconds, bounds training/validation/staging writes and terminates only
  its process tree on ceiling/failure. Require normal child exit and complete receipt
  before marking success. Partial staged files remain diagnostic, never selected.
- [x] **Step 4 - verify GREEN.** Run tests plus canonical checks. Test timeout, nonzero
  child exit, nonfinite loss, head-only change and missing terminal receipt as failures.
  Feasibility uses at most the first 128 sorted training IDs, cycles only those rows for
  at most 50 steps, and never loads validation/test. Its conservative projection is
  `3 * ceil(train_count / 16) * elapsed_seconds / completed_steps`, including observed
  startup overhead; require completed steps >0, normal completion, finite loss, both
  layer deltas and projection <=2,700 seconds before measured launch is enabled.
- [x] **Step 5 - commit.** Commit named files as
  `feat: bound encoder adaptation with durable attempts and owned watchdog`.

### Task 4: Verified saved bundles and local MLflow lineage

**Files:** Create `bundle.py`, `tracking.py`, `tests/test_bundle.py`, `tests/test_tracking.py`.

**Interfaces:** `seal_bundle(staged: Path, destination: Path, config: StudyConfig) -> BundleIdentity`;
`verify_bundle(path: Path) -> BundleIdentity`;
`load_bundle(path: Path) -> BundlePredictor`, with
`BundlePredictor.predict(texts: Sequence[str]) -> tuple[Inference, ...]`;
`log_run(root: Path, purpose: str, identity: RunIdentity, receipt: Mapping[str, object], artifacts: Sequence[Path]) -> str`;
`download_bundle(root: Path, run_id: str, destination: Path) -> BundleIdentity`.

- [x] **Step 1 - write failing tests.** For a locally initialized fixture bundle assert
  `verify_bundle(original).model_id == verify_bundle(relocated).model_id` and equal
  inference after reload. Mutating `vocab.txt`, model weights, labels or saved truncation
  settings must fail verification before model loading; missing files must fail too.
  `test_mlflow_download_matches_original` logs to temporary SQLite/artifacts and asserts
  every downloaded digest/model_id equals the original. Do not use autologging/network.
- [x] **Step 2 - observe RED.** Run
  `uv run --frozen --no-sync pytest tests/test_bundle.py tests/test_tracking.py -q`.
- [x] **Step 3 - implement identity/tracking.** Save actual safetensors/model/tokenizer
  settings plus label mapping. Canonical JSON uses UTF-8, sorted keys, compact separators,
  finite values and no absolute paths; manifest covers relative files/digests/sizes and
  excludes itself. Verify exactly the expected file inventory before loading. Derive
  model_id from the manifest map, labels and inference settings as specified, independent
  of directory or report metadata. Explicit MLflow run-purpose/status/tags capture source,
  lock/config/freeze hashes, input provenance, seed/runtime/RSS/metrics and artifact links.
- [x] **Step 4 - verify GREEN.** Run Task 4 and canonical checks. Repeat bundle readback
  with Hub networking unavailable, without requiring a running MLflow server. Compare
  local downloaded bytes, not only MLflow metadata or a logged path.
- [x] **Step 5 - commit.** Commit named files as
  `feat: retain verified model bundles and local experiment lineage`.

### Task 5: Strict local saved-model API and actual process restart

**Files:** Create `api.py`, `tests/test_api.py`, `tests/test_api_process.py`.

**Interfaces:** `create_app(bundle_dir: Path) -> FastAPI`; startup calls Task 4's verified
local loader. `GET /health -> {"status": "ready", "model_id": str}`;
`POST /predict` accepts exactly `{"text": str}` and returns
`{"label": str, "model_id": str, "truncated": bool}`.

- [x] **Step 1 - write failing tests.** Assert numeric/null text, extra keys, malformed
  JSON, whitespace-only, empty and 2,049 raw characters produce 4xx. A valid 2,048-character
  case succeeds and returns the required keys without a confidence field. With the
  fixture tokenizer assert 62 content tokens plus specials are untruncated, 63 content
  tokens are truncated and final token length is 64. Include decomposed Unicode/outer
  whitespace and compare offline/HTTP labels using the exact normalized token IDs.
  `test_actual_process_restart_retains_identity` starts Uvicorn on an owned free test
  port, calls HTTP, stops/waits, restarts and compares model_id/labels with no Hub access.
- [x] **Step 2 - observe RED.** Run
  `uv run --frozen --no-sync pytest tests/test_api.py tests/test_api_process.py -q`.
- [x] **Step 3 - implement the minimal app.** Strict Pydantic input, saved NFC/strip and
  tokenization policy, local verified startup, evaluation mode and no gradients. Compute
  truncation from pre-truncation token count including specials. Highest-logit ties choose
  lowest label index. No request-time acquisition, MLflow lookup or extra prediction mode.
- [x] **Step 4 - verify GREEN.** Run Task 5 and canonical checks; include failed startup
  with missing/tampered bundle. Inspect real process cleanup and retained fixture evidence.
  Tests use isolated free ports, never steal an existing listener on 8024.
- [x] **Step 5 - commit.** Commit named files as
  `feat: serve verified support-intent artifacts with restart parity`.

### Task 6: One explicit real-data study and bounded acceptance closeout

**Files:** Create `study.py`, `cli.py`, `scripts/acceptance.py`,
`tests/test_study.py`, `tests/test_acceptance.py`, `docs/model-card.md`,
`docs/study-report.md`; update README/ATTRIBUTION/implementation notes/config metadata.

**Interfaces:** `main(argv: Sequence[str] | None = None) -> int` in `cli.py`, exposed as
`p24`; `run_stage(stage: str, config: Path, root: Path) -> dict` in `study.py` supports
explicit stages `acquire`, `prepare`, `baseline`, `feasibility`, `train`, `evaluate`,
`readback`, `serve-check`, `report`. No all-in-one command silently advances spending
or study stages. `verify_acceptance(root: Path) -> dict` returns A1-A8 outcomes with
source/identity-bound evidence paths and explicit FAIL/NOT_RUN, never guessed PASS.

- [x] **Step 1 - write failing orchestration tests.** With fixtures assert final test
  is inaccessible before a completed measured receipt/frozen bundle; a second actual
  evaluation claim or changed model/config/freeze is refused without replacing results.
  Missing A4/A6/A7 receipts must produce NOT_RUN/FAIL, not a complete report. Assert an
  inferior encoder still completes if all mechanical evidence passes; seeded synthetic
  runs remain labeled software checks, never actual pretrained quality evidence.
- [x] **Step 2 - observe RED.** Run
  `uv run --frozen --no-sync pytest tests/test_study.py tests/test_acceptance.py -q`.
- [x] **Step 3 - implement orchestration and verify software GREEN.** Define p24 CLI
  stages with config/root arguments and ownership checks. Reuse earlier interfaces;
  keep prepare freezes, actual attempt claims and evaluation claim durable and separate.
  Record baseline attempt once, require its success before feasibility, and refuse study
  continuation after a failed actual gate. Implement A1-A8 reconciliation, reproducible
  safe reporting/model card and fixtures. Run canonical checks before any real acquisition
  or study claim, commit executable/config source as
  `feat: orchestrate the bounded transformer study and evidence gates`, then verify clean
  source and capture its commit plus tracked-input hashes. Reporting-only later commits
  retain the measured executable/config identity; no favorable silent rerun.
- [x] **Step 4 - acquire and freeze actual inputs.** Recheck host capacity/ownership and
  coordinate CPU/RAM use; record current samples. Run
  `uv run --frozen --no-sync p24 acquire --config config/study.json --root .local/study`,
  inspect actual licence/source/hash/time receipts, then run the corresponding `prepare`
  stage. Resolve privacy flags locally or exclude them, freeze manifests/exclusions/counts,
  and verify A1/A2 before model fitting. Stop on failed counts/provenance. No parameters
  may be changed based on these failures without an explicitly recorded design revision.
- [x] **Step 5 - fit the baseline once.** Run the `baseline` stage with the same arguments;
  retain actual pipeline digest, training-only vocabulary evidence, local MLflow run and
  convergence outcome. It must not read/score final test. Stop on failure, preserving data.
- [x] **Step 6 - run one feasibility attempt.** Run `feasibility` once. Inspect the actual
  claim/receipt, finite loss, encoder changes, steps/time/RSS and <=45-minute projection.
  A missing/failed receipt stops the study. Diagnostic fixture corrections do not clear
  or recreate the consumed actual slot. Do not continue automatically after a failure.
- [x] **Step 7 - run one fresh measured fit.** If the feasibility gate passes, run `train`
  once under the external watchdog. Verify fresh pinned-base initialization, three full
  epochs, final-epoch selection, both-layer deltas and bounded runtime/RSS. Seal and log
  the complete artifact; partial weights do not enter final-test evaluation. Retain all
  receipts and failures; do not refit to improve validation performance.
- [x] **Step 8 - evaluate both arms once.** Run `evaluate` only after source/config/freeze/
  final-bundle identities are fixed. Write the paired prediction files and six-label
  metrics, counts/confusion/errors and truncation counts; preserve results without tuning.
  Independently recompute metrics from those same saved prediction files and verify exact
  IDs, counts and numerical agreement. This readback does not perform test inference again.
- [x] **Step 9 - demonstrate artifact/HTTP readback.** Run `readback` for a real MLflow
  download/hash check, then `serve-check` for actual selected-bundle offline/HTTP parity
  on independently authored examples and API stop/restart at owned 127.0.0.1:8024. Verify
  rejected inputs and stable model_id/labels without network. No second fit/test study.
- [x] **Step 10 - close out and commit evidence.** Run `report`, canonical software checks
  and `uv run --frozen --no-sync python scripts/acceptance.py --root .local/study`.
  Require A1-A8 actual outcomes with source-bound receipts, not a hardcoded passing count.
  Record actual samples, errors and limitations in the report/model card; inspect safe
  files for raw messages/identifiers/model binaries before staging. Stop owned API
  processes, retain/hash-check ignored evidence and update planner tracker/handover with
  actual results or the precise stop branch. Commit only safe docs/manifests/plots as
  `docs: record bounded support-intent study outcomes`. No public push or automatic
  integration/worktree removal; preserve ignored evidence before any later integration.

## Self-review and execution handoff

Self-review completed against all design sections. A1/A2 map to Task 1; A3 to Task 2;
A4 to Task 3; A5 to Tasks 2/6; A6 to Task 4; A7 to Task 5; A8 to Task 6. Each Review
Focus item has an explicit failing-test step in its owning task. Shared types/signatures,
normalization/label order, limits, fixed selection and no-network fixture boundaries were
checked for consistency. The real study is deliberately sequenced after software checks;
data, attempts and holdout cannot be consumed by the canonical command.

Recommend **Native execution**: six tasks share frozen-data, receipt and bundle contracts,
and the real study is necessarily sequential. One implementer plus one fresh whole-branch
review keeps this demonstration bounded. Subagent-driven execution is the alternative:
fresh implementer/reviewer per task plus final review, with higher context/review cost.
Native execution is underway following explicit plan approval; the original handoff was:
review the plan and choose the method before implementation.
