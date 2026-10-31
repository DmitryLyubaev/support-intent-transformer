# Project 24: Six-intent transformer adaptation and saved-model serving

Date: 31 October 2026

Status: approved by Dmitry on 31 October 2026 after the fresh builder check. Dmitry
selected approach A, then explicitly approved this written design. The detailed choices
below originated as assistant proposals and are now accepted for the bounded local scope.
Implementation planning and execution-method selection are next; implementation has not started.

## Intent and completion boundary

Build one small personal portfolio study demonstrating actual adaptation of a pretrained
transformer encoder with PyTorch/Hugging Face, a fair text-baseline comparison, local
experiment lineage and serving the saved tokenizer/model. This is a small project with
a full but bounded design and a subsequent implementation plan.

Completion means actual bounded training, a frozen held-out report, model card, saved
artifact identity, reproducible instructions and local API parity/restart evidence.
Worse performance than the baseline is a valid completed study. A failed feasibility
attempt is an honest stop, not evidence that the training milestone was completed.

Non-goals: production use, real support operations, customer data, generative LLM or
instruction tuning, GPUs/cloud trainers, LoRA, model tournaments, repeated seed trials,
hyperparameter sweeps, calibrated uncertainty, unknown-intent detection, UI, hosted
MLflow/registry, database services, automatic retraining and changes to other projects.
CV/profile editing and public publication are separate requests under RULES.md.

Anti-goals: silently choosing favorable classes or test results, tuning on final test
rows, replacing dataset labels with assistant labels, calling head-only training encoder
fine-tuning, or treating synthetic software tests as measured classification quality.

## Approaches and ownership of decisions

- **A, selected by Dmitry:** six fixed support intents, a TF-IDF/logistic-regression
  baseline and full BERT-Tiny fine-tuning on CPU. This is the smallest proposed route
  that demonstrates pretrained encoder weights changing and saved-model serving.
- **B, not selected:** the same task using DistilBERT. A larger representation would
  increase compute/memory needs; no task-specific quality advantage has been measured.
  It is not an automatic fallback or an additional experimental arm.

Everything below that was not in the approach choice originated as the assistant's call.
Dmitry's design approval accepts these calls together; implementation follows a written plan.

## Stack and isolation

Use a separate local repository proposed at `E:/Projects/support-intent-transformer`.
Check absence/ownership before creating it; never replace an existing directory.
Python 3.12, `uv`, CPU PyTorch, Transformers/Datasets, safetensors, scikit-learn,
local MLflow, FastAPI/Pydantic, pytest and Ruff are sufficient. Use a small explicit
PyTorch training loop rather than adding Trainer/Accelerate machinery. Resolve compatible
stable package versions into a lockfile before acquisition/training; exact versions,
wheel availability and import compatibility are builder verification tasks, not proven now.

Run natively on Windows in a project environment. Do not change global Python, install
CUDA, restart WSL/Docker or reuse another project's environment. If compatible CPU wheels
or imports fail, record the failure and stop rather than changing execution platforms
silently. Docker serving is a deferred optional extension, not a completion requirement.

Keep dataset/model caches, raw rows, prepared rows, trained artifacts, MLflow state and
raw logs under this project's ignored `.local/`. Use experiment `p24-support-intent`,
one local SQLite tracking file and local artifact storage through MLflow's direct tracking
API. No MLflow server/UI or registry is required. The API binds `127.0.0.1:8024`, only
after confirming the port is free. Stop only owned processes; retain evidence and artifacts.

## Pinned source inputs and permissions

Metadata-only checks on 31 October 2026 returned:

| Input | Repository | Immutable revision | Acquisition files |
|---|---|---|---|
| Dataset | `PolyAI/banking77` | `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6` | `README.md`, `data/train-00000-of-00001.parquet`, `data/test-00000-of-00001.parquet` |
| Base model | `google/bert_uncased_L-2_H-128_A-2` | `30b0a37ccaaa32f332884b96992754e246e48c5f` | `README.md`, `config.json`, `vocab.txt`, `model.safetensors` |

[PolyAI's original repository](https://github.com/PolyAI-LDN/task-specific-datasets)
documents the banking benchmark. The [pinned dataset card](https://huggingface.co/datasets/PolyAI/banking77/blob/1846984189f2bc0ac51c6fc3a600fa9e1714e0c6/README.md)
declares CC BY 4.0; the [pinned Google model card](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/blob/30b0a37ccaaa32f332884b96992754e246e48c5f/README.md)
declares Apache 2.0. Preserve attribution, citation, relevant licence notices and a
description of changes. Do not imply creator endorsement. The dataset card leaves
collection/bias details incomplete; it is not proof that all rows lack identifying content.

After implementation approval, fetch only the named files from those revisions. Record
URL, revision, actual UTC retrieval time, byte size and SHA-256 for each. Reject missing
or changed inputs rather than resolving a moving `main`; do not execute remote dataset
scripts or model code. Use parquet via Datasets and safetensors for weights. Load and
save the uncased WordPiece tokenizer from the pinned vocabulary/config, including actual
special-token settings, and verify its compatibility before training. Later runs and API
startup use local files only. No artifact bytes or local dependency compatibility have
been checked by writing this design.

Raw/prepared rows, model binaries and MLflow stores remain ignored and private locally.
Git may hold code, lockfile, configs, attribution, manifests without message text, aggregate
metrics, plots, model card and independently authored demonstration fixtures. Review
candidate report text before sharing. Dataset/model publication and a public source push
need their own decision; design approval does not authorize them.

## Task, preparation and split rules

Use exactly these labels, in this output order:

1. `card_arrival`
2. `card_delivery_estimate`
3. `card_not_working`
4. `lost_or_stolen_card`
5. `declined_card_payment`
6. `pending_card_payment`

These are a deliberately small related-intent subset, not a BANKING77-wide benchmark.
Use upstream labels without relabeling. Classification is closed-set: any valid input
gets one of the six labels, including nonsense or unrelated text. No confidence or
out-of-scope claim follows. Do not invent a seventh unknown class or threshold.

Preparation is deterministic and frozen before any model run:

- Preserve upstream split and zero-based source-row index as immutable row identity.
  Filter to the fixed labels. Reject invalid/empty text; identify possible email, phone,
  account/card-number or other identifying content for local review. Exclude a flagged
  record if it is identifying or cannot confidently be cleared. Log reason/count/row ID
  without copying identifying text. Do not redact it into a new labeled example.
- Model input is Unicode NFC with surrounding whitespace stripped. Training and serving
  use exactly this operation. Separately normalize NFKC, casefold and collapse whitespace
  for duplicate checks. Never fit normalization/vocabulary from validation or test data.
- Build groups across the selected upstream train/test rows: normalized equality, or
  character five-gram set Jaccard similarity at least 0.85 when both normalized strings
  are at least 20 characters. Use connected components. Exclude entire components with
  conflicting labels. This is a defined leakage check, not proof that every semantic
  paraphrase or related author is detected; no author/time grouping metadata is assumed.
- For each component touching upstream test, retain only its lowest-index test row and
  exclude every training member. Never move a test row into development data. Within
  training-only components collapse exact normalized duplicates to lowest-index rows;
  keep near-duplicate members together. Count every exclusion by reason and split.
- From the remaining training-only groups, select up to 160 rows per class (at most 960
  total), scanning groups ordered by SHA-256 of `42|label|group_id`. Group ID is the
  lexicographically smallest immutable row ID. Add a complete group only if it fits
  the per-class cap; otherwise skip it and record that fact. No selection uses scores.
- Within each class, assign complete selected groups in that same order to validation
  until at least 20% of its selected rows are assigned; the remainder is training.
  Require at least 40 training, 10 validation and 10 retained test rows per class.
  If this fails, stop with actual counts; do not change labels/split/threshold silently.

Write sorted manifests, preparation-code/config hashes, all exclusion counts and actual
class counts. Test text access during preparation is only for structural/privacy/leakage
checks, not prediction/error inspection. Once frozen, training/feasibility commands load
only train/validation paths; the evaluation command alone loads final test rows.

## Fixed baseline and training recipe

Fit one scikit-learn pipeline using only training rows: word TF-IDF unigrams/bigrams,
maximum 10,000 features, default lowercasing, then logistic regression with `C=1`,
`max_iter=1000`, no class weighting and seed 42. No vocabulary refit on validation/test,
grid search, baseline retry or post-test tuning. Record convergence/failure honestly.

Use the pinned pretrained BERT encoder, add a newly initialized six-class head and
make all encoder/head parameters trainable. Record actual trainable/total counts. Use
seed 42 for Python/NumPy/PyTorch and deterministic CPU operations where supported;
record the actual settings and exceptions rather than promising cross-platform bit parity.

Fixed recipe: three epochs, batch size 16, token length 64 including special tokens,
dynamic padding, truncation at that limit, AdamW at learning rate `5e-5`, weight decay
0.01, no scheduler/warmup, cross-entropy, gradient clipping at norm 1.0, shuffled training
loader with seeded generator and zero data-loader workers. Limit intra-op/BLAS threads
to four and inter-op threads to one. Log validation macro-F1/loss after each epoch for
description only. Select the final third-epoch artifact, regardless of validation results;
there is no early stopping or best-checkpoint search. Record truncation counts per split.

Before/after evidence must show a nonzero change in pretrained encoder tensors, including
at least one attention/feed-forward weight in each of the two encoder layers. Head-only
changes are insufficient. Retain canonical tensor digests and maximum absolute deltas
for those weights; an entire-file digest alone cannot establish encoder adaptation.

## Compute, failure and experiment bounds

The observed host has four physical CPU cores and 15.9 GiB RAM; the earlier 7.3 GiB free
RAM/83.5 GiB free disk are dated samples, not reserved capacity. Coordinate with Projects
22/23 before running. Recheck at least 6 GiB free RAM and 10 GiB disk before setup/runs;
otherwise wait and report, without stopping another project's processes.

One training-only feasibility attempt is allowed after implementation approval: at most
50 optimizer steps or 10 minutes, whichever comes first. Use the fixed recipe and a
training-only subset; never score final test. Record throughput and sampled process-tree
peak RSS, then discard these candidate weights. Proceed only if finite loss/encoder
changes are observed, RSS stays within 4 GiB, and measured step timing projects the full
three-epoch run within 45 minutes, leaving margin under the measured-run cap.

One fresh measured encoder run starts again from the pinned base model and seed, capped
at three complete epochs or 60 minutes. An external owned-process watchdog bounds the
run, including validation/checkpoint saving. Sample process-tree RSS at least every
five seconds; stop above 4 GiB. These are ceilings, not runtime or memory promises.
Keep an append-only local attempt ledger with source/config/data identities, start/end,
purpose, steps, status and failures; identify and prevent an accidental second measured
run. Partial weights do not qualify as the selected final artifact.

No extra experiment, automatic retry, batch/length/learning-rate adjustment, additional
seed, DistilBERT or paid fallback is authorized by this design. Deterministic software
fixes may be verified with tiny fixtures, but replacing/repeating an actual feasibility
or measured attempt needs an explicit revised allowance and retained history. If the
baseline, data gate, feasibility or measured run fails, preserve evidence and report
the narrow limitation. Do not begin final-test evaluation on an incomplete model.

## Evaluation, lineage and saved artifacts

Freeze source/config, selected final model and split identities before final-test access.
Evaluate the baseline and reloaded final transformer on exactly the same retained test
rows once, preserving both predictions and original labels privately. A shared evaluator
computes macro-F1 over all six labels, per-class precision/recall/F1/support, confusion
matrix and accuracy as secondary context. Use zero for undefined precision/recall and
state that policy. Publish actual counts, dated runtime/environment and unfavorable
results; a single seed/small subset supports no general superiority claim.

Error analysis takes the lowest row IDs among mistakes, at most two per class per arm.
Keep literal rows private; describe error patterns or use clearly marked independently
authored examples in a shareable report. This is assistant inspection, not blinded human
validation. After test scoring, no tuning/reselection; any correction/rescore preserves
original results and needs an explicit retest record. Recomputing metrics from frozen
prediction files is allowed and is not a second training/test-inference study.

MLflow records baseline, feasibility and measured runs with explicit run purpose/status,
Git/source and lockfile hashes, source revisions/file hashes, split manifest/config hashes,
seeds, preprocessing, full parameter selection, environment, runtime, validation metrics
and final evaluation links. Use explicit logging instead of autologging extra inputs.
Log the saved transformer/tokenizer bundle as a local artifact, download it to a fresh
local directory via MLflow, and verify every file digest. No model registry is needed.

The bundle contains safetensors weights, model config, complete saved tokenizer, label
mapping and preprocessing/inference settings. Its manifest maps sorted relative paths
to SHA-256 digests and sizes, excluding the manifest itself. `model_id` is SHA-256 of
canonical UTF-8 JSON of that map, label mapping and inference settings. Verify contents
and identity at load; changing metadata outside this bundle does not change model_id.
Save the baseline pipeline locally as trusted project-generated data. Do not load
third-party pickle artifacts or publish binaries by default.

Retain model card and source-bound report with licences, six-label scope, actual split
counts, exclusion/truncation policy, metrics/errors, compute/run history, artifact identity,
reproduction commands and limits. Reproduction means pinned source/preparation/training
instructions and verified artifact/metric readback; no second full fit is mandatory or
implied to have run. Training replay consumes a separately approved run allowance.

## Minimal API behavior

Load the actual selected local bundle once at startup, verify its manifest, set evaluation
mode and disable gradients. No Hub lookup, training, scoring, dynamic model upload or
MLflow server dependency at request time. Missing/inconsistent artifacts prevent startup.

- `GET /health`: ready only after artifact verification/load, with model_id.
- `POST /predict`: strict JSON object containing only string `text`; require 1-2,048
  characters before normalization and nonempty text after normalization. Reject malformed
  JSON, wrong types, extra fields, empty/whitespace-only and oversized text with 4xx.
  Return the highest-logit label, model_id and `truncated` boolean. No confidence score.

Use exactly the saved normalization, WordPiece settings and 64-token truncation. Compare
token IDs and label for a fixed independent demonstration set against direct offline
bundle inference. Stop/restart the actual API process and verify the same model_id and
labels. This demonstrates local artifact serving and process restart, not production
concurrency, authentication, OOS rejection or an SLA. No latency benchmark is required.

## Acceptance and verification

| Case | Required evidence |
|---|---|
| A1 - provenance | Immutable source files/hashes/retrieval dates, licence attribution, selected-record review and actual exclusion counts |
| A2 - split | Frozen six-label manifests, caps/minimum counts and zero defined duplicate/group overlap across train/validation/test; actual bad-fixture rejection |
| A3 - baseline | Actual fitted training-only vocabulary/model and held-out prediction identities, with convergence outcome |
| A4 - adaptation | Completed bounded fresh three-epoch fit; pretrained tensor changes in both layers; seed/environment/runtime/RSS/attempt history |
| A5 - evaluation | Both arms scored on identical frozen rows, reproducible metrics from retained predictions, per-class errors and honest no-improvement handling |
| A6 - lineage | MLflow run identities and hash-equal artifact download; saved-bundle tamper/missing-file rejection and model card |
| A7 - serving | Actual saved-bundle token/prediction parity, input boundary checks and actual process stop/restart with stable model_id |
| A8 - reproducibility/finish | Pinned setup/preparation/training/serving instructions, source-bound report, preserved private data/artifacts and no owned API process left running |

Software checks use tiny independently authored fixtures and a locally initialized tiny
encoder where needed, with no network or large download. Cover leakage boundaries,
training-only vocabulary, metric correctness, saved-bundle checks, tokenizer parity and
API limits/restart. These tests do not prove actual BANKING77 quality or pretrained
fine-tuning. A local canonical command runs lock/lint/format and focused software checks;
a separate explicit study command handles actual data/training/evaluation. Hosted CI is
not required for this local milestone and must never silently trigger real training.

## What remains to the builder and approval boundary

The builder chooses package patch versions, module layout, CLI syntax, graph rendering,
small-fixture mechanics and MLflow schema/API details while preserving the contracts
above. Verify local source/license bytes, dependency imports, tokenizer loading and
resource availability during the plan's explicit setup/acquisition steps. A failed gate
does not authorize expanding the design. No product repository or implementation yet.

One fresh read-only ML/software builder check on 31 October returned "nothing important
left". No review-driven design change or extra question was needed. This checks planning
completeness, not data suitability, executable correctness or training feasibility.

Design approval is recorded above. Next: write/self-review the implementation plan and
have Dmitry review it and choose the execution method before implementation. Local design
approval does not authorize public publication, cloud resources, provider calls or paid compute.
