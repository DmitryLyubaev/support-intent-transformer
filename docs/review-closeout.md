# Local study review closeout

31 October 2026. One fresh whole-branch review of `b085ef1..076318d` found zero
Critical, three Important and two Minor findings. The reviewer independently matched
the saved predictions' metrics, both pretrained tensor changes, original/downloaded
bundle identity and the principal actual evidence identities. No second review occurred.

The measured executable remains `4dc18f9b31bb39ff1cd96fd967008a0ecc541284`.
Later verifier fixes do not claim that training ran on the new source. Original claims,
receipts, predictions, metrics, model, report and model-card receipt remain unchanged.
The original acceptance output is preserved; the corrected independent verification
is separately recorded in [acceptance-after-review.json](evidence/acceptance-after-review.json).
It records the measured snapshot and current verifier hash separately.

## Important fixes in one pass

1. Source receipts now require schema version, every provenance field, exact immutable
   URL and a parseable UTC retrieval timestamp. Seven regressions first accepted bad
   receipts and failed, then passed after the fix. Actual receipts already had these
   fields, so their original provenance was not invalidated.
2. Acceptance anchors the measured source inventory and bytes to its preserved Git
   commit, reconciles config/lock/freeze identity across records and claims, checks model
   IDs and binds assistant review to the frozen candidate and fixed sample IDs. Eleven
   mixed-source/model/review regressions first produced incorrect PASS, then passed
   after correction. A canonical JSON comparison also resolved tuple/list representation
   differences during the first fix verification.
3. Evidence readback now reads parquet through the already installed PyArrow reader,
   without building a Datasets cache. MLflow verification queries its existing SQLite
   database in immutable read-only mode; absent or active/uncheckpointed stores are
   reported without creation or migration. Cache-write/missing-reader regressions and
   a verifier-created-database regression were observed before correction. All 155
   original evidence entries retained their contents and modification times during
   the corrected actual A1-A8 check.

These are verifier repairs, not an extra experiment. There was no refit, model
reselection, relabeling, final-test inference or HTTP demonstration rerun.

Final canonical verification passed **87 tests**, zero failures/skips, plus lock,
lint and format checks in 103.10 seconds. The same two upstream deprecations remain:
Starlette/httpx TestClient and MLflow/SQLAlchemy noload. Corrected A1-A8 all passed;
the separately recorded verifier hash matches the exact verified source files.

## Deferred minors

- Full local MLflow provenance/config/environment and final-evaluation enrichment was
  appended by the private ignored `.local/close_lineage.py` helper, retained in the
  private archive. The three actual runs contain the information, but the tracked
  source workflow lacks that explicit metadata-only closeout operation.
- Feasibility loaded the validation split through `read_training()` even though fitting
  used only the first 128 training rows and did not use validation loss, predictions or
  selection. The strict planned loader boundary is therefore unproven. Changing that
  loader is deferred; no actual run was repeated to hide the discrepancy.

## Review scope rulings

The reviewer set aside these behaviors; the executor explicitly retains the limits:

1. Production authentication, concurrency, availability and deployment remain outside
   the loopback demonstration. Cost if wrong: these checks cannot justify production use.
2. Installation/wheel portability remains limited to the verified native Windows target.
   Cost if wrong: another host may not install or reproduce this runtime.
3. No extra quality trial or alternative recipe is added. The inferior encoder completes
   the approved experiment. Cost if wrong: this artifact remains a weak classifier.
4. Defined duplicate/privacy checks and assistant sampling do not eliminate exhaustive
   PII, semantic or pretraining overlap risks. Cost if wrong: residual sensitive content
   or overlap can remain in the private data and limit interpretation.
5. Content hashes identify the recorded bundle; signatures and authenticated distribution
   remain outside scope. Cost if wrong: an actor replacing a complete bundle and all its
   identity records is not excluded by this local content-identity mechanism.

The earlier four implementation rulings and their costs remain in
[implementation-notes.md](implementation-notes.md). The working branch and private data
are retained pending the owner's integration choice. No product publication is authorized.
