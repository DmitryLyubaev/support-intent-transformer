# Publication and deferred-minor follow-up

1 November 2026. Dmitry authorized public source/safe-report publication, hosted CI,
both deferred-minor fixes and all required commits/pushes.

The tracked `p24 enrich` command now reproduces the private helper's MLflow metadata
enrichment for existing historically bound runs. Its preflight checks source/config/
freeze and all three run identities before opening a writing client. It logs provenance,
configuration, environment, validation metrics and final evaluation links. It does not
create runs, change statuses, rewrite reports/receipts or claim another training/evaluation
allowance. Duplicate validation metric points are skipped. The command is exercised
against real local SQLite/artifact stores with synthetic metadata in software tests.

Both feasibility paths now read only the training split: the worker and parent timing
projection. Tests fail on validation/test access; the measured worker still uses its
validation split. This is a prospective source correction. The original 31 October
feasibility did open unused validation, as honestly recorded in review-closeout.md.
Its original weights, receipts and outcome were not replaced or repeated.

## Publication boundary

Code, attribution, original safe aggregate metrics/plots, reports and software CI results
may be public. Raw/prepared messages, model and baseline binaries, MLflow, environment
snapshots, original evidence and archives remain private under ignored `.local/`.
The historical study report and model card remain unchanged, bound to their original
receipts. Public CI does not establish full training reproduction, improved model quality,
human validation, production operation or a later-source actual study.

## Hosted software checks

One Windows job uses the same validated Python3.12.16 archive and uv0.10.7, exact locked
dependencies, lock/lint/format and all software tests with the inherited external-network
guard. The job downloads the Python runtime and dependencies only. It never acquires
BANKING77/base weights or invokes an actual study stage. Only JUnit and a source-bound
software receipt are uploaded. Dataset accuracy remains the unchanged 31 October result:
baseline macro-F1 0.896042 versus encoder 0.466797 on 239 paired retained test rows.

## Pre-publication verification and review

Native canonical checks passed 94 tests with two existing upstream deprecations
(pytest114.12s). The actual retained A1-A8 readback passed while all155 original evidence
entries kept their contents and modification times. No actual enrichment, fitting,
held-out inference or API demonstration was repeated. One fresh independent review of
`54079d3..33b73af` found zero Critical, Important or Minor issues and independently passed
the12 focused tests. No second review was needed.

The pre-publication history audit through `33b73af` covered115 blobs,17commits and70
tracked files. Selected credential/employer patterns, forbidden private artifact paths
and exact matches against597 retained message strings of at least40 characters had
zero findings. All authors/committers were the personal identity. The reviewer also
inspected the single aggregate image. This is not an exhaustive semantic PII guarantee.

## Follow-up rulings

1. Reuse the ignored Git worktree fallback after the app's earlier source-root rejection.
   Cost if wrong: the app cannot manage its lifecycle; explicit Git paths are required.
2. Use the already validated, hashed Python3.12.16 archive in Windows CI because the
   official actions/python-versions manifest lacks a Windows build of that patch.
   Cost if wrong: archive availability can block CI; the job must not substitute bytes.
3. Keep full A1-A8 acceptance separate from metadata enrichment. Cost if wrong: callers
   must run the independent verifier before treating metadata as complete study evidence.
4. Target a quiescent local tracking store. Cost if wrong: simultaneous writers can
   produce partial or conflicting metadata; no multi-writer service is claimed.
5. Do not add transactional rollback for storage/malformed-metadata logging failures.
   Cost if wrong: failed enrichment can be partial; original study receipts remain intact.
6. Preserve acquisition's unfrozen identity; final preparation/run/evaluation bind the
   final freeze. Cost if wrong: acquisition alone cannot prove final split lineage.
7. Verify hosted execution, security settings and exact pushed source after publication.
   Cost if wrong: local software checks cannot substantiate hosted success.
8. Add no training trial, quality target, production service or stronger dataset-privacy
   claim. Cost if wrong: the inferior original encoder and bounded guarantees remain.
9. Treat pattern/exact-message audits as bounded checks. Cost if wrong: unknown
   identifying strings are not ruled out by a semantic privacy proof.

Publication and actual hosted results will be recorded after their exact-commit checks.
