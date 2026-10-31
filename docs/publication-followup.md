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

Publication and actual hosted results will be recorded after their exact-commit checks.
