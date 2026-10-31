# Publication follow-up

Dmitry explicitly requested both deferred-minor fixes, public GitHub source/safe reports,
hosted CI, and all required commits/pushes on 1 November 2026, with "Proceed". This is
execution authorization; no new study, model selection or paid/cloud deployment is added.

Keep the existing Python stack and dependency lock. Add a tracked `p24 enrich` operation
that copies the private helper's metadata-only MLflow enrichment into the reproducible
workflow. Validate historical study identity and existing run identities before logging;
never create a run, fit, infer, rewrite original receipts/reports, or force run status.
Repeated enrichment must not duplicate validation metric points.

Feasibility must load only the train split in both its worker and projection path.
Measured training continues to load training and validation. Pin these with fixtures
that fail on validation/test access. Preserve the original historical feasibility's
documented unused-validation read; the fix is prospective, not a retroactive claim.

Use one Windows GitHub Actions job with exact uv0.10.7, the already validated official
Python3.12.16 archive and its recorded SHA-256, the unchanged lock, and the canonical
software-only checks. No actual study commands or model/dataset downloads. Publish
only code, attribution, safe aggregate reports/plots and software CI receipts. Local
models, data, MLflow, environment snapshots and original evidence remain ignored.

The alternative of adding another platform or training in CI expands the agreed scope.
The existing native Windows target is sufficient. The remaining builder details are
small implementation choices, not a new design/approval round.
