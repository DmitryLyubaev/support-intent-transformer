# Publication Follow-up Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. User explicitly said Proceed, publish and push everything required. Preserve Native execution; no additional approval round.

**Goal:** Fix both deferred minors and publish safe source/reports with passing hosted software CI.
**Architecture:** Existing study contracts plus a metadata-only enrichment command, strict feasibility train loader, one Windows CI job.
**Tech Stack:** Unchanged Python3.12/uv/CPU PyTorch/MLflow/FastAPI stack and lock.
**Spec:** [Publication follow-up](../specs/2026-11-01-publication-followup-design.md).

## Global Constraints

- No actual training, test inference, API demonstration repeat or new allowance.
- Preserve original measured source4dc18f9, predictions/metrics/receipts and model identity.
- Publish personal DmitryLyubaev/support-intent-transformer. No raw data/model/MLflow/environment files.
- Exact-lock software tests only; existing primary environment may supply dependencies with worktree src first on PYTHONPATH.
- User authorizes local integration/public creation/push/hosted CI and planner commit/push. No new permission menu.
- Native isolated worktree and one fresh final review before first public push.

## Review Focus

- Enrichment of mismatched or missing runs must refuse before any tracking writes. Task1.
- Enrichment must not mutate receipts, reports, model weights or claims, and replay must not duplicate validation metrics. Task1.
- Both feasibility loader sites must work with validation/test unavailable. Task1.
- CI must not acquire actual data/model weights or run real study stages. Task2 and hosted readback.
- Publication must exclude ignored files and identifying message text throughout all pushed Git history. Publication audit.

### Task 1: Deferred-minor fixes

**Files:** src/p24/enrichment.py, study.py, cli.py, tests/test_enrichment.py, tests/test_study.py, README.md, AGENTS.md.
**Interfaces:** enrich_runs(root:Path, project_root:Path)->dict logs existing bound runs only. `p24 enrich` dispatches separately from source-current fitting gates and performs no study-root creation.
- [ ] Write failing metadata-only real local MLflow tests for retained config/source/environment/evaluation links, unchanged identity/receipts/status, mismatched-run refusal, and repeated metric deduplication.
- [ ] Write failing worker and parent-projection fixtures that reject validation/test loading during feasibility while permitting measured validation.
- [ ] Run focused tests, observe expected failures. Implement minimal fixes. Expected: all focused PASS.
- [ ] Commit fixes after full canonical check. Expected: lock/lint/format and all fixture tests PASS, no actual claims touched.

### Task 2: Publication and CI preparation

**Files:** .github/workflows/ci.yml, README.md, docs/publication-followup.md, safe audit/verification receipt.
- [ ] Add SHA-pinned checkout/setup-uv/upload-artifact actions and one windows-2025 job. Download only exact known Python runtime bytes, validate size/SHA, extract, create venv, locked sync, then canonical check with JUnit output.
- [ ] Emit source-bound result JSON and upload only JUnit/receipt. Read-only GitHub token, no credentials persisted, bounded timeout, no deployment or real study commands.
- [ ] Document retrospective limits and prospective fixes; keep original reports/card unchanged. Run canonical checks and retained-data acceptance through primary evidence read-only, preserving bytes/mtimes. Expected: PASS without fitting/inference.
- [ ] Commit and obtain one fresh review of this whole follow-up branch. Fix Important/Critical once with RED/GREEN; defer Minor findings.

## Publication after review

Audit all historical blobs/current safe files, personal identities and forbidden artifact paths.
Integrate locally as already authorized, create the public personal repository with secret
scanning/push protection, push main, wait for hosted CI and verify exact-source result.
Preserve failures honestly and fix software only if needed. Archive owned scratch/logs,
clean up the follow-up worktree, update tracker/handover/phase5 and push planner updates.
