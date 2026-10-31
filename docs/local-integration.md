# Local integration

31 October 2026. Dmitry selected local integration. Primary `main` fast-forwarded from
`b085ef1` to reviewed source `e54b68c1130e53bdd9459d9309892d44c3221d63`. There is no
remote configured. No product push, hosted CI or public repository creation occurred.

The primary checkout uses a new project-local Python 3.12.16 environment with the
unchanged committed dependency lock, installed offline from the existing package cache.
No global Python, PATH, trust or security setting changed. Setup waited until free RAM
exceeded the approved 6 GiB threshold.

## Verification on main

- The canonical command passed lock/lint/format and **87 tests**, zero failures/skips.
  Pytest took **118.26 seconds** in this single native Windows sample. The same two
  upstream deprecations remain visible.
- Corrected retained-evidence A1-A8 all passed. Package imports came from the primary
  checkout; its executable source hashes matched the reviewed verifier. Verification
  changed none of the 155 retained evidence entries.
- The copied MLflow database had four local artifact URI fields relocated to the
  primary checkout. The original database was unchanged until its archived worktree
  copy was removed. A fresh MLflow artifact download from the relocated store retained
  the complete model ID `ec9219b3b98946ddb328ab28e63509e0f2decbe37b5a6db1048035ffa9317dd7`.
- There was no actual fit, final-test inference, model reselection, relabeling or API
  demonstration rerun. Software fixture tests remain separate from study measurements.

Safe evidence is in [local-integration.json](evidence/local-integration.json).
The measured executable remains `4dc18f9`, with baseline macro-F1 0.896042 versus encoder
0.466797 on the same 239 test rows. Original receipts, predictions, models and report
remain preserved. The two [deferred review minors](review-closeout.md) remain unchanged.

## Private preservation and cleanup

All 3,920 private source files, totaling 302,180,792 bytes, were copied into the primary
`.local/` and SHA-256 checked. The original files were checked again before cleanup;
all matched, with zero additional unpreserved files. The initial evidence and review
archives remain under `.local/archives/p24-20261031-local-study/`. Integration logs,
copy/relocation receipts and the current tracking database are archived separately.

Both old environment directories were moved intact into `.local/preserved-environments/`
as private historical snapshots. They are not the active environment. The primary
`.venv-31216` is active. The merged worktree and `codex/local-study` branch were removed
after preservation and verification. All consumed study claims remain retained in the
primary `.local/study/`; relocation does not reset an attempt allowance.
