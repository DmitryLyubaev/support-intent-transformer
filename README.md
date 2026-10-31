# Support intent transformer

Bounded six-intent CPU pretrained-encoder adaptation study. Implementation is in progress.
The approved design and plan are in docs/superpowers/. No model-quality results yet.

Current gate: source/split fixture checks pass, but the existing native Python 3.12.11
runtime aborts when importing Datasets with `no OPENSSL_Applink`. Task 1 is incomplete;
Tasks 2-6 and actual data/model acquisition are unstarted. The project-local runtime
recovery proposal is in docs/runtime-recovery-proposal.md and awaits approval.
