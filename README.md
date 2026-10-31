# Support intent transformer

Bounded six-intent CPU pretrained-encoder adaptation study. Implementation is in progress.
The approved design and plan are in docs/superpowers/. No model-quality results yet.

Current gate: source/split fixture checks pass, but the existing native Python 3.12.11
runtime aborts when importing Datasets with `no OPENSSL_Applink`. Task 1 is incomplete;
Tasks 2-6 and actual data/model acquisition are unstarted. The project-local runtime
recovery in docs/runtime-recovery-proposal.md is approved. Its setup is waiting on the
approved >=6 GiB free-RAM condition; the latest sample was 4.68 GiB. No additional
recovery approval is needed. Original environment and failed evidence remain retained.
