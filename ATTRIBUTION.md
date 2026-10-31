# Input attribution

The BANKING77 subset is by PolyAI and the authors of Efficient Intent Detection
with Dual Sentence Encoders (Casanueva et al., 2020), under CC BY 4.0.
[Original source](https://github.com/PolyAI-LDN/task-specific-datasets).
[CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/).
Changes: fixed six-intent subset, privacy exclusions, duplicate/group-aware split and caps.
The study retained 610 training, 155 validation and 239 test rows. Actual pinned
revision/retrieval/hash receipts are in docs/evidence/source-receipts.json. Raw rows
remain private locally; shareable aggregates contain no dataset message text.

Google's BERT-Tiny checkpoint is declared Apache 2.0 by its
[model card](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2).
Model attribution: Well-Read Students Learn Better (Turc et al., 2019).
The study adapted all encoder parameters and a new six-class head.
No creator endorsement is implied. No dataset rows or model weights are included in Git.
