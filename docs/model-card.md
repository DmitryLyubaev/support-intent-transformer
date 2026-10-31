# Model card: support-intent BERT-Tiny

Status: one fixed local CPU encoder fine-tune, 31 October 2026. Model ID `ec9219b3b98946ddb328ab28e63509e0f2decbe37b5a6db1048035ffa9317dd7`.
Base: google/bert_uncased_L-2_H-128_A-2 at30b0a37ccaaa32f332884b96992754e246e48c5f,
declared Apache2.0. Data: six BANKING77 intents from PolyAI at
1846984189f2bc0ac51c6fc3a600fa9e1714e0c6, CC BY4.0. Full attribution in ATTRIBUTION.md.
Changes: full encoder plus new classifier adaptation, fixed grouping/privacy exclusions,
development caps and validation split. Original test boundary retained.

Purpose: bounded portfolio study and loopback demonstration. English uncased WordPiece,
NFC/strip, max64 tokens including specials with truncation; fixed six-label mapping.
Training: seed42, three epochs, batch16, AdamW5e-5, decay0.01, clip1.0, all parameters
trainable. Final third epoch selected without test tuning. Macro-F1 on 239
held-out rows: encoder 0.466797; baseline 0.896042.
Per-class counts/errors and weak cases are in study-report.md. One seed/small subset does
not establish general superiority. Labels are upstream references, not assistant-created.

Not for production support decisions, financial advice or customer operations. Forced
six-class outputs are unreliable for unrelated inputs; no uncertainty/OOS claim. Dataset
bias/collection details and pretraining overlap are incomplete. No generative LLM/LoRA/
GPU training, human acceptance, drift monitoring or production deployment is claimed.
Actual saved tokenizer/model remain private locally with manifests and MLflow lineage.
Reproduction of training requires the explicit actual-run allowance; readback/metric
recomputation from retained files requires no additional fit or final-test inference.
