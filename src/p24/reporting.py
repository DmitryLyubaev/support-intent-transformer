from pathlib import Path

from p24.io import digest_file, read_json, write_json


def write_report(root: Path, project_root: Path, observed_at: str) -> dict:
    metrics = read_json(root / "metrics.json")
    training = read_json(root / "training.json")
    metadata = read_json(root / "study.json")
    preparation = read_json(root / "preparation.json")
    model_id = training["model_id"]
    rows = ["| Intent | Test n | Baseline F1 | Encoder F1 |", "|---|---:|---:|---:|"]
    rows += [
        f"| {a['label']} | {a['support']} | {a['f1']:.4f} | {b['f1']:.4f} |"
        for a, b in zip(metrics["baseline"]["per_class"], metrics["encoder"]["per_class"])
    ]
    error_notes = (
        (root / "error-analysis.md").read_text(encoding="utf-8")
        if (root / "error-analysis.md").exists()
        else "Fixed error sample retained privately; detailed inspection pending."
    )
    report = f"# Six-intent CPU transformer study\n\nMeasured 31 October 2026. Bounded local portfolio demonstration, not production.\nExecutable source: `{metadata['source_commit']}`. Model identity: `{model_id}`.\nOriginal BANKING77 labels; exactly {metrics['count']} frozen retained test rows per arm.\nMacro-F1: baseline **{metrics['baseline']['macro_f1']:.6f}**, encoder **{metrics['encoder']['macro_f1']:.6f}**.\nEncoder-minus-baseline difference: **{metrics['delta_macro_f1']:.6f}**. No score threshold selected this result.\n\n{chr(10).join(rows)}\n\n## Training and provenance\n\nDevelopment/test counts: `{preparation['counts']}`. Split/exclusion manifests, seven source\nreceipts and original predictions are retained privately. The fixed recipe used seed42,\nbatch16, length64, three epochs, AdamW5e-5/weight_decay0.01 and full encoder/head updates.\nAll {training['fit']['total_parameters']:,} parameters were trainable. Both encoder-layer\nquery tensors changed, with original/final hashes and actual deltas verified against\npinned base and selected saved weights. Measured worker elapsed {training['elapsed_seconds']:.3f}s,\nsampled process-tree peak RSS {training['peak_rss_bytes'] / 1024**2:.1f} MiB. These are one\nshared-host CPU sample including process initialization/checkpoint saving, not a benchmark/SLA.\nActual feasibility and measured runs are separate; feasibility candidate weights were discarded.\n\n## Error inspection\n\n{error_notes}\n\n## Limits\n\nSix related English intents, one seed and a small fixed subset. This is not the full\nBANKING77 benchmark, generative LLM fine-tuning or instruction tuning. Label provenance\nand dataset collection/bias limits remain; no human-blinded validation or exhaustive PII\naudit was performed. Assistant preparation review used a fixed training sample; regex-\nflagged/uncleared rows were excluded. The detector checks defined duplicates, not all\nsemantic or pretraining overlap. API classification is closed-set with no OOS detector\nor calibrated confidence. Actual model/tokenizer readback, loopback HTTP restart and\nsoftware tests are separate from held-out accuracy. No cloud/hosted CI/production use,\ncustomer adoption or commercial ML tenure is claimed. Raw rows, weights and MLflow remain\nlocal/ignored. See ATTRIBUTION.md and README.md for licences and reproduction.\n"
    card = f"# Model card: support-intent BERT-Tiny\n\nStatus: one fixed local CPU encoder fine-tune, 31 October 2026. Model ID `{model_id}`.\nBase: google/bert_uncased_L-2_H-128_A-2 at30b0a37ccaaa32f332884b96992754e246e48c5f,\ndeclared Apache2.0. Data: six BANKING77 intents from PolyAI at\n1846984189f2bc0ac51c6fc3a600fa9e1714e0c6, CC BY4.0. Full attribution in ATTRIBUTION.md.\nChanges: full encoder plus new classifier adaptation, fixed grouping/privacy exclusions,\ndevelopment caps and validation split. Original test boundary retained.\n\nPurpose: bounded portfolio study and loopback demonstration. English uncased WordPiece,\nNFC/strip, max64 tokens including specials with truncation; fixed six-label mapping.\nTraining: seed42, three epochs, batch16, AdamW5e-5, decay0.01, clip1.0, all parameters\ntrainable. Final third epoch selected without test tuning. Macro-F1 on {metrics['count']}\nheld-out rows: encoder {metrics['encoder']['macro_f1']:.6f}; baseline {metrics['baseline']['macro_f1']:.6f}.\nPer-class counts/errors and weak cases are in study-report.md. One seed/small subset does\nnot establish general superiority. Labels are upstream references, not assistant-created.\n\nNot for production support decisions, financial advice or customer operations. Forced\nsix-class outputs are unreliable for unrelated inputs; no uncertainty/OOS claim. Dataset\nbias/collection details and pretraining overlap are incomplete. No generative LLM/LoRA/\nGPU training, human acceptance, drift monitoring or production deployment is claimed.\nActual saved tokenizer/model remain private locally with manifests and MLflow lineage.\nReproduction of training requires the explicit actual-run allowance; readback/metric\nrecomputation from retained files requires no additional fit or final-test inference.\n"
    docs = {"docs/study-report.md": report, "docs/model-card.md": card}
    for path, content in docs.items():
        (project_root / path).write_text(content, encoding="utf-8")
    receipt = {
        "time": observed_at,
        "model_id": model_id,
        "metrics_sha256": digest_file(root / "metrics.json"),
        "documents": {path: digest_file(project_root / path) for path in docs},
    }
    write_json(root / "report.json", receipt)
    return receipt
