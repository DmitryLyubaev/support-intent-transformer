import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizerFast

from p24.config import StudyConfig
from p24.contracts import Prediction, PreparedSplit
from p24.data import normalize
from p24.io import digest_bytes, write_json
from p24.metrics import score


def _tensor_digest(tensor) -> str:
    return digest_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def fit_encoder(
    train: PreparedSplit,
    validation: PreparedSplit | None,
    base_dir: Path,
    output: Path,
    config: StudyConfig,
    purpose: str,
    emit,
) -> dict:
    if purpose not in ("feasibility", "measured") or not train.rows:
        raise ValueError("Invalid training purpose/population")
    if output.exists():
        raise ValueError("Staged output exists; retain prior attempt")
    started = time.monotonic()
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    tokenizer = BertTokenizerFast(vocab_file=str(base_dir / "vocab.txt"), do_lower_case=True)
    model, loading = BertForSequenceClassification.from_pretrained(
        base_dir,
        local_files_only=True,
        use_safetensors=True,
        num_labels=len(config.labels),
        id2label=dict(enumerate(config.labels)),
        label2id={v: i for i, v in enumerate(config.labels)},
        output_loading_info=True,
    )
    if set(loading["missing_keys"]) - {"classifier.weight", "classifier.bias"}:
        raise ValueError("Pretrained encoder has missing parameters")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    names = [f"bert.encoder.layer.{layer}.attention.self.query.weight" for layer in range(2)]
    parameters = dict(model.named_parameters())
    before = {name: parameters[name].detach().clone() for name in names}
    total = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    selected = (
        tuple(sorted(train.rows, key=lambda r: r.row_id))[:128]
        if purpose == "feasibility"
        else train.rows
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        list(selected),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )
    maximum_steps = 50 if purpose == "feasibility" else config.epochs * len(loader)
    epochs = math.ceil(maximum_steps / len(loader))
    steps, complete_epochs, last_loss = 0, 0, None
    history = []
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            inputs = tokenizer(
                [normalize(r.text) for r in batch],
                padding=True,
                truncation=True,
                max_length=config.length,
                return_tensors="pt",
            )
            labels = torch.tensor([config.labels.index(r.label) for r in batch], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs, labels=labels).loss
            last_loss = float(loss.detach())
            if not math.isfinite(last_loss):
                raise ValueError("Training loss is not finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            emit({"event": "step", "step": steps, "loss": last_loss})
            if steps >= maximum_steps:
                break
        complete_epochs = epoch + 1
        if validation is not None and purpose == "measured":
            model.eval()
            predictions, loss_sum = [], 0.0
            with torch.no_grad():
                for offset in range(0, len(validation.rows), config.batch_size):
                    batch = validation.rows[offset : offset + config.batch_size]
                    inputs = tokenizer(
                        [normalize(r.text) for r in batch],
                        padding=True,
                        truncation=True,
                        max_length=config.length,
                        return_tensors="pt",
                    )
                    labels = torch.tensor([config.labels.index(r.label) for r in batch])
                    result = model(**inputs, labels=labels)
                    loss_sum += float(result.loss) * len(batch)
                    predictions.extend(
                        Prediction(r.row_id, config.labels[int(i)])
                        for r, i in zip(batch, result.logits.argmax(dim=-1))
                    )
            item = {
                "epoch": epoch + 1,
                "validation_loss": loss_sum / len(validation.rows),
                "validation_macro_f1": score(validation.rows, predictions, config.labels)[
                    "macro_f1"
                ],
            }
            history.append(item)
            emit({"event": "epoch", **item})
        if steps >= maximum_steps:
            break
    changes = [
        {
            "layer": layer,
            "name": name,
            "before_sha256": _tensor_digest(before[name]),
            "after_sha256": _tensor_digest(parameters[name]),
            "max_abs_delta": float((parameters[name].detach() - before[name]).abs().max()),
        }
        for layer, name in enumerate(names)
    ]
    receipt = {
        "schema_version": 1,
        "purpose": purpose,
        "complete": True,
        "completed_steps": steps,
        "epochs_completed": complete_epochs,
        "last_loss": last_loss,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": total,
        "encoder_changes": changes,
        "validation_history": history,
        "elapsed_seconds": time.monotonic() - started,
        "selected_epoch": complete_epochs,
        "seed": config.seed,
        "deterministic_algorithms": True,
        "cpu_threads": 4,
        "interop_threads": 1,
        "missing_keys": loading["missing_keys"],
        "unexpected_keys": loading["unexpected_keys"],
        "training_truncated": sum(
            len(tokenizer(normalize(r.text))["input_ids"]) > config.length for r in train.rows
        ),
        "validation_truncated": sum(
            len(tokenizer(normalize(r.text))["input_ids"]) > config.length
            for r in (validation.rows if validation is not None else ())
        ),
    }
    validate_adaptation(receipt)
    output.mkdir(parents=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    write_json(output.parent / "fit-receipt.json", receipt)
    return receipt


def validate_adaptation(receipt):
    changes = receipt.get("encoder_changes", [])
    if {c["layer"] for c in changes} != {0, 1} or any(
        not math.isfinite(c["max_abs_delta"]) or c["max_abs_delta"] <= 0 for c in changes
    ):
        raise ValueError("Both pretrained encoder layers must show actual parameter changes")
