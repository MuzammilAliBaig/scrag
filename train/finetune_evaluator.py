"""Fine-tune DeBERTa-v3-small as the Module B retrieval evaluator.

    python -m train.finetune_evaluator --epochs 3
    python -m train.finetune_evaluator --epochs 3 --device cuda   # Kaggle / Colab

This is the only GPU job in SCRAG, and the only place the contribution is
actually trained. It runs unmodified on CPU (slower) or on a free Kaggle/Colab
T4 - see KAGGLE.md for the notebook cells.

Three choices worth defending:

* **Cross-encoder, not bi-encoder.** The question and chunk are concatenated
  into one sequence so the model attends across them. CRAG does the same:
  "the question is concatenated with each single document as the input, and the
  evaluator predicts the relevance score for each question-document pair
  individually" (Yan et al. 2024, section 4.2).
* **Class-weighted loss.** The weak-labeled set is roughly 74% `wrong`, 19%
  `correct`, 7% `ambiguous`. Unweighted, the model can score ~74% by never
  predicting `ambiguous` at all - which is precisely the class the whole design
  hinges on.
* **Model selection on macro-F1, not accuracy.** For the same reason. Accuracy
  rewards the majority class; macro-F1 forces `ambiguous` to count.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

import config
from core.types import ChunkLabel

LABELS = [ChunkLabel.CORRECT.value, ChunkLabel.AMBIGUOUS.value, ChunkLabel.WRONG.value]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path} - run `python -m train.build_dataset` first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(rows: list[dict], tokenizer, max_length: int):
    """Cross-encoder encoding: question as sequence A, chunk as sequence B."""
    import torch

    encoded = tokenizer(
        [r["question"] for r in rows],
        [r["chunk_text"] for r in rows],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    labels = torch.tensor([LABEL_TO_ID[r["label"]] for r in rows], dtype=torch.long)
    return encoded["input_ids"], encoded["attention_mask"], labels


def class_weights(rows: list[dict]):
    """Inverse-frequency weights, normalised to mean 1."""
    import torch

    counts = np.array([sum(1 for r in rows if r["label"] == name) for name in LABELS], dtype=float)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(LABELS) * counts)
    return torch.tensor(weights / weights.mean(), dtype=torch.float)


def macro_f1(true: np.ndarray, pred: np.ndarray) -> tuple[float, dict]:
    per_class = {}
    f1s = []
    for i, name in enumerate(LABELS):
        tp = int(((pred == i) & (true == i)).sum())
        fp = int(((pred == i) & (true != i)).sum())
        fn = int(((pred != i) & (true == i)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int((true == i).sum()),
        }
        f1s.append(f1)
    return float(np.mean(f1s)), per_class


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for input_ids, mask, labels in loader:
            logits = model(
                input_ids=input_ids.to(device), attention_mask=mask.to(device)
            ).logits
            preds.append(logits.argmax(-1).cpu().numpy())
            trues.append(labels.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    parser = argparse.ArgumentParser(description="Fine-tune the Module B evaluator")
    parser.add_argument("--data", default="data/eval_labels")
    parser.add_argument("--out", default="data/models/evaluator")
    parser.add_argument("--model", default=config.EVALUATOR.base_model)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=config.EVALUATOR.max_length)
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    parser.add_argument("--limit", type=int, default=None, help="cap training rows, for smoke runs")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.SEED)

    data_dir = Path(args.data)
    train_rows = load_jsonl(data_dir / "train.jsonl")
    val_rows = load_jsonl(data_dir / "val.jsonl")
    if args.limit:
        train_rows = train_rows[: args.limit]

    print(f"device={device}  model={args.model}  max_length={args.max_length}")
    print(f"train={len(train_rows)}  val={len(val_rows)}")
    dist = {name: sum(1 for r in train_rows if r["label"] == name) for name in LABELS}
    print(f"train label distribution: {dist}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    load_kwargs = dict(
        num_labels=len(LABELS),
        id2label={i: name for i, name in enumerate(LABELS)},
        label2id=LABEL_TO_ID,
        # The published checkpoint is fp16. Loading it as-is gives Half logits
        # against Float class weights ("expected scalar type Half but found
        # Float"), and half-precision training on CPU is wrong regardless.
        dtype=torch.float32,
    )
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, **load_kwargs)
    except TypeError:
        # Older transformers spell it torch_dtype.
        load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        model = AutoModelForSequenceClassification.from_pretrained(args.model, **load_kwargs)
    model = model.to(device)

    train_ds = TensorDataset(*encode(train_rows, tokenizer, args.max_length))
    val_ds = TensorDataset(*encode(val_rows, tokenizer, args.max_length))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    weights = class_weights(train_rows).to(device)
    print(f"class weights ({', '.join(LABELS)}): {[round(w, 3) for w in weights.tolist()]}")

    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (input_ids, mask, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            logits = model(
                input_ids=input_ids.to(device), attention_mask=mask.to(device)
            ).logits
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            if step % 20 == 0 or step == len(train_loader):
                elapsed = time.time() - started
                print(
                    f"  epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss {running / step:.4f}  [{elapsed / 60:.1f} min]",
                    flush=True,
                )

        true, pred = evaluate(model, val_loader, device)
        accuracy = float((true == pred).mean())
        f1, per_class = macro_f1(true, pred)
        print(f"  epoch {epoch}: val accuracy {accuracy:.4f}  macro-F1 {f1:.4f}")
        for name, stats in per_class.items():
            print(f"      {name:>10}  P {stats['precision']:.3f}  R {stats['recall']:.3f}  "
                  f"F1 {stats['f1']:.3f}  n={stats['support']}")
        history.append(
            {"epoch": epoch, "train_loss": running / len(train_loader),
             "val_accuracy": accuracy, "val_macro_f1": f1, "per_class": per_class}
        )

        # Select on macro-F1: accuracy would reward ignoring `ambiguous`.
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"      saved checkpoint (best macro-F1 {f1:.4f})")

    meta = {
        "base_model": args.model,
        "labels": LABELS,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "device": device,
        "seed": config.SEED,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "best_val_macro_f1": best_f1,
        "train_minutes": round((time.time() - started) / 60, 2),
        "history": history,
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nbest val macro-F1 {best_f1:.4f}  ->  {out_dir}")
    print(f"training took {meta['train_minutes']} min on {device}")


if __name__ == "__main__":
    main()
