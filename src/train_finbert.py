"""
train_finbert.py — Fine-tune ProsusAI/finbert for entity-aware financial
sentiment classification using HuggingFace Trainer API.

Usage:
    python src/train_finbert.py --config config/config.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import SentFinDataset
from utils import (
    ensure_dirs,
    get_device,
    int_to_label,
    load_config,
    save_json,
    set_seed,
    setup_logging,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune FinBERT on SEntFiN")
    p.add_argument("--config", default="config/config.yaml")
    return p.parse_args()


def compute_metrics(eval_pred):
    """Compute metrics inside HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
    }


def run(cfg: dict) -> None:
    t_cfg = cfg["training"]
    m_cfg = cfg["model"]
    paths = cfg["paths"]

    seed = t_cfg["seed"]
    set_seed(seed)

    ensure_dirs(
        paths["output_dir"],
        paths["model_save_dir"],
        paths["logs_dir"],
        "outputs/checkpoints",
    )

    logger = setup_logging(paths["logs_dir"], "train_finbert")
    device = get_device()
    logger.info(f"Device: {device}")

    processed_dir = Path(paths["processed_dir"])
    model_name = m_cfg["name"]
    max_length = m_cfg["max_length"]
    num_labels = m_cfg["num_labels"]

    # ── Tokenizer ──────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ── Datasets ───────────────────────────────────────────────────────────
    logger.info("Loading datasets …")
    train_ds = SentFinDataset(str(processed_dir / "train.csv"), tokenizer, max_length)
    val_ds = SentFinDataset(str(processed_dir / "val.csv"), tokenizer, max_length)
    logger.info(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ── Model ──────────────────────────────────────────────────────────────
    logger.info(f"Loading model: {model_name}  num_labels={num_labels}")
    id2label = int_to_label(cfg)
    label2id = cfg["label_map"]
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # ── TrainingArguments ──────────────────────────────────────────────────
    checkpoint_dir = "outputs/checkpoints"
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=t_cfg["epochs"],
        per_device_train_batch_size=t_cfg["batch_size"],
        per_device_eval_batch_size=t_cfg["eval_batch_size"],
        learning_rate=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
        warmup_ratio=t_cfg["warmup_ratio"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        fp16=t_cfg["fp16"] and device.type == "cuda",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=t_cfg["save_total_limit"],
        seed=seed,
        report_to="none",
        logging_dir=paths["logs_dir"],
        logging_steps=50,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=t_cfg["early_stopping_patience"])]

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    logger.info("Starting training …")
    train_result = trainer.train()

    # ── Save best model ────────────────────────────────────────────────────
    model_save_dir = paths["model_save_dir"]
    trainer.save_model(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    logger.info(f"Best model saved → {model_save_dir}")

    # ── Save training metrics ──────────────────────────────────────────────
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)

    val_metrics = trainer.evaluate()
    trainer.log_metrics("eval", val_metrics)

    all_metrics = {**train_metrics, **val_metrics}
    metrics_path = Path(paths["output_dir"]) / "train_metrics.json"
    save_json(all_metrics, str(metrics_path))
    logger.info(f"Training metrics saved → {metrics_path}")
    logger.info("Training complete.")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
