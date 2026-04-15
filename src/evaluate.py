"""
evaluate.py — Full evaluation on test set.

Loads the best saved model, runs inference on test.csv, and generates:
  - outputs/test_metrics.json       (accuracy, precision, recall, f1s)
  - outputs/classification_report.txt
  - outputs/confusion_matrix.png
  - outputs/roc_auc.json            (one-vs-rest AUC if probabilities available)

Usage:
    python src/evaluate.py --config config/config.yaml
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import SentFinDataset
from utils import (
    ensure_dirs,
    get_device,
    int_to_label,
    load_config,
    save_json,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate FinBERT on test set")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--model_dir", default=None, help="Override model directory from config")
    return p.parse_args()


def run_inference(
    model: AutoModelForSequenceClassification,
    dataset: SentFinDataset,
    batch_size: int,
    device: torch.device,
) -> tuple:
    """Return (all_logits, all_labels) as numpy arrays."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.vstack(all_logits), np.concatenate(all_labels)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run(cfg: dict, model_dir_override: str | None = None) -> None:
    paths = cfg["paths"]
    t_cfg = cfg["training"]
    m_cfg = cfg["model"]

    output_dir = Path(paths["output_dir"])
    logs_dir = paths["logs_dir"]
    ensure_dirs(str(output_dir), logs_dir)
    logger = setup_logging(logs_dir, "evaluate")

    model_dir = model_dir_override or paths["model_save_dir"]
    processed_dir = Path(paths["processed_dir"])
    device = get_device()

    logger.info(f"Loading tokenizer + model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    logger.info("Loading test set …")
    test_ds = SentFinDataset(
        str(processed_dir / "test.csv"),
        tokenizer,
        m_cfg["max_length"],
    )
    logger.info(f"  Test samples: {len(test_ds)}")

    logger.info("Running inference …")
    logits, true_labels = run_inference(model, test_ds, t_cfg["eval_batch_size"], device)
    probs = softmax(logits, axis=1)
    pred_labels = np.argmax(logits, axis=1)

    # ── Metrics ────────────────────────────────────────────────────────────
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    precision_macro = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
    recall_macro = recall_score(true_labels, pred_labels, average="macro", zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
    }

    # ROC-AUC (one-vs-rest)
    try:
        auc = roc_auc_score(true_labels, probs, multi_class="ovr", average="macro")
        metrics["roc_auc_macro_ovr"] = float(auc)
    except Exception as exc:
        logger.warning(f"ROC-AUC computation failed: {exc}")

    save_json(metrics, str(output_dir / "test_metrics.json"))
    logger.info(f"Test metrics: {metrics}")

    # ── Classification Report ──────────────────────────────────────────────
    id2label = int_to_label(cfg)
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    report_str = classification_report(
        true_labels,
        pred_labels,
        target_names=target_names,
        zero_division=0,
    )
    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report_str)
    logger.info(f"\n{report_str}")
    logger.info(f"Classification report saved → {report_path}")

    # ── Confusion Matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(true_labels, pred_labels)
    cm_path = str(output_dir / "confusion_matrix.png")
    plot_confusion_matrix(cm, target_names, cm_path)
    logger.info(f"Confusion matrix saved → {cm_path}")

    logger.info("Evaluation complete.")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run(cfg, args.model_dir)


if __name__ == "__main__":
    main()
