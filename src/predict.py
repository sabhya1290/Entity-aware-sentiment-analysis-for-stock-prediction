"""
predict.py — Run inference on a single text + entity input.

Usage:
    python src/predict.py --text "Mahindra rises but Tata falls" --target "Mahindra"
    python src/predict.py --text "Infosys beats Q3 estimates" --target "Infosys" --config config/config.yaml
    python src/predict.py --csv_path data/custom_inputs.csv --out_path outputs/predictions.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_device, int_to_label, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict sentiment for a headline + entity")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--model_dir", default=None, help="Override model dir from config")
    # Single prediction
    p.add_argument("--text", default=None, help="Raw headline text")
    p.add_argument("--target", default=None, help="Target entity name")
    # Batch prediction from CSV
    p.add_argument("--csv_path", default=None, help="CSV with 'text' and 'entity' columns for batch prediction")
    p.add_argument("--out_path", default="outputs/predictions.csv", help="Where to save batch predictions")
    return p.parse_args()


def build_entity_text(headline: str, entity: str, template: str) -> str:
    return template.format(entity=entity.strip(), headline=headline.strip())


def predict_single(
    text: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int,
    id2label: Dict[int, str],
) -> Dict:
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().numpy()[0]

    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))
    pred_label = id2label[pred_idx]

    return {
        "predicted_label": pred_label,
        "predicted_index": pred_idx,
        "probabilities": {id2label[i]: float(probs[i]) for i in range(len(probs))},
    }


def predict_batch(
    records: List[Dict],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int,
    id2label: Dict[int, str],
    batch_size: int = 32,
) -> List[Dict]:
    results = []
    texts = [r["text"] for r in records]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoding = tokenizer(
            batch_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()

        probs_batch = softmax(logits, axis=1)
        preds = np.argmax(probs_batch, axis=1)

        for j, (pred_idx, probs) in enumerate(zip(preds, probs_batch)):
            results.append(
                {
                    **records[i + j],
                    "predicted_label": id2label[int(pred_idx)],
                    "predicted_index": int(pred_idx),
                    **{f"prob_{id2label[k]}": float(probs[k]) for k in range(len(probs))},
                }
            )
    return results


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_dir = args.model_dir or cfg["paths"]["model_save_dir"]
    max_length = cfg["model"]["max_length"]
    template = cfg["entity_format"]["template"]
    id2label = int_to_label(cfg)
    device = get_device()

    if not Path(model_dir).exists():
        print(f"[ERROR] Model directory not found: {model_dir}. Run train_finbert.py first.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    # ── Single prediction mode ──────────────────────────────────────────
    if args.text and args.target:
        entity_text = build_entity_text(args.text, args.target, template)
        print(f"\nInput text   : {args.text}")
        print(f"Target entity: {args.target}")
        print(f"Model input  : {entity_text}\n")
        result = predict_single(entity_text, model, tokenizer, device, max_length, id2label)
        print(f"Predicted sentiment : {result['predicted_label'].upper()}")
        print("Class probabilities :")
        for label, prob in sorted(result["probabilities"].items()):
            print(f"  {label:<12} {prob:.4f}  {'█' * int(prob * 30)}")
        return

    # ── Batch CSV mode ──────────────────────────────────────────────────
    if args.csv_path:
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            print(f"[ERROR] CSV not found: {csv_path}")
            sys.exit(1)
        df = pd.read_csv(csv_path)
        required = ["text", "entity"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            # If raw headline column exists, build entity text
            alt_required = ["headline", "entity"]
            alt_missing = [c for c in alt_required if c not in df.columns]
            if alt_missing:
                print(f"[ERROR] CSV must have columns {required} or {alt_required}. Found: {df.columns.tolist()}")
                sys.exit(1)
            df["text"] = df.apply(
                lambda r: build_entity_text(str(r["headline"]), str(r["entity"]), template), axis=1
            )

        records = df.to_dict(orient="records")
        # Ensure text column has entity-aware format
        records = [
            {**r, "text": build_entity_text(str(r.get("headline", r["text"])), str(r["entity"]), template)}
            if "headline" in r else r
            for r in records
        ]

        results = predict_batch(records, model, tokenizer, device, max_length, id2label)
        out_df = pd.DataFrame(results)
        Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out_path, index=False)
        print(f"Batch predictions saved → {args.out_path}  ({len(out_df)} rows)")
        return

    print("[ERROR] Provide either --text and --target  OR  --csv_path")
    sys.exit(1)


if __name__ == "__main__":
    main()
