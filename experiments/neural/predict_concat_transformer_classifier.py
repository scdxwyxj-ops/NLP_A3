import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from a3_factcheck.evaluation import run_eval
from a3_factcheck.rerank.api import write_predictions
from experiments.neural.train_concat_transformer_classifier import (
    LABELS,
    build_prediction_json,
    read_jsonl,
    row_to_text,
)


def collate_texts(rows, tokenizer, max_length, evidence_top_k, evidence_token_budget):
    texts = [
        row_to_text(row, evidence_top_k, evidence_token_budget)
        for row in rows
    ]
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def main():
    parser = argparse.ArgumentParser(description="Predict labels with a saved concat classifier.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-groundtruth", default="")
    parser.add_argument("--eval-script", default="eval.py")
    parser.add_argument("--evidence-top-k", type=int, default=10)
    parser.add_argument("--evidence-token-budget", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    loader = DataLoader(
        rows,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )
    predicted_ids = []
    with torch.no_grad():
        for batch_rows in loader:
            batch = collate_texts(
                batch_rows,
                tokenizer,
                args.max_length,
                args.evidence_top_k,
                args.evidence_token_budget,
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predicted_ids.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())

    prediction_json = build_prediction_json(rows, predicted_ids)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(prediction_json, output_path)
    print(json.dumps({"output": str(output_path), "rows": len(rows), "labels": LABELS}, indent=2))

    if args.eval_groundtruth:
        print(run_eval(args.eval_script, output_path, args.eval_groundtruth))


if __name__ == "__main__":
    main()
