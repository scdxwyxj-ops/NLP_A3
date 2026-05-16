import argparse
from pathlib import Path

from a3_factcheck.rerank.dataset import read_jsonl


class PairDataset:
    def __init__(self, rows, tokenizer, max_length, label_dtype="int"):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_dtype = label_dtype

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        encoded = self.tokenizer(
            row["claim_text"],
            row["evidence_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        if self.label_dtype == "float":
            encoded["labels"] = float(row["label"])
        else:
            encoded["labels"] = int(row["label"])
        return encoded


def build_bce_trainer_class():
    from transformers import Trainer

    class BCETrainer(Trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            import torch

            labels = inputs.pop("labels").float()
            outputs = model(**inputs)
            logits = outputs.logits.view(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels.view(-1)
            )
            return (loss, outputs) if return_outputs else loss

    return BCETrainer


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a HuggingFace cross-encoder evidence reranker."
    )
    parser.add_argument(
        "--model-name",
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
        help="Any HF sequence-classification checkpoint runnable on Colab.",
    )
    parser.add_argument(
        "--train-pairs",
        default="outputs/round05/train-reranker-pairs-top50-neg5.jsonl",
    )
    parser.add_argument("--output-dir", default="models/round05/minilm-reranker")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-labels",
        type=int,
        default=1,
        choices=[1, 2],
        help="Use 1 to preserve reranker checkpoints with a single-logit head.",
    )
    args = parser.parse_args()

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "This experiment needs transformers and torch. In Colab, run: "
            "pip install transformers torch"
        ) from exc

    set_seed(args.seed)
    rows = read_jsonl(args.train_pairs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
        ignore_mismatched_sizes=True,
    )
    dataset = PairDataset(
        rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        label_dtype="float" if args.num_labels == 1 else "int",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=25,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer_class = build_bce_trainer_class() if args.num_labels == 1 else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved reranker to {output_dir}")


if __name__ == "__main__":
    main()
