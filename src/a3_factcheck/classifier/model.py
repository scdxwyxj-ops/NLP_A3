from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

class ClaimClassifier:
    def __init__(self):
        self.labels = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
        self.label2id = {l: i for i, l in enumerate(self.labels)}
        self.id2label = {i: l for i, l in enumerate(self.labels)}

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=4,
            id2label=self.id2label,
            label2id=self.label2id
        )

    # prepare dataset
    def preprocess(self, claim, evidence, label):
        text = claim + " [SEP] " + evidence
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256
        )
        inputs["labels"] = self.label2id[label]
        return inputs

    # build training dataset
    def build_dataset(self, data, evidence):
        dataset = []

        for claim in data.values():
            claim_text = claim["claim_text"]
            label = claim["claim_label"]

            # use first gold evidence (simple baseline)
            if len(claim["evidences"]) == 0:
                continue

            evidence_id = claim["evidences"][0]
            if evidence_id not in evidence:
                continue

            evidence_text = evidence[evidence_id]

            item = self.preprocess(claim_text, evidence_text, label)
            dataset.append(item)

        return dataset

    # TRAIN MODEL
    def train(self, train_data, evidence):
        train_dataset = self.build_dataset(train_data, evidence)

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,              # keep small (Colab friendly)
            per_device_train_batch_size=8,
            logging_steps=50,
            save_strategy="no",
            report_to=[],
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()

    # PREDICT
    def predict(self, claim, evidence_text):
        self.model.eval()

        text = claim + " [SEP] " + evidence_text

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        pred = torch.argmax(outputs.logits).item()

        return self.id2label[pred]