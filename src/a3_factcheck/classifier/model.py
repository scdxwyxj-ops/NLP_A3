from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class ClaimClassifier:
    def __init__(self):
        self.labels = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=4
        )

    def predict(self, claim, evidence_text):
        text = claim + " [SEP] " + evidence_text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)

        outputs = self.model(**inputs)
        pred = torch.argmax(outputs.logits).item()

        return self.labels[pred] 