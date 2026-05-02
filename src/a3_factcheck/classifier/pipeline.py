from a3_factcheck.classifier.model import ClaimClassifier

classifier = ClaimClassifier()

def classify_claim(claim, evidence_text):
    return classifier.predict(claim, evidence_text)