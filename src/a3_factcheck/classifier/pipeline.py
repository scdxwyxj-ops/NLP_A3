from a3_factcheck.classifier.model import ClaimClassifier

class Pipeline:
    def __init__(self, retrieve_fn, evidence):
        self.retrieve_fn = retrieve_fn
        self.evidence = evidence
        self.classifier = ClaimClassifier()

    def run(self, claim):
        # retrieve top evidence IDs
        top_ids = self.retrieve_fn(claim)

        # get evidence text
        evidence_text = " ".join([self.evidence[eid] for eid in top_ids])

        # predict label
        label = self.classifier.predict(claim, evidence_text)

        return label, top_ids