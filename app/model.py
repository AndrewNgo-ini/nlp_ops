from transformers import pipeline

class SentimentAnalysis():
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    def __init__(self) -> None:
        self.classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")

    def __call__(self, text):
        return self.classifier(text)