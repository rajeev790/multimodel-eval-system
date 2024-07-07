from transformers import pipeline

class ModelLoader:
    def _init_(self):
        self.models = {
            "bert-base": pipeline("text-classification", model="bert-base-uncased"),
            "bert-large": pipeline("text-classification", model="bert-large-uncased"),
            "roberta-base": pipeline("text-classification", model="roberta-base"),
            "distilbert-base": pipeline("text-classification", model="distilbert-base-uncased"),
            "albert-base": pipeline("text-classification", model="albert-base-v2"),
            "xlnet-base": pipeline("text-classification", model="xlnet-base-cased"),
            "t5-small": pipeline("summarization", model="t5-small"),
            "t5-base": pipeline("summarization", model="t5-base")
        }

    def get_model(self, model_name):
        return self.models.get(model_name)