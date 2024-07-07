import csv

def benchmark_models(models, dataset):
    results = {}
    reader = csv.reader(dataset.decode('utf-8').splitlines())
    for row in reader:
        text = row[0]
        labels = row[1]
        for model_name, model in models.items():
            if model.model.config.architectures[0] in ["BertForSequenceClassification", "RobertaForSequenceClassification", "DistilBertForSequenceClassification", "AlbertForSequenceClassification", "XLNetForSequenceClassification"]:
                predictions = model(text)
                # Calculate metrics (accuracy, F1 score, etc.) and store in results
    return results