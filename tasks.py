def text_classification(models, text):
    results = {}
    for model_name, model in models.items():
        if model.model.config.architectures[0] in ["BertForSequenceClassification", "RobertaForSequenceClassification", "DistilBertForSequenceClassification", "AlbertForSequenceClassification", "XLNetForSequenceClassification"]:
            results[model_name] = model(text)
    return results

def named_entity_recognition(models, text):
    results = {}
    for model_name, model in models.items():
        if model.model.config.architectures[0] == "BertForTokenClassification":
            results[model_name] = model(text)
    return results

def question_answering(models, question, context):
    results = {}
    for model_name, model in models.items():
        if model.model.config.architectures[0] in ["BertForQuestionAnswering", "RobertaForQuestionAnswering", "DistilBertForQuestionAnswering", "AlbertForQuestionAnswering", "XLNetForQuestionAnswering"]:
            results[model_name] = model(question=question, context=context)
    return results

def text_summarization(models, text):
    results = {}
    for model_name, model in models.items():
        if model.model.config.architectures[0] == "T5ForConditionalGeneration":
            results[model_name] = model(text)
    return results