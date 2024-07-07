from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from models import ModelLoader
from tasks import text_classification, named_entity_recognition, question_answering, text_summarization
from cache import cached_predict
from benchmark import benchmark_models
from auth import get_current_user
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis
import logging

app = FastAPI()
model_loader = ModelLoader()
models = model_loader.models

logging.basicConfig(level=logging.INFO)

class TextRequest(BaseModel):
    text: str
    task: str

class QARequest(BaseModel):
    question: str
    context: str

@app.on_event("startup")
async def startup():
    redis_pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
    redis_conn = redis.StrictRedis(connection_pool=redis_pool)
    FastAPILimiter.init(redis_conn)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Service is healthy"}

@app.post("/predict", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
def predict(request: TextRequest):
    task = request.task
    text = request.text
    results = {}

    if task == "text-classification":
        results = text_classification(models, text)
    elif task == "ner":
        results = named_entity_recognition(models, text)
    elif task == "question-answering":
        raise HTTPException(status_code=400, detail="Use /predict_qa for question answering")
    elif task == "summarization":
        results = text_summarization(models, text)
    else:
        raise HTTPException(status_code=400, detail="Invalid task type")

    return results

@app.post("/predict_qa", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
def predict_qa(request: QARequest):
    question = request.question
    context = request.context
    results = question_answering(models, question, context)
    return results

@app.post("/benchmark")
def benchmark(dataset: UploadFile = File(...)):
    metrics = benchmark_models(models, dataset.file)
    return metrics

@app.post("/upload_model")
def upload_model(model_file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    # Handle user model upload
    pass