from functools import lru_cache
from models import ModelLoader

model_loader = ModelLoader()

@lru_cache(maxsize=128)
def cached_predict(model_name, text):
    model = model_loader.get_model(model_name)
    return model(text)