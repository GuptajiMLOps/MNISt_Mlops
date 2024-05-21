from fastapi import FastAPI
from pydantic import BaseModel
from predict import load_model, prediction

class Item(BaseModel):
    data: list

app = FastAPI()

model = load_model()

@app.post("/predict/")
def get_prediction(item: Item):
    predictions = prediction(model, item.data)
    return {"prediction": predictions}