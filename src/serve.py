from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import load_model, predict

class Item(BaseModel):
    data: list

app = FastAPI()

model = load_model()

@app.post("/predict/")
def get_prediction(item: Item):
    prediction = predict(model, item.data)
    return {"prediction": prediction}