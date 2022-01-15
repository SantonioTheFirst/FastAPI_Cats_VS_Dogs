from fastapi import FastAPI, Request
import numpy as np
from tensorflow.keras.models import load_model
import json


model_path = 'model/cats_vs_dogs.h5'
model = load_model(model_path)


app = FastAPI()


@app.post('/')
async def index(request: Request):
    image_list = json.loads(await request.json())
    image_np = np.array(image_list, dtype=np.uint8)
    prediction = model.predict(np.expand_dims(image_np, axis=0))
    prediction_list = prediction[0].tolist()
    return {'prediction': prediction_list}
