from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
import sys
import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


#making object for  fastapi
app_desc = """<h2>Try this app by uploading  image of monkey  with `/predict`</h2>
<h2>This app will help you  to predict the breed of the monkey!!</h2>
<br>By Rehan Shaikh"""

app = FastAPI(title='Hands On FastAPI ', description=app_desc)


#loading the model
MODEL_PATH ='./model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


def model_predict(image: Image.Image, model):
    resp={}
    # Preprocessing the image
    x = np.asarray(image.resize((224, 224)))

    ## Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x)
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "mantled_howler"
    elif preds == 1:
        preds = "patas_monkey"
    elif preds == 2:
        preds = "bald_uakari"
    elif preds == 3:
        preds = "japanese_macaque"
    elif preds == 4:
        preds = "pygmy_marmoset"
    elif preds == 5:
        preds = "white_headed_capuchin"
    elif preds == 6:
        preds = "silvery_marmoset"
    elif preds == 7:
        preds = "common_squirrel_monkey"
    elif preds == 8:
        preds = "black_headed_night_monkey"
    elif preds == 9:
        preds = "nilgiri_langur"
    #
    # resp["image"] = image
    # resp["Prdiction"] = preds
    return preds


@app.get('/')
def index():
    return {'message': 'Hello, World'}


@app.post('/predict')
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    preds = model_predict(image, model)
    return preds

if __name__ == '__main__':
    uvicorn.run(app,debug=True)