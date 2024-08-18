import dill as pickle
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()
with open('sber_avto.pkl', 'rb') as model:
    best_pipe = pickle.load(model)


class Form(BaseModel):
    session_id: object
    hit_date: object
    hit_time: float
    hit_number: float
    hit_type: object
    hit_referer: object
    hit_page_path: object
    event_category: object
    event_action: object
    event_label: object
    client_id: object
    visit_date: object
    visit_time: object
    visit_number: float
    utm_source: object
    utm_medium: object
    utm_campaign: object
    utm_adcontent: object
    utm_keyword: object
    device_category: object
    device_os: object
    device_brand: object
    device_model: object
    device_screen_resolution: object
    device_browser: object
    geo_country: object
    geo_city: object


class Prediction(BaseModel):
    Session_id: str
    Result: float


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return best_pipe['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):

    df = pd.DataFrame.from_dict([form.dict()])
    y = best_pipe['model'].predict(df)

    return {
        "Session_id": form.session_id,
        "Result": y[0]
    }