"""
Creates an API to serve our model predictions to users.
"""
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


from starter.constants import FEATURES, FEATURES_CATEGORICAL, PATH_ENCODER, PATH_MODEL
from starter.ml.data import process_data
from starter.ml.model import load_training_pipeline, inference

app = FastAPI()


class ModelInput(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
                "example": {
                        "age": 39,
                        "workclass": "State-gov",
                        "fnlgt": 77516,
                        "education": "Bachelors",
                        "education_num": 13,
                        "marital_status": "Never-married",
                        "occupation": "Adm-clerical",
                        "relationship": "Not-in-family",
                        "race": "White",
                        "sex": "Male",
                        "capital_gain": 2174,
                        "capital_loss": 0,
                        "hours_per_week": 40,
                        "native_country": "United-States"
                        }
                }
        alias_generator = lambda x: x.replace('_', '-')


@app.get('/')
def hello_world():
    return "Hello there"


@app.post('/wage')
def get_prediction(model_input: ModelInput):

    encoder, trained_model = load_training_pipeline(PATH_ENCODER, PATH_MODEL)

    data = pd.DataFrame(model_input.dict(by_alias=True), index=[0])

    x_predict, _, _, _ = process_data(
            data,
            categorical_features=FEATURES_CATEGORICAL,
            label=None,
            training=False,
            encoder=encoder,
            lb=None
            )
    prediction = inference(trained_model, x_predict)

    return {"prediction": bool(prediction[0]), "sample": model_input.dict(by_alias=True)}
