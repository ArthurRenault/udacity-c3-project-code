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
    sample: list = []

    class Config:
        schema_extra = {
                "example": {
                        "sample": [39,
        'State-gov',
        77516,
        'Bachelors',
        13,
        'Never-married',
        'Adm-clerical',
        'Not-in-family',
        'White',
        'Male',
        2174,
        0,
        40,
        'United-States'
        ]
                        }
                }


@app.get('/')
def hello_world():
    return "Hello there"


@app.post('/wage')
def get_prediction(model_input: ModelInput):

    input_size = len(model_input.sample)
    if input_size != 14:
        raise HTTPException(status_code=422, detail=f"14 features expected, received {input_size} features.")

    encoder, trained_model = load_training_pipeline(PATH_ENCODER, PATH_MODEL)

    data = pd.DataFrame(data=[model_input.sample], columns=FEATURES)

    x_predict, _, _, _ = process_data(
            data,
            categorical_features=FEATURES_CATEGORICAL,
            label=None,
            training=False,
            encoder=encoder,
            lb=None
            )
    prediction = inference(trained_model, x_predict)

    return {"sample": model_input.sample, "prediction": bool(prediction[0])}
