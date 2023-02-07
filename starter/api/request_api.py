"""
Simulate and API request from a user.

First, make sure FastAPI runs locally
$ cd starter/
$ uvicorn main:app

Then, execute $ python request_api.py
The script will print the sample used and the prediction in the terminal.
"""
import json
import requests


data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
        }
r = requests.post("https://udacity-deployment.onrender.com/wage", data=json.dumps(data))
print(f'status code: {r.status_code}', f'response: {r.json()}', sep='\n')
