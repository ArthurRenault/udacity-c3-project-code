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


data = [39,
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
r = requests.post("http://127.0.0.1:8000/wage", data=json.dumps({"sample": data}))
print(r.json())
