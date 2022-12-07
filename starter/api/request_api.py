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
