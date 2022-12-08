import json
from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


def test_api_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == 'Hello there'


def test_malformed_url():
    r = client.get('http://127.0.0.1:8000/wagee')
    assert r.status_code == 404


def test_post_negative_class():
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

    r = client.post('http://127.0.0.1:8000/wage', data=json.dumps(data))
    assert r.status_code == 200
    assert r.json()["sample"] == data
    assert not r.json()["prediction"]


def test_post_positive_class():
    data = {
            "age": 52,
            "workclass": "Self-emp-not-inc",
            "fnlgt": 209642,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 45,
            "native-country": "United-States"
            }

    r = client.post("http://127.0.0.1:8000/wage", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json()["sample"] == data
    assert r.json()["prediction"]


def test_post_invalid_sample():
    r = client.post("http://127.0.0.1:8000/wage", data=json.dumps({"age": 15}))
    assert r.status_code == 422
