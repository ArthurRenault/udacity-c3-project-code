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
    data = [
        39,
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

    r = client.post('http://127.0.0.1:8000/wage', data=json.dumps({"sample": data}))
    assert r.status_code == 200
    assert r.json()["sample"] == data
    assert not r.json()["prediction"]


def test_post_positive_class():
    data = [
        52,
        'Self-emp-not-inc',
        209642,
        'HS-grad',
        9,
        'Married-civ-spouse',
        'Exec-managerial',
        'Husband',
        'White',
        'Male',
        0,
        0,
        45,
        'United-States'
        ]

    r = client.post("http://127.0.0.1:8000/wage", data=json.dumps({"sample": data}))
    assert r.status_code == 200
    assert r.json()["sample"] == data
    assert r.json()["prediction"]


def test_post_invalid_sample():
    r = client.post("http://127.0.0.1:8000/wage", data=json.dumps({"sample": ['foo']*15}))
    assert r.status_code == 422
