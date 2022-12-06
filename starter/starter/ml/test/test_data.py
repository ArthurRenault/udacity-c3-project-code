from numpy import ndarray
from ..data import process_data


def test_data_values(data):
    assert data['sex'].isin(['Male', 'Female']).all()
    assert data['age'].between(18, 100).all()


def test_process_data_training(data):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, _, _ = process_data(data, categorical_features=cat_features, label="salary", training=True)

    assert isinstance(X, ndarray)
    assert isinstance(y, ndarray)
    assert X.shape == (100, 68)
    assert y.shape == (100,)


def test_process_data_no_training(data):
    X, y, _, _ = process_data(data, training=False)
    assert isinstance(y, ndarray)
    assert y.shape == (0,)
