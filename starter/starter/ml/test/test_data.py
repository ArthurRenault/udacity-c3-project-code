from numpy import ndarray
from starter.starter.ml.data import process_data

from starter.conftest import TEST_SIZE
from starter.starter.constants import FEATURES_CATEGORICAL


def test_data_values(data):
    assert data['sex'].isin(['Male', 'Female']).all()
    assert data['age'].between(18, 100).all()


def test_process_data_training(data):

    X, y, _, _ = process_data(data, categorical_features=FEATURES_CATEGORICAL, label="salary", training=True)

    assert isinstance(X, ndarray)
    assert isinstance(y, ndarray)
    assert X.shape == (TEST_SIZE, 68)
    assert y.shape == (TEST_SIZE,)


def test_process_data_no_training(data):
    X, y, _, _ = process_data(data, training=False)
    assert isinstance(y, ndarray)
    assert y.shape == (0,)
