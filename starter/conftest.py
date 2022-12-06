import pytest
from pandas import read_csv


@pytest.fixture(scope='session')
def data():
    return read_csv('data/census.csv', skipinitialspace=True)[:100]
