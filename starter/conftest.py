import pytest
from pandas import read_csv

from starter.starter.constants import PATH_SOURCE_DATA


TEST_SIZE = 100


@pytest.fixture(scope='session')
def data():
    return read_csv(PATH_SOURCE_DATA, skipinitialspace=True)[:TEST_SIZE]
