import os
dirname = os.path.dirname(__file__)

COLUMN_LABEL = 'salary'

PATH_ENCODER = os.path.join(dirname, '../model/encoder.pkl')
PATH_MODEL = os.path.join(dirname, '../model/model.pkl')
PATH_MODEL_METRICS = os.path.join(dirname, '../model/slice_output.txt')
PATH_SOURCE_DATA = os.path.join(dirname, '../data/census.csv')

FEATURES = [
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country"
        ]
FEATURES_CATEGORICAL = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
FEATURES_SLICING = [
    'education',
    'sex',
    'occupation'
    ]
N_ESTIMATORS = 100
RANDOM_SEED = 1234
TEST_SIZE = 0.2
