COLUMN_LABEL = 'salary'

PATH_ENCODER = 'model/encoder.pkl'
PATH_MODEL = 'model/model.pkl'
PATH_MODEL_METRICS = 'model/slice_output.txt'
PATH_SOURCE_DATA = 'data/census.csv'

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
TEST_SIZE = 0.2
