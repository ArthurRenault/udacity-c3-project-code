COLUMN_LABEL = 'salary'

PATH_ENCODER = 'model/encoder.pkl'
PATH_MODEL = 'model/model.pkl'
PATH_MODEL_METRICS = 'slice_output.txt'
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
TEST_SIZE = 0.2
