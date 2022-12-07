"""
Script to train machine learning model.

To execute the script run the following commands:
$ cd starter/
$ python starter/train_model.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from starter.constants import (
        PATH_ENCODER,
        PATH_MODEL,
        PATH_MODEL_METRICS,
        PATH_SOURCE_DATA,
        FEATURES_CATEGORICAL,
        FEATURE_TARGET,
        N_ESTIMATORS,
        RANDOM_SEED,
        TEST_SIZE
    )
from starter.ml import model
from starter.ml.data import process_data


if __name__ == '__main__':

    raw_data = pd.read_csv(PATH_SOURCE_DATA, skipinitialspace=True)

    train, test = train_test_split(raw_data, test_size=TEST_SIZE)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=FEATURES_CATEGORICAL, label=FEATURE_TARGET, training=True
    )

    X_test, y_test, _, _ = process_data(
            test, categorical_features=FEATURES_CATEGORICAL, label=FEATURE_TARGET, training=False, encoder=encoder, lb=lb
            )

    trained_model = model.train_model(X_train, y_train, n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)

    model.save_training_pipeline(encoder, trained_model, PATH_ENCODER, PATH_MODEL)

    predictions = model.inference(trained_model, X_test)

    report = model.generate_metric_report(test, y_test, predictions)

    model.save_metric_report(report, PATH_MODEL_METRICS)
