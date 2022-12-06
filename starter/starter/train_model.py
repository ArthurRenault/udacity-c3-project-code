"""Script to train machine learning model."""
import pandas as pd
from sklearn.model_selection import train_test_split

from ml import model
from ml.data import process_data


if __name__ == '__main__':

    raw_data = pd.read_csv('data/census.csv', skipinitialspace=True)

    train, test = train_test_split(raw_data, test_size=0.20)

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
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
            test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
            )

    trained_model = model.train_model(X_train, y_train)

    model.save_training_pipeline(encoder, trained_model, 'model/encoder.pkl', 'model/model.pkl')

    predictions = model.inference(trained_model, X_test)

    report = model.generate_metric_report(test, y_test, predictions)

    model.save_metric_report(report, 'model/results.txt')
