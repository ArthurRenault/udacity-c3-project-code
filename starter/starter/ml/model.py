import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from starter.starter.constants import FEATURES_SLICING


def train_model(X_train, y_train, **kwargs):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    kwargs : dict
        Additional keyword arguments passed to the sklearn classifier.
    Returns
    -------
    model
        Trained machine learning model.
    """
    rfc = RandomForestClassifier(**kwargs)
    rfc.fit(X_train, y_train)

    return rfc


def save_training_pipeline(encoder, model, encoder_path, model_path):
    with open(encoder_path, 'wb') as file:
        pickle.dump(encoder, file)

    with open(model_path, 'wb') as file:
        pickle.dump(model, file)


def load_training_pipeline(encoder_path, model_path):
    with open(encoder_path, 'rb') as file:
        encoder = pickle.load(file)

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return encoder, model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def generate_metric_report(data, targets, predictions):

    data = data.copy()
    data['target'] = targets
    data['predictions'] = predictions

    precision, recall, fbeta = compute_model_metrics(targets, predictions)

    results = [pd.DataFrame({'slice_name': ['total'],
                             'slice_value': ['total'],
                             'precision': [precision],
                             'recall': [recall],
                             'fbeta': [fbeta]})
               ]
    for col in FEATURES_SLICING:
        result = (data.groupby(col)
                      .apply(lambda x: compute_model_metrics(x['target'], x['predictions']))
                      .rename('metrics')
                      .reset_index()
                      .rename(columns={col: 'slice_value'})
                      .assign(slice_name=col)
                  )
        result[['precision', 'recall', 'fbeta']] = result['metrics'].values.tolist()
        result.drop(columns='metrics', inplace=True)
        results.append(result)

    report = pd.concat(results, ignore_index=True)

    return report


def save_metric_report(report, file_path):

    with open(file_path, 'w') as file:
        print(report, file=file)


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
