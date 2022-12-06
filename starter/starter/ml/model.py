import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


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
