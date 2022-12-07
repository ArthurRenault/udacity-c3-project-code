from starter.ml.model import compute_model_metrics


def test_compute_metrics():
    mock_y = [0, 0, 0, 1]
    mock_predictions = [0, 0, 0, 1]

    precision, recall, fbeta = compute_model_metrics(mock_y, mock_predictions)

    assert (precision, recall, fbeta) == (1, 1, 1)
