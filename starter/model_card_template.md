# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The trained model is a Random Forest Classifier from
[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
with the default hyper-parameters. The model has been fed a fraction of ``TEST_SIZE=0.2`` of the data samples
randomly selected. Neither stratified sampling, nor cross validation have been used.
The categorical feature encoder and the model are saved in the ``model/`` folder.

## Intended Use
The model aims to predict whether or not an individual has an income greater than $50,000.
A dataset provided by the University of California, Irvine is used.
The dataset contains numerous demographic features.
See the following [link](https://archive.ics.uci.edu/ml/datasets/census+income) for more information on the dataset.

## Training Data
80% of the samples are used during training.

## Evaluation Data
20% of the samples are used during evaluation. No statistical tests have been performed
to ensure that train and validation sets are similar alongside various features.

## Metrics
To evaluate the performances of the model 3 metrics have been used:
- Precision
- Recall
- Fbeta

The current model version scores `precision=0.73`, `recall=0.60` and `fbeta=0.66`.
The 3 metrics are also computed against various slices of the data. All the results are
provided in ``model/slice_output.txt``.

## Ethical Considerations
Race-based analyses are not allowed in the EU.

## Caveats and Recommendations
Some classes are imbalanced which may explains why the model performs poorly on some less-represented slices.
Rather than spending days tweaking the hyper parameters or selecting another model,
we would recommend to collect more data for those classes.