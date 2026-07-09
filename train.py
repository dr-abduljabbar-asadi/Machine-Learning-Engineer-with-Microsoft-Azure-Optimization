"""Train a Logistic Regression model on the Bank Marketing dataset.

This script is the training entry point for an Azure ML HyperDrive run. It
loads the Bank Marketing data, cleans and encodes it, trains a Logistic
Regression classifier with configurable hyperparameters, logs the accuracy,
and saves the trained model to the ``outputs`` folder.
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# Location of the training data.
DATA_URL = (
    "https://automlsamplenotebookdata.blob.core.windows.net/"
    "automl-sample-notebook-data/bankmarketing_train.csv"
)

# Mappings used to encode categorical date fields as integers.
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
WEEKDAYS = {
    "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7,
}

TEST_SIZE = 0.3
RANDOM_STATE = 42
MODEL_OUTPUT_PATH = "outputs/hyperdrive_model.joblib"


def clean_data(dataset):
    """Clean and one-hot encode the raw dataset.

    Args:
        dataset: An Azure ML ``TabularDataset`` containing the raw data.

    Returns:
        A tuple ``(features, labels)`` of the feature matrix and target labels.
    """
    features = dataset.to_pandas_dataframe().dropna()

    # One-hot encode multi-category columns.
    for column in ("job", "contact", "education"):
        dummies = pd.get_dummies(features[column], prefix=column)
        features.drop(column, axis=1, inplace=True)
        features = features.join(dummies)

    # Binary-encode yes/no and category-specific columns.
    features["marital"] = features.marital.apply(lambda s: 1 if s == "married" else 0)
    features["default"] = features.default.apply(lambda s: 1 if s == "yes" else 0)
    features["housing"] = features.housing.apply(lambda s: 1 if s == "yes" else 0)
    features["loan"] = features.loan.apply(lambda s: 1 if s == "yes" else 0)
    features["poutcome"] = features.poutcome.apply(lambda s: 1 if s == "success" else 0)

    # Map date fields to integers.
    features["month"] = features.month.map(MONTHS)
    features["day_of_week"] = features.day_of_week.map(WEEKDAYS)

    # Extract and binary-encode the target column.
    labels = features.pop("y").apply(lambda s: 1 if s == "yes" else 0)

    return features, labels


def parse_args():
    """Parse the command-line hyperparameters."""
    parser = argparse.ArgumentParser(
        description="Train a Logistic Regression model on the Bank Marketing dataset."
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse of regularization strength. Smaller values cause stronger regularization.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum number of iterations to converge.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run = Run.get_context()

    # Load, clean, and split the data.
    dataset = TabularDatasetFactory.from_delimited_files(path=DATA_URL)
    features, labels = clean_data(dataset)
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Log the hyperparameters for this run.
    run.log("Regularization Strength:", float(args.C))
    run.log("Max iterations:", int(args.max_iter))

    # Train and evaluate the model.
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(features_train, labels_train)
    accuracy = model.score(features_test, labels_test)
    run.log("Accuracy", float(accuracy))

    # Persist the trained model.
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(value=model, filename=MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()
