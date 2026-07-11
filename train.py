import argparse
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# Bank marketing dataset (public sample data).
DATA_URL = (
    "https://automlsamplenotebookdata.blob.core.windows.net/"
    "automl-sample-notebook-data/bankmarketing_train.csv"
)


def clean_data(data):
    """Clean and one-hot encode the bank marketing dataset."""
    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    weekdays = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7}

    x_df = data.to_pandas_dataframe().dropna()

    jobs = pd.get_dummies(x_df.job, prefix="job")
    x_df.drop("job", inplace=True, axis=1)
    x_df = x_df.join(jobs)

    x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)

    contact = pd.get_dummies(x_df.contact, prefix="contact")
    x_df.drop("contact", inplace=True, axis=1)
    x_df = x_df.join(contact)

    education = pd.get_dummies(x_df.education, prefix="education")
    x_df.drop("education", inplace=True, axis=1)
    x_df = x_df.join(education)

    x_df["month"] = x_df.month.map(months)
    x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

    y_df = x_df.pop("y").apply(lambda s: 1 if s == "yes" else 0)
    return x_df, y_df


def main():
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    run = Run.get_context()
    run.log("Regularization Strength:", float(args.C))
    run.log("Max iterations:", int(args.max_iter))

    # Load the data. validate=False avoids the remote "Authentication failed
    # when trying to access the stream" error; the data is still read by
    # to_pandas_dataframe().
    ds = TabularDatasetFactory.from_delimited_files(path=DATA_URL, validate=False)
    x, y = clean_data(ds)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", float(accuracy))

    # Save the model so the best HyperDrive run can register it.
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/hyperdrive_model.joblib")


if __name__ == "__main__":
    main()