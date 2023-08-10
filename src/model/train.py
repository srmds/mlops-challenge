import argparse
import glob
import os
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import ModelType

MODEL_NAME = "diabetes-model"
DESCRIPTION = "model for diabetes detection"


def main(args):
    mlflow.autolog()
    mlflow_run = mlflow.active_run()


    df = get_csvs_df(args.training_data)
    X_train, X_test, y_train, y_test = split_data(df)
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)

    # If model is trained in prd, then we need to register the model,
    # so it can be used to deploy it as an API endpoint
    autolog_run = mlflow.last_active_run()

    if args.env == "prd":
        register_model(
            args,
            autolog_run.info.run_id,
            f"{MODEL_NAME}-{args.env}",
            DESCRIPTION
        )


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def split_data(df):
    X, y = df[
        ['Pregnancies',
         'PlasmaGlucose',
         'DiastolicBloodPressure',
         'TricepsThickness',
         'SerumInsulin',
         'BMI',
         'DiabetesPedigree',
         'Age']].values, df['Diabetic'].values

    mlflow.log_param("unique_count", np.unique(y, return_counts=True))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0
    )

    return X_train, X_test, y_train, y_test


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    return LogisticRegression(
        C=1/reg_rate,
        solver="liblinear"
    ).fit(X_train, y_train)


def register_model(args, run_id, model_name, description):
    ml_client = MLClient(
        DefaultAzureCredential(),
        args.subscription_id,
        args.resource_group,
        args.workspace
    )

    run_model = Model(
        path=f"runs:/{run_id}/model/",
        name=model_name,
        description=description,
        type=ModelType.MLFLOW
    )

    ml_client.models.create_or_update(run_model)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training_data",
        dest='training_data',
        type=str
    )
    parser.add_argument(
        "--reg_rate",
        dest='reg_rate',
        type=float,
        default=0.01
    )
    parser.add_argument(
        "--subscription_id",
        dest='subscription_id',
        type=str
    )
    parser.add_argument(
        "--resource_group",
        dest='resource_group',
        type=str,
    )
    parser.add_argument(
        "--workspace",
        dest='workspace',
        type=str,
    )
    parser.add_argument(
        "--env",
        dest='env',
        type=str,
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
