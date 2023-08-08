import argparse
import glob
import os
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split


def main(args):
    mlflow.autolog() 
    mlflow.start_run()
    df = get_csvs_df(args.training_data)
    X_train, X_test, y_train, y_test = split_data(df)
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def split_data(df):
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values
    mlflow.log_param("X", X)

    mlflow.log_param("unique_count", np.unique(y, return_counts=True))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    
    return X_train, X_test, y_train, y_test

def train_model(reg_rate, X_train, X_test, y_train, y_test):
    return LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)
    

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

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)