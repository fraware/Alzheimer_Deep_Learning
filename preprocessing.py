import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(df, impute=True):
    """
    Takes in a DataFrame (already loaded) and performs:
      - Imputation or dropping missing values
      - Train/val/test split
      - Scaling
    Returns scaled splits + the scaler object.
    """
    if impute:
        # Imputation approach
        # e.g., fill SES with median grouped by EDUC
        df["SES"].fillna(df.groupby("EDUC")["SES"].transform("median"), inplace=True)
    else:
        # Drop approach
        df = df.dropna(axis=0, how="any")

    Y = df["Group"].values
    X = df[["M/F", "Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"]]

    X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, random_state=0)

    scaler = MinMaxScaler().fit(X_trainval)

    X_trainval_scaled = scaler.transform(X_trainval)
    X_test_scaled = scaler.transform(X_test)

    return X_trainval_scaled, X_test_scaled, Y_trainval, Y_test, scaler


if __name__ == "__main__":
    # Simple test
    df = pd.read_csv("../data/oasis_longitudinal.csv")
    # Suppose we do the same filtering here that was done in EDA
    df = df.loc[df["Visit"] == 1]
    df = df.reset_index(drop=True)
    df["M/F"] = df["M/F"].replace(["F", "M"], [0, 1])
    df["Group"] = df["Group"].replace(["Converted"], ["Demented"])
    df["Group"] = df["Group"].replace(["Demented", "Nondemented"], [1, 0])
    df = df.drop(["MRI ID", "Visit", "Hand"], axis=1)

    X_trainval_scaled, X_test_scaled, Y_trainval, Y_test, scaler = preprocess_data(df)
    print("Train shape:", X_trainval_scaled.shape)
    print("Test shape:", X_test_scaled.shape)
