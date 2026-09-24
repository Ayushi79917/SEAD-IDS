"""
STEP 1: Load and preprocess CIC-IDS2017 data.

What this does:
- Loads the CSV files from data/
- Cleans column names, removes NaN/infinite values
- Encodes labels (attack type as text -> number)
- Splits into "known" attacks (for training) and one "unseen" attack
  type (held out to simulate a zero-day attack later)
- Normalizes features and saves the processed data

TODO for you:
- Update DATA_FOLDER path to where your CSVs are
- Choose which attack type to hold out as your "zero-day" (see
  HOLD_OUT_ATTACK below) - pick one that has a good number of samples
"""

import pandas as pd
import numpy as np
import glob
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_FOLDER = "../data"
HOLD_OUT_ATTACK = "PortScan"  # <- change this: the "zero-day" attack to hide during training


def load_all_csvs(folder):
    all_files = glob.glob(os.path.join(folder, "*.csv"))
    print(f"Found {len(all_files)} CSV files")
    df_list = [pd.read_csv(f, low_memory=False) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    return df


def clean_data(df):
    df.columns = df.columns.str.strip()  # remove extra spaces in column names
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def main():
    df = load_all_csvs(DATA_FOLDER)
    df = clean_data(df)

    print("Label distribution:\n", df["Label"].value_counts())

    # Separate the zero-day (held-out) attack from the rest
    zero_day_df = df[df["Label"] == HOLD_OUT_ATTACK].copy()
    known_df = df[df["Label"] != HOLD_OUT_ATTACK].copy()

    # Encode labels for known classes only
    le = LabelEncoder()
    known_df["Label_enc"] = le.fit_transform(known_df["Label"])

    # Columns that identify traffic (for dashboard display) but should
    # NOT be used as ML features (IP is high-cardinality, not a real
    # traffic pattern signal, and would cause overfitting/leakage)
    identifier_cols = [c for c in
                        ["Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP",
                         "Source Port", "Src Port", "Destination Port", "Dst Port",
                         "Protocol", "Timestamp"]
                        if c in df.columns]

    # Normalize features (fit scaler on known data only, apply to both)
    feature_cols = [c for c in df.columns if c not in ["Label"] + identifier_cols]
    scaler = StandardScaler()
    known_df[feature_cols] = scaler.fit_transform(known_df[feature_cols])
    zero_day_df[feature_cols] = scaler.transform(zero_day_df[feature_cols])

    known_df.to_csv("../data/processed_known.csv", index=False)
    zero_day_df.to_csv("../data/processed_zero_day.csv", index=False)

    # Save scaler + label encoder + feature column order so the dashboard
    # can preprocess new uploaded files the exact same way
    joblib.dump(scaler, "../results/scaler.pkl")
    joblib.dump(le, "../results/label_encoder.pkl")
    joblib.dump(feature_cols, "../results/feature_cols.pkl")
    print("Saved processed_known.csv, processed_zero_day.csv, scaler.pkl, label_encoder.pkl, feature_cols.pkl")


if __name__ == "__main__":
    main()
