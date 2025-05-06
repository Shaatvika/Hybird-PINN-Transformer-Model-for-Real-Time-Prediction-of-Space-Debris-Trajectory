import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['EPOCH'] = pd.to_datetime(df['EPOCH'])
    ref_time = df['EPOCH'].min()
    df['TIME_SINCE_REF'] = (df['EPOCH'] - ref_time).dt.total_seconds()

    features = [
        'TIME_SINCE_REF', 'MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION',
        'RA_OF_ASC_NODE', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY',
        'BSTAR', 'MEAN_MOTION_DOT', 'MEAN_MOTION_DDOT'
    ]
    df = df.dropna(subset=features)
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    X = df[features].values[:-1]
    y = df[features].values[1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test, scaler, ref_time, features
