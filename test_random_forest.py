import os
import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

from train_random_forest import load_data


def load_pipeline(path='rf_pipeline.pkl'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline file not found: {path}. Run train_random_forest.py first.")
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_features(X, y, lags=(1, 3, 24)):
    df = X.copy().reset_index(drop=True)
    y = y.copy().reset_index(drop=True)
    df['traffic_volume'] = y

    if 'date_time' in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'])
        df = df.sort_values('date_time').reset_index(drop=True)
        df['hour'] = df['date_time'].dt.hour
        df['dayofweek'] = df['date_time'].dt.dayofweek
    else:
        df['hour'] = df.index % 24
        df['dayofweek'] = (df.index // 24) % 7

    for lag in lags:
        df[f'lag_{lag}'] = df['traffic_volume'].shift(lag)

    candidate_cols = []
    if 'weather_main' in df.columns:
        candidate_cols.append('weather_main')
    if 'weather_description' in df.columns:
        candidate_cols.append('weather_description')

    candidate_cols += ['hour', 'dayofweek']
    candidate_cols += [f'lag_{lag}' for lag in lags]

    X_features = df[candidate_cols].copy()
    y_target = df['traffic_volume'].copy()

    #drop rows with NaNs (e.g., from lagging)
    valid_idx = X_features.dropna().index
    return X_features.loc[valid_idx].reset_index(drop=True), y_target.loc[valid_idx].reset_index(drop=True)


def main():
    pipeline = load_pipeline()
    X, y = load_data()
    Xf, yf = build_features(X, y)

    if len(Xf) == 0:
        print('No valid rows after feature construction. Cannot evaluate.')
        return

    _, X_test, _, y_test = train_test_split(Xf, yf, test_size=0.2, random_state=42)

    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Test MAE: {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R2: {r2:.3f}")

if __name__ == '__main__':
    main()
