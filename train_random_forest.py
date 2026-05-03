from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import pickle


def load_data():
    ds = fetch_ucirepo(id=492)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    # ensure y is a 1-d array/series
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]
    return X, y


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

    # drop rows with NaNs (e.g., from lagging)
    valid_idx = X_features.dropna().index
    return X_features.loc[valid_idx].reset_index(drop=True), y_target.loc[valid_idx].reset_index(drop=True)


def load_pipeline(path='rf_pipeline.pkl'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline file not found: {path}. Run train_random_forest.py first.")
    with open(path, 'rb') as f:
        return pickle.load(f)


def preprocess_and_train(X, y, lags=(1, 3, 24)):
    X_features, y_target = build_features(X, y, lags=lags)

    cat_cols = X_features.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    X_train, _, y_train, _ = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)


    return pipeline


def main():
    X, y = load_data()
    model_pipeline = preprocess_and_train(X, y, lags=(1, 3, 24))
    #Save full pipeline (preprocessing + model)
    with open('rf_pipeline.pkl', 'wb') as f:
        pickle.dump(model_pipeline, f)
    print('Saved pipeline to rf_pipeline.pkl')


if __name__ == '__main__':
    main()
