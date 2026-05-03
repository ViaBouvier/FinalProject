from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
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


def preprocess_and_train(X, y, lags=(1, 3, 24)):
    df = X.copy()
    df = df.reset_index(drop=True)
    y = y.reset_index(drop=True)
    df['traffic_volume'] = y

    #Require datetime to create time-based and lag features
    if 'date_time' in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'])
        df = df.sort_values('date_time').reset_index(drop=True)
        df['hour'] = df['date_time'].dt.hour
        df['dayofweek'] = df['date_time'].dt.dayofweek
    else:
        #If no datetime, create placeholder time features from index
        df['hour'] = df.index % 24
        df['dayofweek'] = (df.index // 24) % 7

    #Create lag features for historical traffic volume
    for lag in lags:
        df[f'lag_{lag}'] = df['traffic_volume'].shift(lag)

    #We will use weather_main, the time features, and lag features
    candidate_cols = []
    if 'weather_main' in df.columns:
        candidate_cols.append('weather_main')
    if 'weather_description' in df.columns:
        candidate_cols.append('weather_description')

    candidate_cols += ['hour', 'dayofweek']
    candidate_cols += [f'lag_{lag}' for lag in lags]

    X_features = df[candidate_cols].copy()
    y_target = df['traffic_volume'].copy()

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

    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.3f}")

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
