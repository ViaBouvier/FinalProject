import os
import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

from train_random_forest import load_data, build_features, load_pipeline


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
