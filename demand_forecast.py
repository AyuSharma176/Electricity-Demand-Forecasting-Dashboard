# demand_forecast.py

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from src.features import create_features
from src.train import train_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import numpy as np


def evaluate_model(name, y_true, y_pred):
    """Utility to print RMSE and MAE"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"📊 {name} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    return rmse, mae


def main():
    print("📂 Loading data from data/AEP_hourly.csv")
    df = pd.read_csv("data/AEP_hourly.csv", parse_dates=["Datetime"])
    df = df.set_index("Datetime")
    print(f"✅ Data loaded. Shape: {df.shape}")

    # --- Feature Engineering ---
    df = create_features(df)

    # Add lag features
    for lag in [1, 24, 168]:
        df[f"lag_{lag}"] = df["AEP_MW"].shift(lag)

    # Drop rows with NaN values
    df = df.dropna()

    # Train/test split
    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = train.drop("AEP_MW", axis=1), train["AEP_MW"]
    X_test, y_test = test.drop("AEP_MW", axis=1), test["AEP_MW"]

    # ---------------------------
    # 1️⃣ Baseline: Naïve Forecast
    # ---------------------------
    naive_pred = test["lag_1"]
    evaluate_model("Naïve Forecast", y_test, naive_pred)

    # ---------------------------
    # 2️⃣ Baseline: Linear Regression
    # ---------------------------
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    evaluate_model("Linear Regression", y_test, lr_pred)

    # ---------------------------
    # 3️⃣ XGBoost Model
    # ---------------------------
    model_path = "models/xgb_model.pkl"
    os.makedirs("models", exist_ok=True)

    if os.path.exists(model_path):
        print("💾 Found saved model, loading...")
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
    else:
        print("⚡ No saved model found. Training new one...")
        model, _, _, _ = train_model(df)
        y_pred = model.predict(X_test)
        joblib.dump(model, model_path)
        print(f"✅ Model saved to {model_path}")

    evaluate_model("XGBoost", y_test, y_pred)

    # --- Visualization: Save instead of show ---
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index[:200], y_test[:200], label="Actual", color="blue")
    plt.plot(y_test.index[:200], naive_pred[:200], label="Naïve", color="green", alpha=0.6)
    plt.plot(y_test.index[:200], lr_pred[:200], label="LinearReg", color="orange", alpha=0.7)
    plt.plot(y_test.index[:200], y_pred[:200], label="XGBoost", color="red")
    plt.xlabel("Time")
    plt.ylabel("MW")
    plt.title("Electricity Demand Prediction (first 200 test hours)")
    plt.legend()
    plt.savefig("plots/model_comparison.png")
    plt.close()

    print("📊 Saved model comparison plot -> plots/model_comparison.png")


if __name__ == "__main__":
    main()
