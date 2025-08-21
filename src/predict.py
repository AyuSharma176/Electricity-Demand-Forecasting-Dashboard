import joblib
import pandas as pd

def load_model(path="models/demand_model.pkl"):
    return joblib.load(path)

def predict_future(df, steps=24, model_path="models/demand_model.pkl"):
    model, scaler = load_model(model_path)

    # take last row as base for future prediction
    future_data = df.iloc[-1:].copy()

    preds = []
    for _ in range(steps):
        X = future_data.drop("AEP_MW", axis=1)
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)[0]

        preds.append(y_pred)

        # shift lag features
        future_data["lag_1"] = y_pred
        future_data["lag_24"] = future_data["lag_1"].shift(23)
        future_data["lag_168"] = future_data["lag_1"].shift(167)

    return preds
