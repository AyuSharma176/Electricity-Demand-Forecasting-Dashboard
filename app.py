# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

from src.features import create_features
from src.train import train_model

# -------------------
# Utility Functions
# -------------------
def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"Model": name, "RMSE": rmse, "MAE": mae, "MAPE": mape}


def load_data(file):
    df = pd.read_csv(file, parse_dates=["Datetime"])
    df = df.set_index("Datetime")

    # If "AEP_MW" not present, detect numeric column
    if "AEP_MW" not in df.columns:
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        if len(numeric_cols) == 0:
            st.error("❌ No numeric column found in dataset.")
            st.stop()
        elif len(numeric_cols) == 1:
            df = df.rename(columns={numeric_cols[0]: "AEP_MW"})
            st.warning(f"⚠️ Column `{numeric_cols[0]}` renamed to `AEP_MW`")
        else:
            # Let user pick target column
            target_col = st.sidebar.selectbox("Select target demand column", numeric_cols)
            df = df.rename(columns={target_col: "AEP_MW"})
            st.success(f"✅ Using `{target_col}` as target demand column")

    return df


# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="⚡ Demand Forecasting Dashboard", layout="wide")

st.title("⚡ Electricity Demand Forecasting Dashboard")
st.markdown("Compare electricity demand prediction models interactively.")

# Sidebar
st.sidebar.header("⚙️ Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV with 'Datetime' + demand column", type=["csv"])
forecast_horizon = st.sidebar.slider("Number of test points to visualize", 100, 1000, 300, step=50)
model_choice = st.sidebar.selectbox("Choose Model", ["Naïve Forecast", "Linear Regression", "XGBoost"])

# Load dataset
if uploaded_file:
    df = load_data(uploaded_file)
    st.success("✅ Custom dataset uploaded!")
else:
    st.info("ℹ️ Using default dataset: `data/AEP_hourly.csv`")
    df = load_data("data/AEP_hourly.csv")

# Feature engineering
df = create_features(df)

# Add lags
for lag in [1, 24, 168]:
    df[f"lag_{lag}"] = df["AEP_MW"].shift(lag)

df = df.dropna()

# Train/test split
split_idx = int(len(df) * 0.8)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]
X_train, y_train = train.drop("AEP_MW", axis=1), train["AEP_MW"]
X_test, y_test = test.drop("AEP_MW", axis=1), test["AEP_MW"]

# -------------------
# Model Selection
# -------------------
if model_choice == "Naïve Forecast":
    y_pred = test["lag_1"]
    metrics = evaluate_model("Naïve", y_test, y_pred)

elif model_choice == "Linear Regression":
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    metrics = evaluate_model("Linear Regression", y_test, y_pred)

else:  # XGBoost
    model_path = "models/xgb_model.pkl"
    try:
        model = joblib.load(model_path)
    except:
        model, _, _, _ = train_model(df)
        joblib.dump(model, model_path)
    y_pred = model.predict(X_test)
    metrics = evaluate_model("XGBoost", y_test, y_pred)

# -------------------
# Metrics Section
# -------------------
st.subheader("📊 Model Performance Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{metrics['RMSE']:.2f}")
col2.metric("MAE", f"{metrics['MAE']:.2f}")
col3.metric("MAPE (%)", f"{metrics['MAPE']:.2f}")

# -------------------
# Plot Actual vs Predicted
# -------------------
st.subheader("📈 Demand Prediction Visualization")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_test.index[:forecast_horizon], y_test[:forecast_horizon], label="Actual", color="blue")
ax.plot(y_test.index[:forecast_horizon], y_pred[:forecast_horizon], label=f"Predicted ({model_choice})", color="red")
ax.set_xlabel("Time")
ax.set_ylabel("MW")
ax.legend()
st.pyplot(fig)

st.success("✅ Forecasting complete!")
