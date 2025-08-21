import os
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

def train_model(df):
    X = df.drop("AEP_MW", axis=1)
    y = df["AEP_MW"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Model
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    print("✅ Model trained successfully")

    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)

    # --- 📊 Plot 1: Actual vs Predicted ---
    plt.figure(figsize=(12,6))
    plt.plot(y_test.values[:500], label="Actual", linewidth=2)
    plt.plot(y_pred[:500], label="Predicted", linewidth=2)
    plt.title("Actual vs Predicted Demand (first 500 test points)")
    plt.xlabel("Time steps")
    plt.ylabel("Demand (MW)")
    plt.legend()
    plt.savefig("plots/actual_vs_predicted.png")
    plt.close()

    # --- 📊 Plot 2: Feature Importance ---
    xgb.plot_importance(model, importance_type="weight", max_num_features=15, height=0.5)
    plt.title("Feature Importance")
    plt.savefig("plots/feature_importance.png")
    plt.close()

    # --- 📊 Plot 3: Residual Plot ---
    residuals = y_test.values - y_pred
    plt.figure(figsize=(12,6))
    plt.scatter(range(len(residuals)), residuals, alpha=0.5, color="red")
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Residual Plot (Errors: Actual - Predicted)")
    plt.xlabel("Test sample index")
    plt.ylabel("Residuals")
    plt.savefig("plots/residual_plot.png")
    plt.close()

    # --- 📊 Plot 4: Residual Distribution ---
    plt.figure(figsize=(10,6))
    plt.hist(residuals, bins=50, color="blue", alpha=0.6, density=True)
    plt.axvline(0, color="black", linestyle="--")
    plt.title("Residual Distribution")
    plt.xlabel("Residual value")
    plt.ylabel("Density")
    plt.savefig("plots/residual_distribution.png")
    plt.close()

    # --- 💾 Save trained model ---
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_model.pkl")
    print("💾 Model saved to models/xgb_model.pkl")

    return model, X_test, y_test, y_pred
