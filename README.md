# ⚡ Electricity Demand Forecasting Dashboard

📊 **Forecast electricity demand with Machine Learning and visualize results interactively.**  
This project applies **time-series forecasting techniques** and machine learning models to predict power demand, providing both **baseline models** and an advanced **XGBoost regressor**, all inside a user-friendly **Streamlit dashboard**.

🚀 **Live Demo**: [Electricity Demand Forecasting Dashboard](https://electricity-demand-forecasting-dashboard.streamlit.app/)

---

## 🌍 Why This Project?

Electricity demand forecasting plays a crucial role in **energy management, grid stability, and cost optimization**.  
Power suppliers, government agencies, and industries rely on accurate forecasts to:

- ⚡ Balance supply and demand in real-time.  
- 💸 Optimize generation costs and reduce wastage.  
- 🌱 Integrate renewable sources effectively.  
- 🏙️ Plan for future infrastructure and energy needs.  

By combining **feature engineering** with **machine learning models**, this project demonstrates how data-driven approaches can improve forecast accuracy over traditional baselines.

---

## 🚀 Key Features

✅ **Interactive Dashboard** built with Streamlit  
✅ **Upload custom datasets** (CSV with `Datetime` + demand column)  
✅ **Feature Engineering**:  
   - Time-based features (hour, day, month, year, weekday, weekend/holiday flag)  
   - Lag features (previous 1, 24, 168 hours)  
   - Rolling averages  
✅ **Model Comparison**:  
   - Naïve Forecast (baseline)  
   - Linear Regression  
   - XGBoost (advanced ML model)  
✅ **Metrics Section** – RMSE, MAE, MAPE displayed live  
✅ **Visualization** – Interactive plots of Actual vs Predicted demand  
✅ **Sidebar Controls**:  
   - Choose model type  
   - Adjust forecast horizon  
   - Upload your own CSV  
✅ **Deployed on Streamlit Cloud** – No setup needed for recruiters  

---

## 📊 Example Dashboard

![Dashboard Preview](plots/dashboard.png)  
*(XGBoost model predicting electricity demand for test horizon)*

---

## 🏗️ Project Structure

