<div align="center">

# ⚡ Electricity Demand Forecasting Dashboard ⚡

**An interactive Streamlit dashboard that forecasts electricity demand using time-series analysis and machine learning models like XGBoost.**

</div>

This project provides an end-to-end solution for time-series forecasting, allowing users to upload their own data, visualize predictions from multiple models, and evaluate their performance using key metrics. It serves as a powerful demonstration of applying data science to solve real-world energy management challenges.

---

## 🚀 Key Features

| Feature | Description |
| :--- | :--- |
| 📊 **Interactive Dashboard** | Built with Streamlit for a seamless and intuitive user experience. |
| 📂 **Custom Data Upload** | Upload your own CSV file with a `datetime` and a demand column to generate forecasts on your data. |
| 🛠️ **Automated Feature Engineering** | Automatically creates time-based features (hour, day, month), lag features, and rolling window statistics. |
| 🤖 **Multiple Forecasting Models** | Compare the performance of a Naïve Forecast, Linear Regression, and an advanced **XGBoost** model. |
| 📈 **Dynamic Visualizations** | Uses Plotly to create interactive charts that let you zoom in on actual vs. predicted demand. |
| ⚙️ **Live Performance Metrics** | Instantly view key metrics like **RMSE, MAE, and MAPE** to evaluate model accuracy. |
| 🎛️ **User-Friendly Controls** | Easily adjust the model type and forecast horizon using simple sidebar controls. |

---

## 💻 Technical Stack

-   **Language:** Python 3.11
-   **Dashboard:** Streamlit
-   **Data Manipulation:** Pandas, NumPy
-   **Machine Learning:** Scikit-learn, XGBoost, pmdarima
-   **Visualization:** Plotly, Matplotlib, Seaborn
-   **Deployment:** Streamlit Community Cloud

---

## ⚙️ How It Works

The dashboard follows a systematic process to deliver accurate forecasts:

1.  **Data Ingestion:** The user can either use the default dataset (hourly electricity consumption for the UK) or upload their own CSV file.
2.  **Feature Engineering:** The application automatically generates a rich set of features from the timestamp, which are crucial for the models to identify patterns:
    * **Time-Based Features:** Hour, Day of the week, Day of the year, Month, Year.
    * **Lag Features:** Demand from the previous hour, 24 hours ago (previous day), and 168 hours ago (previous week).
    * **Rolling Window Features:** Rolling mean of demand over the past 24 hours.
3.  **Model Training:** The data is split into training and testing sets. The selected model (Naïve, Linear Regression, or XGBoost) is then trained on the historical data.
4.  **Forecasting & Evaluation:** The trained model predicts future demand. The predictions are then compared against the actual values in the test set, and performance metrics (RMSE, MAE, MAPE) are calculated and displayed.
5.  **Visualization:** The actual demand and the model's predictions are plotted on an interactive chart, making it easy to see how well the model performs.

---

## 📂 Project Structure
├── 📜 app.py                # Main Streamlit application script
├── 📜 requirements.txt      # Project dependencies
├── 📜 runtime.txt           # Specifies the Python version for deployment
└── 📜 README.md             # You are here!

---

## 🚀 Getting Started

To run this project on your local machine, follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/ayusharma176/electricity-demand-forecasting-dashboard.git](https://github.com/ayusharma176/electricity-demand-forecasting-dashboard.git)
    cd electricity-demand-forecasting-dashboard
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit App**
    ```bash
    streamlit run app.py
    ```
    Your browser should automatically open with the dashboard running.

---

## 🙏 Acknowledgments

-   **Data Source:** The default dataset used is the [Hourly Electricity Consumption in the UK](https://www.kaggle.com/datasets/blue-pen-team/hourly-electricity-consumption-in-the-uk) available on Kaggle.
