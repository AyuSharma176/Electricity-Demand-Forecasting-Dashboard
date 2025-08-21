import pandas as pd

def create_features(df):
    # Handle datetime whether it's index or column
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")
    else:
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)

    # Basic time features
    df["hour"] = df.index.hour.astype("int32")
    df["dayofweek"] = df.index.dayofweek.astype("int32")
    df["month"] = df.index.month.astype("int32")
    df["year"] = df.index.year.astype("int32")

    # Weekend flag
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype("int32")

    # Holiday flag
    try:
        import holidays
        us_holidays = holidays.US()
        df["is_holiday"] = df.index.to_series().apply(
            lambda x: int(x in us_holidays)
        ).astype("int32")
    except ImportError:
        print("⚠️ holidays package not installed. Skipping holiday feature.")
        df["is_holiday"] = 0

    # Lag features
    df["lag1"] = df["AEP_MW"].shift(1)
    df["lag24"] = df["AEP_MW"].shift(24)
    df["lag168"] = df["AEP_MW"].shift(168)

    # Rolling stats
    df["rolling_mean_24h"] = df["AEP_MW"].shift(1).rolling(24).mean()
    df["rolling_mean_168h"] = df["AEP_MW"].shift(1).rolling(168).mean()
    df["rolling_std_24h"] = df["AEP_MW"].shift(1).rolling(24).std()
    df["rolling_min_24h"] = df["AEP_MW"].shift(1).rolling(24).min()
    df["rolling_max_24h"] = df["AEP_MW"].shift(1).rolling(24).max()

    # Drop NA
    df = df.dropna()

    return df
