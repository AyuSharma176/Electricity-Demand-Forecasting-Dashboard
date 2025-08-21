import pandas as pd

def load_data(path: str):
    """Load dataset from CSV"""
    print("📂 Loading data from", path)
    df = pd.read_csv(path)
    print("✅ Data loaded. Shape:", df.shape)
    return df
