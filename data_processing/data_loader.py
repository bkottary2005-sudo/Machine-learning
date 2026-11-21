import pandas as pd
from pathlib import Path


def load_dataset(file_path):
    """
    Load a CSV dataset into a Pandas DataFrame.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    print("Dataset loaded successfully!")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    return df