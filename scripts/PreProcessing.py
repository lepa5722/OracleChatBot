import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from abc import ABC, abstractmethod
import os

# ============ STEP 1: ABSTRACT BASE CLASS ============ #
class GenericDataFormatter(ABC):
    """Abstract base class for data formatting across different network datasets."""

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.label_encoders = {}

    @abstractmethod
    def load_data(self, file_path: str):
        """Loads raw dataset from file."""
        pass

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs dataset-specific preprocessing."""
        pass

    def normalize(self, df: pd.DataFrame, columns: list):
        """Applies MinMax scaling to numeric columns."""
        df[columns] = self.scaler.fit_transform(df[columns])
        return df

    def encode_categorical(self, df: pd.DataFrame, columns: list):
        """Encodes categorical variables using LabelEncoder."""
        for col in columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        return df

    def handle_missing_values(self, df: pd.DataFrame):
        """Handles missing values by filling NaNs with median."""
        df.fillna(df.median(), inplace=True)
        return df

    def create_time_features(self, df: pd.DataFrame, time_col: str):
        """Extracts time-based features from a timestamp column."""
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df["Hour"] = df[time_col].dt.hour
        df["Minute"] = df[time_col].dt.minute
        df["Second"] = df[time_col].dt.second
        df.drop(columns=[time_col], inplace=True)
        return df

    def split_data(self, df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
        """Splits data into train, validation, and test sets."""
        train_size = int(len(df) * train_ratio)
        val_size = int(len(df) * val_ratio)

        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]

        return train_df, val_df, test_df

    def save_splits(self, train_df, val_df, test_df, output_dir):
        """Saves the train, validation, and test splits as CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        print(f"Saved processed data to {output_dir}")

# ============ STEP 2: CUSTOM FORMATTER FOR NETWORK TRAFFIC ============ #
class TrafficDataFormatter(GenericDataFormatter):
    """Custom implementation for processing network traffic datasets."""

    def load_data(self, file_path: str):
        """Loads dataset from CSV file."""
        return pd.read_csv(file_path)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all preprocessing steps to the dataset."""
        # Extract time-based features
        df = self.create_time_features(df, time_col="Timestamp")

        # Identify numerical and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Normalize numeric features
        df = self.normalize(df, numeric_cols)

        # Encode categorical variables
        df = self.encode_categorical(df, categorical_cols)

        # Handle missing values
        df = self.handle_missing_values(df)

        return df

# ============ STEP 3: EXECUTION FOR A DATASET ============ #
if __name__ == "__main__":
    # Example dataset paths
    dataset_paths = {
        "5GTrafficDataset": "data/raw/5GTrafficDataset.csv",
        "CSE-CIC-IDS2018": "data/raw/CSE-CIC-IDS2018.csv",
        "INDDOS24Dataset": "data/raw/INDDOS24Dataset.csv",
        "MicrosoftSecurityIncidentPrediction": "data/raw/MicrosoftSecurityIncidentPrediction.csv",
        "TokyoDroneCommunication": "data/raw/TokyoDroneCommunication.csv"
    }

    for dataset_name, file_path in dataset_paths.items():
        print(f"Processing {dataset_name}...")

        # Initialize formatter
        formatter = TrafficDataFormatter()

        # Load and preprocess dataset
        df = formatter.load_data(file_path)
        processed_df = formatter.preprocess(df)

        # Split and save data
        train_df, val_df, test_df = formatter.split_data(processed_df)
        formatter.save_splits(train_df, val_df, test_df, output_dir=f"data/preprocessed/{dataset_name}")

    print("All datasets processed successfully!")

