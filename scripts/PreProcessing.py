import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer


def preprocess_dataset(file_path, output_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Identify numeric and categorical columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Exclude non-feature columns (like session_id, labels)
    non_feature_columns = ['session_id', 'src_ip', 'dst_ip', 'source_ip', 'dest_ip']  # Avoid processing IDs
    label_columns = ['attack_detected', 'label', 'attacklabel', 'class', 'target']  # Possible attack labels
    numeric_columns = [col for col in numeric_columns if col not in non_feature_columns + label_columns]

    # Handle missing values for numeric columns
    if numeric_columns:
        numeric_imputer = SimpleImputer(strategy='mean')
        df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

    # Handle missing values for categorical columns
    if categorical_columns:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

    # Apply MinMaxScaler to numerical features (excluding IDs)
    scaler = MinMaxScaler()
    if numeric_columns:
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # One-Hot Encode categorical variables
    if categorical_columns:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_data = encoder.fit_transform(df[categorical_columns])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
        df = df.drop(columns=categorical_columns).reset_index(drop=True)
        df = pd.concat([df, encoded_df], axis=1)

    # Standardize attack labels across datasets
    for label_col in label_columns:
        if label_col in df.columns:
            df.rename(columns={label_col: 'attack_label'}, inplace=True)
            break  # Rename only the first found label

    # Save preprocessed dataset
    df.to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved to {output_path}")


# Preprocess each dataset
datasets = [
    'cybersecurity_intrusion_data.csv',
    'embedded_system_network_security_dataset.csv',
    'cybersecurity_dataset.csv',
    'GPS_Spoofing_Attack.csv',
    'DDoS_dataset.csv'
]

for dataset in datasets:
    input_path = f'/Path/to/File/{dataset}'
    output_path = f'/Path/to/File/preprocessed_{dataset}'
    preprocess_dataset(input_path, output_path)
