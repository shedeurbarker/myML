import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import joblib

# Create output directories
output_dir = 'results/prepare_ml_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{output_dir}/prepare_ml_data.log', mode='a'),
        logging.StreamHandler()
    ]
)

# Debug logging to confirm script execution
logging.debug("Script started")

# Load feature names from scripts/feature_names.npy
feature_names = np.load('scripts/feature_names.npy')
logging.info(f"Features used: {feature_names}")

def prepare_data():
    """Prepare data for machine learning."""
    # Load the interface data
    logging.info("Loading interface data...")
    data = pd.read_csv('results/extract/extracted_data.csv')
    # Remove duplicate rows
    data = data.drop_duplicates()
    # Log the number of rows after removing duplicates
    logging.info(f"Number of rows after removing duplicates: {len(data)}")
    
    # Check for missing values
    logging.info(f"Missing values in the dataset: {data.isnull().sum().sum()}")
    if data.isnull().any().any():
        logging.warning("Missing values found in the data. Filling with 0.")
        data = data.fillna(0)
    
    # Define target variables
    target_vars = ['IntSRHn', 'IntSRHp']
    
    # Display basic statistics of target variables
    logging.info("Statistics of target variables:")
    logging.info(f"\n{data[target_vars].describe()}")
    
    # Get feature columns (all columns except targets)
    feature_cols = feature_names
    
    # Split features and targets
    X = data[feature_cols]
    y = data[target_vars]
    
    # Log the X and y features
    logging.info(f"X features: {X.columns.tolist()}")
    logging.info(f"y features: {y.columns.tolist()}")
    
    # Log transform the target variables
    y_log = np.log10(np.abs(y))
    y_log_sign = np.sign(y)
    
    # Scale the features
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_log, test_size=0.2, random_state=42
    )
    
    # Save the prepared data
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save the original target values for inverse transformation
    np.save(os.path.join(output_dir, 'y_train_original.npy'), y.values[train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)[0]])
    np.save(os.path.join(output_dir, 'y_test_original.npy'), y.values[train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)[1]])
    
    # Save the scalers
    joblib.dump(X_scaler, os.path.join(output_dir, 'X_scaler.joblib'))
    
    logging.info("Data prepared and saved to:")
    logging.info(f"- {output_dir}/X_train.npy, {output_dir}/X_test.npy")
    logging.info(f"- {output_dir}/y_train.npy, {output_dir}/y_test.npy")
    logging.info("Scalers saved:")
    logging.info(f"- {output_dir}/X_scaler.joblib")
    
    return X_train, X_test, y_train, y_test, feature_cols

if __name__ == "__main__":
    prepare_data()
