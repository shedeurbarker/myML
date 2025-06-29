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
output_dir = 'results/prepared_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{output_dir}/prepared_data.log', mode='a'),
        logging.StreamHandler()
    ]
)

# Debug logging to confirm script execution
logging.debug("Script started")

# Load feature names from scripts/feature_names.npy
feature_names = np.load('scripts/feature_names.npy')
logging.info(f"Features used: {feature_names}")

def prepare_data():
    """Prepare data for machine learning with proper sign handling."""
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
    
    # FIXED: Use signed log transformation to preserve sign information
    # This approach: sign(y) * log10(|y| + epsilon)
    # This preserves the sign while allowing log transformation of magnitude
    epsilon = 1e-30  # Small number to avoid log(0)
    y_values = y.values
    
    # Apply signed log transformation
    y_signed_log = np.sign(y_values) * np.log10(np.abs(y_values) + epsilon)
    
    # Log the transformation results
    logging.info(f"Original y range: {y_values.min():.2e} to {y_values.max():.2e}")
    logging.info(f"Signed log y range: {y_signed_log.min():.2f} to {y_signed_log.max():.2f}")
    
    # Check sign preservation
    original_signs = np.sign(y_values)
    transformed_signs = np.sign(y_signed_log)
    sign_preserved = np.all(original_signs == transformed_signs)
    logging.info(f"Signs preserved: {sign_preserved}")
    
    # Scale the features
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_signed_log, test_size=0.2, random_state=42
    )
    
    # Save the prepared data
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save the original target values for inverse transformation
    train_indices, test_indices = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
    np.save(os.path.join(output_dir, 'y_train_original.npy'), y_values[train_indices])
    np.save(os.path.join(output_dir, 'y_test_original.npy'), y_values[test_indices])
    
    # Save the scalers
    joblib.dump(X_scaler, os.path.join(output_dir, 'X_scaler.joblib'))
    
    # Log the sign distribution in training data
    logging.info(f"Sign distribution in training data:")
    logging.info(f"IntSRHn negative: {(y_train[:, 0] < 0).sum()}, positive: {(y_train[:, 0] > 0).sum()}")
    logging.info(f"IntSRHp negative: {(y_train[:, 1] < 0).sum()}, positive: {(y_train[:, 1] > 0).sum()}")
    
    # Log some examples of the transformation
    logging.info(f"Sample transformations:")
    for i in range(min(5, len(y_values))):
        logging.info(f"Original: {y_values[i]}, Signed log: {y_signed_log[i]}")
    
    logging.info("Data prepared and saved to:")
    logging.info(f"- {output_dir}/X_train.npy, {output_dir}/X_test.npy")
    logging.info(f"- {output_dir}/y_train.npy, {output_dir}/y_test.npy")
    logging.info("Scalers saved:")
    logging.info(f"- {output_dir}/X_scaler.joblib")
    
    return X_train, X_test, y_train, y_test, feature_cols

if __name__ == "__main__":
    prepare_data()
