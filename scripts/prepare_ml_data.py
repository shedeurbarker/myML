import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Create output directories
output_dir = 'results/prepare_ml_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logging.info(f"Created directory: {output_dir}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/prepare_ml_data/prepare_ml_data.log'),
        logging.StreamHandler()
    ]
)

# Load the interface data
logging.info("Loading interface data...")
df = pd.read_csv('results/extract/interface_data.csv')

# Display basic information about the dataset
logging.info(f"Dataset shape: {df.shape}")
logging.info(f"Columns in the dataset: {df.columns.tolist()}")

# Check for missing values
logging.info(f"Missing values in the dataset: {df.isnull().sum().sum()}")

# Define target variables
target_vars = ['IntSRHn', 'IntSRHp']

# Display basic statistics of target variables
logging.info("Statistics of target variables:")
logging.info(f"\n{df[target_vars].describe()}")

# Identify potential feature columns (excluding target variables and other non-feature columns)
non_feature_cols = ['lid', 'x', 'y', 'z', 'IntSRHn', 'IntSRHp', 'BulkSRHn', 'BulkSRHp']
feature_cols = [col for col in df.columns if col not in non_feature_cols]

logging.info(f"Number of potential features: {len(feature_cols)}")
logging.info(f"Sample of feature columns: {feature_cols[:10]}")

# Prepare data for ML
X = df[feature_cols]
y = df[target_vars]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logging.info("Data prepared for ML:")
logging.info(f"Training set shape: {X_train.shape}")
logging.info(f"Testing set shape: {X_test.shape}")

# Save the prepared data
np.save(os.path.join(output_dir, 'X_train.npy'), X_train_scaled)
np.save(os.path.join(output_dir, 'X_test.npy'), X_test_scaled)
np.save(os.path.join(output_dir, 'y_train.npy'), y_train.values)
np.save(os.path.join(output_dir, 'y_test.npy'), y_test.values)

logging.info("Prepared data saved as numpy arrays:")
logging.info(f"- {output_dir}/X_train.npy, {output_dir}/X_test.npy")
logging.info(f"- {output_dir}/y_train.npy, {output_dir}/y_test.npy")
