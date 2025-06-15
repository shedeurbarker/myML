import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the interface data
print("Loading interface data...")
df = pd.read_csv('interface_data_padded.csv')

# Display basic information about the dataset
print("\nDataset shape:", df.shape)
print("\nColumns in the dataset:")
print(df.columns.tolist())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum().sum())

# Define target variables
target_vars = ['IntSRHn', 'IntSRHp']

# Display basic statistics of target variables
print("\nStatistics of target variables:")
print(df[target_vars].describe())

# Visualize the distribution of target variables
plt.figure(figsize=(12, 5))
for i, var in enumerate(target_vars):
    plt.subplot(1, 2, i+1)
    sns.histplot(df[var], kde=True)
    plt.title(f'Distribution of {var}')
plt.tight_layout()
plt.savefig('target_distributions.png')
print("\nSaved target variable distributions to 'target_distributions.png'")

# Identify potential feature columns (excluding target variables and other non-feature columns)
# You may need to adjust this based on your specific dataset
non_feature_cols = ['lid', 'x', 'y', 'z', 'IntSRHn', 'IntSRHp', 'BulkSRHn', 'BulkSRHp']
feature_cols = [col for col in df.columns if col not in non_feature_cols]

print(f"\nNumber of potential features: {len(feature_cols)}")
print("Sample of feature columns:", feature_cols[:10])

# Prepare data for ML
X = df[feature_cols]
y = df[target_vars]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData prepared for ML:")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Save the prepared data
np.save('X_train.npy', X_train_scaled)
np.save('X_test.npy', X_test_scaled)
np.save('y_train.npy', y_train.values)
np.save('y_test.npy', y_test.values)

print("\nPrepared data saved as numpy arrays:")
print("- X_train.npy, X_test.npy")
print("- y_train.npy, y_test.npy")

# Example of how to load the data for ML training
print("\nExample of loading the data for ML training:")
print("""
# Load the prepared data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Now you can use these arrays to train your ML model
# For example, with scikit-learn:
from sklearn.ensemble import RandomForestRegressor

# Train a model for IntSRHn
model_n = RandomForestRegressor(n_estimators=100, random_state=42)
model_n.fit(X_train, y_train[:, 0])  # First column is IntSRHn

# Train a model for IntSRHp
model_p = RandomForestRegressor(n_estimators=100, random_state=42)
model_p.fit(X_train, y_train[:, 1])  # Second column is IntSRHp

# Evaluate the models
score_n = model_n.score(X_test, y_test[:, 0])
score_p = model_p.score(X_test, y_test[:, 1])

print(f"R² score for IntSRHn: {score_n:.4f}")
print(f"R² score for IntSRHp: {score_p:.4f}")
""") 