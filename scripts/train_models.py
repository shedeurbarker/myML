import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import joblib
import os
import logging
from datetime import datetime


# Set up logging
log_dir = 'results/train_ml_models'
scatter_plot = 'results/train_ml_models/scatter_plot'
residual_plot = 'results/train_ml_models/residual_plot'
importance_plot = 'results/train_ml_models/importance_plot'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(scatter_plot, exist_ok=True)
os.makedirs(residual_plot, exist_ok=True)
os.makedirs(importance_plot, exist_ok=True)
log_file = os.path.join(log_dir, f'ml_training.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Create a directory for saving models and results
os.makedirs('results/train_ml_models/models', exist_ok=True)

# Load the prepared data
logging.info("Loading prepared data...")
X_train = np.load('results/prepare_ml_data/X_train.npy')
X_test = np.load('results/prepare_ml_data/X_test.npy')
y_train = np.load('results/prepare_ml_data/y_train.npy')
y_test = np.load('results/prepare_ml_data/y_test.npy')

# Load feature names from scripts/feature_names.npy
feature_names = np.load('scripts/feature_names.npy')
logging.info(f"Loaded feature names from scripts/feature_names.npy")

# Verify the number of features matches the training data
logging.info(f"Number of features: {len(feature_names)}")
logging.info(f"Number of features in training data: {X_train.shape[1]}")
if len(feature_names) != X_train.shape[1]:
    logging.error(f"Feature count mismatch! Expected {X_train.shape[1]} features but got {len(feature_names)}")
    logging.error("Feature names:")
    for name in feature_names:
        logging.error(f"- {name}")
    raise ValueError("Feature count mismatch between training data and feature names")

logging.info("Feature names:")
for name in feature_names:
    logging.info(f"- {name}")

# Define target variable names
target_names = ['IntSRHn', 'IntSRHp']

# After loading y_train and y_test
logging.info(f"First 5 rows of y_train (log space):\n{y_train[:5]}")
logging.info(f"First 5 rows of y_test (log space):\n{y_test[:5]}")

# Use feature_names in the script
X = X_train

# Function to evaluate and plot model performance
def evaluate_model(y_true, y_pred, target_name, model_name):
    """Evaluate model performance and create visualizations."""
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    logging.info(f"\n{model_name} - {target_name} Performance:")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"R²: {r2:.4f}")
    
    # Create scatter plot of true vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel(f'True {target_name}')
    plt.ylabel(f'Predicted {target_name}')
    plt.title(f'{model_name} - {target_name}: True vs Predicted')
    plt.tight_layout()
    plt.savefig(f'{scatter_plot}/{model_name}_{target_name}_scatter.png')
    
    # Create residual plot
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel(f'Predicted {target_name}')
    plt.ylabel('Residuals')
    plt.title(f'{model_name} - {target_name}: Residual Plot')
    plt.tight_layout()
    plt.savefig(f'{residual_plot}/{model_name}_{target_name}_residuals.png')
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

# Function to train and evaluate a model
def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, target_idx):
    """Train a model and evaluate its performance."""
    # Train the model
    model.fit(X_train, y_train[:, target_idx])
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Debug: Show first 5 predictions in log space and original space
    logging.info(f"First 5 predictions for {model_name} ({target_names[target_idx]}) in log space: {y_pred[:5]}")
    logging.info(f"First 5 predictions for {model_name} ({target_names[target_idx]}) in original space: {np.power(10, y_pred[:5])}")
    
    # Evaluate the model
    metrics = evaluate_model(y_test[:, target_idx], y_pred, target_names[target_idx], model_name)
    
    # Save the model
    model_path = f'results/train_ml_models/models/{model_name}_{target_names[target_idx]}.joblib'
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")
    
    return model, metrics

# Define models to train
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'LinearRegression': LinearRegression()
}

# Train and evaluate models for each target variable
results = {}

for target_idx, target_name in enumerate(target_names):
    logging.info(f"\n{'='*50}")
    logging.info(f"Training models for {target_name}")
    logging.info(f"{'='*50}")
    
    target_results = {}
    
    for model_name, model in models.items():
        logging.info(f"\nTraining {model_name} for {target_name}...")
        trained_model, metrics = train_and_evaluate_model(
            model, model_name, X_train, X_test, y_train, y_test, target_idx
        )
        target_results[model_name] = metrics
    
    results[target_name] = target_results

# Compare model performance
logging.info("\n" + "="*50)
logging.info("Model Performance Comparison")
logging.info("="*50)

for target_name in target_names:
    logging.info(f"\n{target_name} Performance:")
    logging.info("-"*30)
    
    # Create a DataFrame for comparison
    comparison = pd.DataFrame(results[target_name]).T
    logging.info(f"\n{comparison}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    comparison[['rmse', 'mae']].plot(kind='bar')
    plt.title(f'{target_name}: Model Comparison (RMSE and MAE)')
    plt.ylabel('Error')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'results/train_ml_models/{target_name}_model_comparison.png')
    logging.info(f"Comparison plot saved to results/train_ml_models/{target_name}_model_comparison.png")

# Feature importance for Random Forest models
for target_idx, target_name in enumerate(target_names):
    rf_model = joblib.load(f'results/train_ml_models/models/RandomForest_{target_name}.joblib')
    
    # Get feature importance
    importance = rf_model.feature_importances_
    
    # Verify lengths match before creating DataFrame
    if len(feature_names) != len(importance):
        logging.error(f"Feature importance length mismatch for {target_name}!")
        logging.error(f"Feature names length: {len(feature_names)}")
        logging.error(f"Importance array length: {len(importance)}")
        continue
    
    # Create a DataFrame for feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Plot top 40 features
    plt.figure(figsize=(15, 10))  # Increased figure size to accommodate more features
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(25))
    plt.title(f'Top 20 Features for {target_name} (Random Forest)')
    plt.tight_layout()
    plt.savefig(f'{importance_plot}/{target_name}_feature_importance.png')
    logging.info(f"Feature importance plot saved to {importance_plot}/{target_name}_feature_importance.png")
    
    # Save feature importance to CSV
    csv_path = f'{importance_plot}/{target_name}_feature_importance.csv'
    feature_importance.to_csv(csv_path, index=False)
    logging.info(f"Feature importance data saved to {csv_path}")

    # Log the contents of the feature_importance DataFrame
    logging.info("Feature Importance DataFrame:")
    logging.info(feature_importance)

    # Log top 25 important features for IntSRHn
    logging.info("Top 25 important features for IntSRHn:")
    for feature, importance in zip(feature_names, rf_model.feature_importances_):
        logging.info(f"{feature}: {importance}")

    # Log top 25 important features for IntSRHp
    logging.info("Top 25 important features for IntSRHp:")
    for feature, importance in zip(feature_names, rf_model.feature_importances_):
        logging.info(f"{feature}: {importance}")

logging.info("\nTraining and evaluation complete!")
logging.info("Models saved in the 'models' directory")
logging.info("Results and visualizations saved in the 'results/train_ml_models' directory")
logging.info(f"Log file saved to {log_file}") 