import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Set up logging
log_dir = 'results/predict'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'predictions.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def ensure_predict_results_dir():
    os.makedirs('results/predict', exist_ok=True)

def load_model_and_features(target_name, model_type):
    """Load the trained model and get the required feature names."""
    model_path = f'results/train_ml_models/models/{model_type}_{target_name}.joblib'
    model = joblib.load(model_path)
    logging.info(f"Loaded model from {model_path}")
    original_data = pd.read_csv('results/extract/interface_data.csv')
    non_feature_cols = ['lid', 'x', 'y', 'z', 'IntSRHn', 'IntSRHp', 'BulkSRHn', 'BulkSRHp']
    feature_names = [col for col in original_data.columns if col not in non_feature_cols]
    return model, feature_names

def prepare_input_data(input_data, feature_names):
    """Prepare input data for prediction."""
    # Ensure all required features are present
    missing_features = set(feature_names) - set(input_data.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select features in the correct order
    X = input_data[feature_names].values
    
    return X

def make_predictions_all_models(input_data, target_name='IntSRHn'):
    """Make predictions using all three models."""
    model_types = ['RandomForest', 'GradientBoosting', 'LinearRegression']
    predictions = {}
    for model_type in model_types:
        model, feature_names = load_model_and_features(target_name, model_type)
        X = prepare_input_data(input_data, feature_names)
        preds = model.predict(X)
        predictions[model_type] = preds
    # Add predictions to DataFrame
    result = input_data.copy()
    for model_type in model_types:
        result[f'predicted_{target_name}_{model_type}'] = predictions[model_type]
    return result

def validate_predictions_all_models():
    ensure_predict_results_dir()
    """Validate all models' predictions against test data and compare accuracy."""
    X_test = np.load('results/prepare_ml_data/X_test.npy')
    y_test = np.load('results/prepare_ml_data/y_test.npy')
    original_data = pd.read_csv('results/extract/interface_data.csv')
    non_feature_cols = ['lid', 'x', 'y', 'z', 'IntSRHn', 'IntSRHp', 'BulkSRHn', 'BulkSRHp']
    feature_names = [col for col in original_data.columns if col not in non_feature_cols]
    test_data = pd.DataFrame(X_test, columns=feature_names)
    model_types = ['RandomForest', 'GradientBoosting', 'LinearRegression']
    target_names = ['IntSRHn', 'IntSRHp']
    all_metrics = {t: {} for t in target_names}
    
    # Create a list to store all metrics for CSV
    metrics_rows = []
    
    for target_idx, target_name in enumerate(target_names):
        actual = y_test[:, target_idx]
        for model_type in model_types:
            model, feature_names = load_model_and_features(target_name, model_type)
            X = prepare_input_data(test_data, feature_names)
            predicted = model.predict(X)
            
            # Calculate basic metrics
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mae = mean_absolute_error(actual, predicted)
            r2 = r2_score(actual, predicted)
            
            # Calculate accuracy using a more robust method
            # Handle zero values and very small numbers
            epsilon = 1e-30  # Small number to avoid division by zero
            actual_abs = np.abs(actual)
            predicted_abs = np.abs(predicted)
            
            # Calculate relative error
            relative_error = np.abs(predicted_abs - actual_abs) / (actual_abs + epsilon)
            
            # Convert to accuracy percentage (100% - relative error)
            accuracies = np.maximum(0, 100 * (1 - relative_error))
            
            # Handle special cases
            # If both predicted and actual are very small (close to zero), consider it accurate
            both_small = (actual_abs < epsilon) & (predicted_abs < epsilon)
            accuracies[both_small] = 100.0
            
            # If actual is very small but predicted is not (or vice versa), consider it inaccurate
            one_small = (actual_abs < epsilon) | (predicted_abs < epsilon)
            accuracies[one_small & ~both_small] = 0.0
            
            # Calculate statistics
            mean_accuracy = np.mean(accuracies)
            median_accuracy = np.median(accuracies)
            accuracy_std = np.std(accuracies)
            within_90 = np.sum(accuracies >= 90) / len(accuracies) * 100
            within_80 = np.sum(accuracies >= 80) / len(accuracies) * 100
            within_70 = np.sum(accuracies >= 70) / len(accuracies) * 100
            
            metrics = {
                'Target': target_name,
                'Model': model_type,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'Mean_Accuracy': mean_accuracy,
                'Median_Accuracy': median_accuracy,
                'Accuracy_Std': accuracy_std,
                'Within_90%': within_90,
                'Within_80%': within_80,
                'Within_70%': within_70
            }
            metrics_rows.append(metrics)
            all_metrics[target_name][model_type] = metrics
            
            logging.info(f"\nValidation metrics for {target_name} ({model_type}):")
            logging.info(f"RMSE: {rmse:.4e}")
            logging.info(f"MAE: {mae:.4e}")
            logging.info(f"R²: {r2:.4f}")
            logging.info(f"Mean Accuracy: {mean_accuracy:.2f}%")
            logging.info(f"Median Accuracy: {median_accuracy:.2f}%")
            logging.info(f"Accuracy Std: {accuracy_std:.2f}%")
            logging.info(f"Predictions within 90% accuracy: {within_90:.2f}%")
            logging.info(f"Predictions within 80% accuracy: {within_80:.2f}%")
            logging.info(f"Predictions within 70% accuracy: {within_70:.2f}%")
        
        # Summarize best model
        best_model = max(all_metrics[target_name], key=lambda m: all_metrics[target_name][m]['Mean_Accuracy'])
        best_acc = all_metrics[target_name][best_model]['Mean_Accuracy']
        logging.info(f"\nBest model for {target_name}: {best_model} (Mean Accuracy: {best_acc:.2f}%)\n")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv('results/predict/model_validation_metrics.csv', index=False)
    logging.info("Validation metrics saved to results/predict/model_validation_metrics.csv")
    
    # Print summary table
    for target_name in target_names:
        logging.info(f"\nSummary for {target_name}:")
        df = pd.DataFrame(all_metrics[target_name]).T
        logging.info(f"\n{df}")
    
    return all_metrics

def example_usage():
    # Example input data using values from interface_data.csv
    example_data = pd.DataFrame({
        'V': [0.345623],
        'Evac': [0.254377],
        'Ec': [-4.14562],
        'Ev': [-5.44562],
        'phin': [-5.00075],
        'phip': [-5.05961],
        'n': [5.41653e+09],
        'p': [5.59667e+17],
        'ND': [1e+20],
        'NA': [1e+21],
        'anion': [1.9456e+22],
        'cation': [1.63177e+14],
        'ntb': [3.18671e+11],
        'nti': [0],
        'mun': [0.0001],
        'mup': [0.0001],
        'G_ehp': [1.54171e+28],
        'Gfree': [1.54171e+28],
        'Rdir': [3.04141e+10],
        'Jn': [3.71792e+19],
        'Jp': [-202.215],
        'Jint': [-0.0945037],
        'Vext': [-202.31],
        'left_L': [2.5e-08],
        'left_eps_r': [5],
        'left_E_c': [4],
        'left_E_v': [5.9],
        'left_N_c': [5e+26],
        'left_mu_n': [1e-06],
        'left_mu_p': [1e-09],
        'left_N_t_int': [4e+12],
        'left_C_n_int': [2e-14],
        'left_C_p_int': [2e-14],
        'left_E_t_int': [4.7],
        'right_L': [5e-07],
        'right_eps_r': [24],
        'right_E_c': [3.9],
        'right_E_v': [5.53],
        'right_N_c': [2.2e+24],
        'right_mu_n': [0.0001],
        'right_mu_p': [0.0001],
        'right_N_t_int': [1e+12],
        'right_C_n_int': [2e-14],
        'right_C_p_int': [2e-14],
        'right_E_t_int': [4.7]
    })
    # Make predictions for all models and both targets
    for target in ['IntSRHn', 'IntSRHp']:
        result = make_predictions_all_models(example_data, target)
        logging.info(f"\nPredictions for {target} (all models):")
        logging.info(result[[col for col in result.columns if col.startswith('predicted_')]])
    # Validate all models
    logging.info("\nValidating all models against test data...")
    validate_predictions_all_models()

if __name__ == "__main__":
    example_usage() 