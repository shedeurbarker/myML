import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from create_example_data import create_example_data
from ml_models import ML_MODEL_NAMES

# Set up logging
log_dir = 'results/predict'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'predictions.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
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
    original_data = pd.read_csv('results/extract/extracted_data.csv')
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
    
    # Create a DataFrame with the correct feature names
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Load and apply the same scaler used during training
    scaler = joblib.load('results/prepared_data/X_scaler.joblib')
    X_scaled = scaler.transform(X_df)
    
    logging.debug(f"Input data shape: {X_df.shape}")
    logging.debug(f"Scaled input data shape: {X_scaled.shape}")
    
    return X_scaled

def reconstruct_original_values(predictions_signed_log):
    """Reconstruct original values from signed log predictions."""
    # The model predicts: sign(y) * log10(|y| + epsilon)
    # To reconstruct: sign(pred) * 10^|pred|
    epsilon = 1e-30  # Same epsilon used in training
    
    # Extract sign and magnitude
    signs = np.sign(predictions_signed_log)
    magnitudes = np.abs(predictions_signed_log)
    
    # Reconstruct original values
    predictions_original = signs * np.power(10, magnitudes)
    
    return predictions_original

def make_predictions_all_models(input_data, include_targets=False):
    """Make predictions using all trained models with proper sign handling."""
    ensure_predict_results_dir()
    
    # Load feature names
    feature_names = np.load('scripts/feature_names.npy')
    
    # Prepare input data
    X_scaled = prepare_input_data(input_data, feature_names)
    
    # Dictionary to store predictions
    predictions = {}
    
    # Make predictions for each target and model
    for target_name in ['IntSRHn', 'IntSRHp']:
        for model_type in ML_MODEL_NAMES:
            try:
                model, _ = load_model_and_features(target_name, model_type)
                
                # Make prediction in signed log space
                prediction_signed_log = model.predict(X_scaled)
                
                logging.debug(f"Model {model_type} prediction in signed log space: {prediction_signed_log}")
                
                # Reconstruct original values with proper sign
                prediction_original = reconstruct_original_values(prediction_signed_log)
                
                logging.debug(f"Reconstructed prediction for {target_name}: {prediction_original}")
                
                predictions[f'predicted_{target_name}_{model_type}'] = prediction_original
                
            except Exception as e:
                logging.error(f"Error making prediction for {target_name} with {model_type}: {e}")
                predictions[f'predicted_{target_name}_{model_type}'] = np.nan
    
    # Create result DataFrame
    result_df = pd.DataFrame(predictions)
    
    if include_targets:
        # Add input features to the result
        for i, feature in enumerate(feature_names):
            if isinstance(input_data, pd.DataFrame):
                result_df[feature] = input_data[feature].values
            else:
                result_df[feature] = input_data[:, i]
    
    return result_df

def validate_predictions_all_models():
    """Validate all models against test data with proper sign handling."""
    # Load test data
    X_test = np.load('results/prepared_data/X_test.npy')
    y_test = np.load('results/prepared_data/y_test.npy')  # This is now signed log values
    y_test_original = np.load('results/prepared_data/y_test_original.npy')
    
    # Load feature names for input preparation
    feature_names = np.load('scripts/feature_names.npy')
    
    # Create a DataFrame for easier feature access
    test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Dictionary to store all metrics
    all_metrics = {'IntSRHn': {}, 'IntSRHp': {}}
    metrics_rows = []
    
    for target_name in ['IntSRHn', 'IntSRHp']:
        for model_type in ML_MODEL_NAMES:
            try:
                model, _ = load_model_and_features(target_name, model_type)
                
                # Make predictions in signed log space
                predictions_signed_log = model.predict(X_test)
                
                # Reconstruct original predictions
                predictions_original = reconstruct_original_values(predictions_signed_log)
                
                # Get actual values
                actual = y_test_original[:, 0] if target_name == 'IntSRHn' else y_test_original[:, 1]
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(actual, predictions_original))
                mae = mean_absolute_error(actual, predictions_original)
                r2 = r2_score(actual, predictions_original)
                
                # Calculate accuracy
                epsilon = 1e-30
                actual_abs = np.abs(actual)
                predicted_abs = np.abs(predictions_original)
                relative_error = np.abs(predicted_abs - actual_abs) / (actual_abs + epsilon)
                accuracies = np.maximum(0, 100 * (1 - relative_error))
                
                # Handle special cases
                both_small = (actual_abs < epsilon) & (predicted_abs < epsilon)
                accuracies[both_small] = 100.0
                
                one_small = (actual_abs < epsilon) | (predicted_abs < epsilon)
                accuracies[one_small & ~both_small] = 0.0
                
                # Calculate statistics
                mean_accuracy = np.mean(accuracies)
                median_accuracy = np.median(accuracies)
                accuracy_std = np.std(accuracies)
                within_90 = np.sum(accuracies >= 90) / len(accuracies) * 100
                within_80 = np.sum(accuracies >= 80) / len(accuracies) * 100
                within_70 = np.sum(accuracies >= 70) / len(accuracies) * 100
                
                # Calculate sign accuracy
                actual_signs = np.sign(actual)
                predicted_signs = np.sign(predictions_original)
                sign_accuracy = np.mean(actual_signs == predicted_signs) * 100
                
                metrics = {
                    'Target': target_name,
                    'Model': model_type,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R²': r2,
                    'Mean_Accuracy': mean_accuracy,
                    'Median_Accuracy': median_accuracy,
                    'Accuracy_Std': accuracy_std,
                    'Sign_Accuracy': sign_accuracy,
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
                logging.info(f"Sign Accuracy: {sign_accuracy:.2f}%")
                logging.info(f"Predictions within 90% accuracy: {within_90:.2f}%")
                logging.info(f"Predictions within 80% accuracy: {within_80:.2f}%")
                logging.info(f"Predictions within 70% accuracy: {within_70:.2f}%")
                
            except Exception as e:
                logging.error(f"Error validating {target_name} with {model_type}: {e}")
        
        # Summarize best model
        if all_metrics[target_name]:
            best_model = max(all_metrics[target_name], key=lambda m: all_metrics[target_name][m]['Mean_Accuracy'])
            best_acc = all_metrics[target_name][best_model]['Mean_Accuracy']
            logging.info(f"\nBest model for {target_name}: {best_model} (Mean Accuracy: {best_acc:.2f}%)\n")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv('results/predict/model_validation_metrics.csv', index=False)
    logging.info("Validation metrics saved to results/predict/model_validation_metrics.csv")
    
    # Print summary table
    for target_name in ['IntSRHn', 'IntSRHp']:
        logging.info(f"\nSummary for {target_name}:")
        df = pd.DataFrame(all_metrics[target_name]).T
        logging.info(f"\n{df}")
    
    return all_metrics

def write_predictions_to_file(feature_names, all_predictions, experimental_values, filename='results/predict/predicted_values.txt'):
    """Write predictions, errors, and percent accuracy for each model and target to a file."""
    with open(filename, 'w') as f:
        f.write("Predicted Interface Recombination Values\n")
        f.write("=====================================\n\n")
        f.write("Features\n\n")
        f.write(f'{feature_names}\n')
        f.write("=====================================\n\n")
        for target in ['IntSRHn', 'IntSRHp']:
            f.write(f"\n{target} Predictions:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Experimental Value: {experimental_values[target]:.6E}\n")
            for model in ML_MODEL_NAMES:
                pred = all_predictions[target][model]
                error = abs(pred - experimental_values[target])
                # Avoid division by zero
                if abs(experimental_values[target]) > 1e-30:
                    accuracy = (1 - error / abs(experimental_values[target])) * 100
                else:
                    accuracy = 0.0
                f.write(f"Model: {model}\n")
                f.write(f"Predicted: {pred:.6E}\n")
                f.write(f"Error: {error:.6E}\n")
                f.write(f"Accuracy: {accuracy:.2f}%\n")
                f.write(f"Sign Match: {'Yes' if np.sign(pred) == np.sign(experimental_values[target]) else 'No'}\n\n")

def plot_predictions(predictions, experimental_values, save_dir='results/predict'):
    """Create and save plots of the predictions."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot IntSRHn predictions
    models = list(predictions['IntSRHn'].keys())
    values_n = [predictions['IntSRHn'][model] for model in models]
    exp_n = experimental_values['IntSRHn']
    
    bars_n = ax1.bar(models, values_n, color='blue', alpha=0.7)
    # Add experimental value as a horizontal line
    ax1.axhline(y=exp_n, color='red', linestyle='--', label='Experimental')
    ax1.set_title('IntSRHn Predictions')
    ax1.set_ylabel('Value')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(values_n):
        ax1.text(i, v, f'{v:.2E}', ha='center', va='bottom')
    
    # Plot IntSRHp predictions
    values_p = [predictions['IntSRHp'][model] for model in models]
    exp_p = experimental_values['IntSRHp']
    
    bars_p = ax2.bar(models, values_p, color='blue', alpha=0.7)
    # Add experimental value as a horizontal line
    ax2.axhline(y=exp_p, color='red', linestyle='--', label='Experimental')
    ax2.set_title('IntSRHp Predictions')
    ax2.set_ylabel('Value')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(values_p):
        ax2.text(i, v, f'{v:.2E}', ha='center', va='bottom')
    
    # Add legend
    ax1.legend()
    ax2.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_comparison.png'))
    plt.close()
    
    logging.info(f"Prediction plots saved to {save_dir}/prediction_comparison.png")

def example_usage():
    # Get the example data and experimental values
    example_data, experimental_values = create_example_data()
    feature_names = example_data.columns.tolist()
    
    # Dictionary to store all predictions
    all_predictions = {}
    
    # Make predictions for all models and both targets
    for target in ['IntSRHn', 'IntSRHp']:
        result = make_predictions_all_models(example_data, include_targets=True)
        logging.info(f"\nPredictions for {target} (all models):")
        logging.info(result[[col for col in result.columns if col.startswith('predicted_')]])
        
        # Store predictions for this target
        all_predictions[target] = {}
        for model_type in ML_MODEL_NAMES:
            pred_col = f'predicted_{target}_{model_type}'
            all_predictions[target][model_type] = result[pred_col].iloc[0]
    
    # Write predictions to file
    write_predictions_to_file(feature_names, all_predictions, experimental_values)
    
    # Create and save plots
    plot_predictions(all_predictions, experimental_values)
    
    # Validate all models
    logging.info("\nValidating all models against test data...")
    validate_predictions_all_models()

    # Calculate accuracy for IntSRHn
    accuracy_IntSRHn = (1 - abs(all_predictions['IntSRHn']['RandomForest'] - experimental_values['IntSRHn']) / experimental_values['IntSRHn']) * 100
    logging.info(f"Accuracy for IntSRHn: {accuracy_IntSRHn:.2f}%")

    # Calculate accuracy for IntSRHp
    accuracy_IntSRHp = (1 - abs(all_predictions['IntSRHp']['RandomForest'] - experimental_values['IntSRHp']) / experimental_values['IntSRHp']) * 100
    logging.info(f"Accuracy for IntSRHp: {accuracy_IntSRHp:.2f}%")
    
if __name__ == "__main__":
    example_usage() 