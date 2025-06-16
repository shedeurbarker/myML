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
    
    # Create a DataFrame with the correct feature names
    X_df = pd.DataFrame(X, columns=feature_names)
    
    return X_df

def make_predictions_all_models(input_data, include_targets=False):
    """Make predictions using all trained models."""
    # Load the scalers
    X_scaler = joblib.load('results/prepare_ml_data/X_scaler.joblib')
    
    # Scale the input features
    X_scaled = X_scaler.transform(input_data)
    print("\nDEBUG: Input data shape:", input_data.shape)
    print("DEBUG: Scaled input data shape:", X_scaled.shape)
    
    # If targets are included, print their log-transformed values
    if include_targets and all(col in input_data.columns for col in ['IntSRHn', 'IntSRHp']):
        target_data = input_data[['IntSRHn', 'IntSRHp']].values
        target_log = np.log10(np.abs(target_data))
        print("\nDEBUG: Original target values:", target_data[0])
        print("DEBUG: Log-transformed target values:", target_log[0])
    
    predictions = {}
    
    # Process each model
    for model_name in ['RandomForest', 'GradientBoosting', 'LinearRegression']:
        for target in ['IntSRHn', 'IntSRHp']:
            model_path = f'results/train_ml_models/models/{model_name}_{target}.joblib'
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                logging.info(f"Loaded model from {model_path}")
                
                # Make prediction
                pred_log = model.predict(X_scaled)
                print(f"\nDEBUG: Model {model_name} prediction in log space:", pred_log)
                
                # Inverse transform the prediction
                pred = np.power(10, pred_log)
                print(f"DEBUG: Unscaled prediction for {target}:", pred[0])
                
                predictions[f'predicted_{target}_{model_name}'] = pred[0]
    
    return pd.DataFrame([predictions])

def validate_predictions_all_models():
    ensure_predict_results_dir()
    """Validate all models' predictions against test data and compare accuracy."""
    X_test = np.load('results/prepare_ml_data/X_test.npy')
    y_test = np.load('results/prepare_ml_data/y_test.npy')
    feature_names = np.load('scripts/feature_names.npy')
    test_data = pd.DataFrame(X_test, columns=feature_names)
    model_types = ['RandomForest', 'GradientBoosting', 'LinearRegression']
    target_names = ['IntSRHn', 'IntSRHp']
    all_metrics = {t: {} for t in target_names}
    
    # Create a list to store all metrics for CSV
    metrics_rows = []
    
    for target_idx, target_name in enumerate(target_names):
        actual = y_test[:, target_idx]
        for model_type in model_types:
            model, _ = load_model_and_features(target_name, model_type)
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

def write_predictions_to_file(all_predictions, experimental_values, filename='results/predict/predicted_values.txt'):
    """Write predictions, errors, and percent accuracy for each model and target to a file."""
    with open(filename, 'w') as f:
        f.write("Predicted Interface Recombination Values\n")
        f.write("=====================================\n\n")
        for target in ['IntSRHn', 'IntSRHp']:
            f.write(f"\n{target} Predictions:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Experimental Value: {experimental_values[target]:.6E}\n")
            for model in ['RandomForest', 'GradientBoosting', 'LinearRegression']:
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
                f.write(f"Accuracy: {accuracy:.2f}%\n\n")

def plot_predictions(predictions, experimental_values, save_dir='results/predict'):
    """Create and save plots of the predictions."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot IntSRHn predictions
    models = list(predictions['IntSRHn'].keys())
    # Set negative values to zero
    values_n = [max(0, predictions['IntSRHn'][model]) for model in models]
    exp_n = max(0, experimental_values['IntSRHn'])
    
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
    # Set negative values to zero
    values_p = [max(0, predictions['IntSRHp'][model]) for model in models]
    exp_p = max(0, experimental_values['IntSRHp'])
    
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
    # Example data for prediction
    example_data = pd.DataFrame({
        'p': [1.297e+18],
        'n': [9.18056e+23],
        'cation': [1.41364e+19],
        'anion': [2.37984e+22],
        'phip': [-4.1167],
        'phin': [-3.30357],
        'mun': [0.0001],
        'mup': [0.0001],
        'Ec': [-3.28135],
        'Ev': [-4.48135],
        'Jp': [3.27136],
        'Jn': [-18.3825],
        'Gfree': [1.27696e+28],
        'G_ehp': [1.27696e+28],
        'Jint': [-15.1111],
        'Vext': [-0.05],
        'Evac': [1.21865],
        'ntb': [9.99081e+20],
        'V': [-0.61865],
        'Rdir': [1.19422e+25],
        'NA': [3.16e+20],
        'left_mu_n': [1e-06],
        'right_mu_n': [0.0001],
        'left_eps_r': [5],
        'right_eps_r': [24],
        'left_mu_p': [1e-09],
        'right_mu_p': [0.0001],
        'left_L': [2.5e-08],
        'right_L': [5e-07],
        'left_E_c': [4],
        'right_E_c': [3.9],
        'left_E_v': [5.9],
        'right_E_v': [5.53],
        'left_N_t_int': [4e+12],
        'right_N_t_int': [1e+12],
        'left_N_c': [5e+26],
        'right_N_c': [2.2e+24]
    })
    
    # Dictionary to store all predictions
    all_predictions = {}
    
    # Make predictions for all models and both targets
    for target in ['IntSRHn', 'IntSRHp']:
        result = make_predictions_all_models(example_data, include_targets=True)
        logging.info(f"\nPredictions for {target} (all models):")
        logging.info(result[[col for col in result.columns if col.startswith('predicted_')]])
        
        # Store predictions for this target
        all_predictions[target] = {}
        for model_type in ['RandomForest', 'GradientBoosting', 'LinearRegression']:
            pred_col = f'predicted_{target}_{model_type}'
            all_predictions[target][model_type] = result[pred_col].iloc[0]
    
    # Define experimental values for comparison
    experimental_values = {
        'IntSRHn': 2.56097E+29,  # experimental value
        'IntSRHp': 1.24002E+27   # experimental value
    }
    
    # Write predictions to file
    write_predictions_to_file(all_predictions, experimental_values)
    
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
    
    # # Add accuracy percentages to the file
    # with open(f'{log_dir}/predicted_values.txt', 'a') as f:
    #     f.write(f"\nAccuracy for IntSRHn: {accuracy_IntSRHn:.2f}%\n")
    #     f.write(f"Accuracy for IntSRHp: {accuracy_IntSRHp:.2f}%\n")
    #     f.write(f"Model: RandomForest\n")
    #     f.write(f"Predicted IntSRHn: {all_predictions['IntSRHn']['RandomForest']:.6E}\n")
    #     f.write(f"Error: {abs(all_predictions['IntSRHn']['RandomForest'] - experimental_values['IntSRHn']):.6E}\n")
    #     f.write(f"Accuracy: {accuracy_IntSRHn:.2f}%\n")
    #     f.write(f"Model: GradientBoosting\n")
    #     f.write(f"Predicted IntSRHn: {all_predictions['IntSRHn']['GradientBoosting']:.6E}\n")
    #     f.write(f"Error: {abs(all_predictions['IntSRHn']['GradientBoosting'] - experimental_values['IntSRHn']):.6E}\n")
    #     f.write(f"Accuracy: {accuracy_IntSRHn:.2f}%\n")
    #     f.write(f"Model: LinearRegression\n")
    #     f.write(f"Predicted IntSRHn: {all_predictions['IntSRHn']['LinearRegression']:.6E}\n")
    #     f.write(f"Error: {abs(all_predictions['IntSRHn']['LinearRegression'] - experimental_values['IntSRHn']):.6E}\n")
    #     f.write(f"Accuracy: {accuracy_IntSRHn:.2f}%\n")

if __name__ == "__main__":
    example_usage() 