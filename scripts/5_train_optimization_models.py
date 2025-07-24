import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
from datetime import datetime
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_models import ML_MODELS, ML_MODEL_NAMES

# Set up logging
log_dir = 'results/train_optimization_models'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'optimization_training.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Create directories for saving models and results
os.makedirs('results/train_optimization_models/models', exist_ok=True)
os.makedirs('results/train_optimization_models/plots', exist_ok=True)

def load_enhanced_data():
    """Load the enhanced data with efficiency and recombination metrics."""
    data_path = 'results/generate_enhanced/combined_output_with_efficiency.csv'
    
    if not os.path.exists(data_path):
        logging.error(f"Enhanced data not found at {data_path}")
        logging.error("Please run scripts/2_generate_simulations_enhanced.py first")
        raise FileNotFoundError(f"Enhanced data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logging.info(f"Loaded enhanced data: {df.shape}")
    logging.info(f"Columns: {list(df.columns)}")
    
    return df

def prepare_optimization_data(df):
    """Prepare data for optimization models."""
    logging.info("Preparing optimization data...")
    
    # Define device parameters (features)
    device_params = [col for col in df.columns if col.startswith('L') and '_' in col]
    logging.info(f"Device parameters: {device_params}")
    
    # Define efficiency metrics (targets)
    efficiency_metrics = ['MPP', 'Jsc', 'Voc', 'FF']
    available_efficiency = [col for col in efficiency_metrics if col in df.columns]
    logging.info(f"Available efficiency metrics: {available_efficiency}")
    
    # Define recombination metrics (targets)
    recombination_metrics = ['IntSRHn_mean', 'IntSRHn_std', 'IntSRHn_min', 'IntSRHn_max']
    available_recombination = [col for col in recombination_metrics if col in df.columns]
    logging.info(f"Available recombination metrics: {available_recombination}")
    
    # Remove rows with missing data
    required_cols = device_params + available_efficiency + available_recombination
    df_clean = df[required_cols].dropna()
    logging.info(f"Clean data shape: {df_clean.shape}")
    
    # Split features and targets
    X = df_clean[device_params]
    y_efficiency = df_clean[available_efficiency]
    y_recombination = df_clean[available_recombination]
    
    logging.info(f"Features (X): {X.shape}")
    logging.info(f"Efficiency targets: {y_efficiency.shape}")
    logging.info(f"Recombination targets: {y_recombination.shape}")
    
    return X, y_efficiency, y_recombination, device_params

def train_efficiency_predictor(X, y_efficiency, device_params):
    """Train model to predict efficiency from device parameters."""
    logging.info("\n=== Training Efficiency Predictor ===")
    
    # Handle small datasets
    if len(X) < 5:
        logging.warning(f"Very small dataset ({len(X)} samples) - using all data for training")
        X_train, X_test = X, X
        y_train, y_test = y_efficiency, y_efficiency
    else:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_efficiency, test_size=0.2, random_state=42
        )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models for each efficiency metric
    efficiency_models = {}
    efficiency_scalers = {}
    
    for target in y_efficiency.columns:
        logging.info(f"\nTraining model for {target}...")
        
        # Train multiple models
        models = {}
        for model_name in ML_MODEL_NAMES:
            model = ML_MODELS[model_name]
            model.fit(X_train_scaled, y_train[target])
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test[target], y_pred)
            rmse = np.sqrt(mean_squared_error(y_test[target], y_pred))
            
            logging.info(f"{model_name} - {target}: R²={r2:.4f}, RMSE={rmse:.4f}")
            models[model_name] = model
        
        # Save best model (highest R²)
        best_model_name = max(models.keys(), key=lambda m: r2_score(y_test[target], models[m].predict(X_test_scaled)))
        efficiency_models[target] = models[best_model_name]
        efficiency_scalers[target] = scaler
        
        # Save model
        model_path = f'results/train_optimization_models/models/efficiency_{target}.joblib'
        scaler_path = f'results/train_optimization_models/models/efficiency_{target}_scaler.joblib'
        joblib.dump(models[best_model_name], model_path)
        joblib.dump(scaler, scaler_path)
        logging.info(f"Saved best model for {target}: {best_model_name}")
    
    return efficiency_models, efficiency_scalers

def train_recombination_predictor(X, y_recombination, device_params):
    """Train model to predict recombination from device parameters."""
    logging.info("\n=== Training Recombination Predictor ===")
    
    # Handle small datasets
    if len(X) < 5:
        logging.warning(f"Very small dataset ({len(X)} samples) - using all data for training")
        X_train, X_test = X, X
        y_train, y_test = y_recombination, y_recombination
    else:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_recombination, test_size=0.2, random_state=42
        )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models for each recombination metric
    recombination_models = {}
    recombination_scalers = {}
    
    for target in y_recombination.columns:
        logging.info(f"\nTraining model for {target}...")
        
        # Train multiple models
        models = {}
        for model_name in ML_MODEL_NAMES:
            model = ML_MODELS[model_name]
            model.fit(X_train_scaled, y_train[target])
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test[target], y_pred)
            rmse = np.sqrt(mean_squared_error(y_test[target], y_pred))
            
            logging.info(f"{model_name} - {target}: R²={r2:.4f}, RMSE={rmse:.4f}")
            models[model_name] = model
        
        # Save best model (highest R²)
        best_model_name = max(models.keys(), key=lambda m: r2_score(y_test[target], models[m].predict(X_test_scaled)))
        recombination_models[target] = models[best_model_name]
        recombination_scalers[target] = scaler
        
        # Save model
        model_path = f'results/train_optimization_models/models/recombination_{target}.joblib'
        scaler_path = f'results/train_optimization_models/models/recombination_{target}_scaler.joblib'
        joblib.dump(models[best_model_name], model_path)
        joblib.dump(scaler, scaler_path)
        logging.info(f"Saved best model for {target}: {best_model_name}")
    
    return recombination_models, recombination_scalers

def train_inverse_optimizer(X, y_efficiency, device_params):
    """Train inverse model to predict optimal device parameters from target efficiency."""
    logging.info("\n=== Training Inverse Optimizer ===")
    
    # Find the best efficiency configuration for each target
    inverse_models = {}
    inverse_scalers = {}
    
    for target in y_efficiency.columns:
        logging.info(f"\nTraining inverse model for {target}...")
        
        # Find optimal configurations (top 10% efficiency)
        efficiency_threshold = y_efficiency[target].quantile(0.9)
        optimal_mask = y_efficiency[target] >= efficiency_threshold
        
        if optimal_mask.sum() < 10:
            logging.warning(f"Too few optimal configurations for {target}, using top 10")
            optimal_mask = y_efficiency[target] >= y_efficiency[target].quantile(0.9)
        
        X_optimal = X[optimal_mask]
        y_optimal = y_efficiency[target][optimal_mask]
        
        logging.info(f"Optimal configurations for {target}: {len(X_optimal)} samples")
        
        # Skip inverse training if we have too few samples
        if len(X_optimal) < 2:
            logging.warning(f"Skipping inverse training for {target} - insufficient samples ({len(X_optimal)})")
            continue
        
        # Train model to predict device parameters from efficiency
        # We'll use the efficiency as an additional feature
        X_with_efficiency = X_optimal.copy()
        X_with_efficiency[f'target_{target}'] = y_optimal
        
        # For very small datasets, use all data for training
        if len(X_with_efficiency) < 5:
            X_train, X_test = X_with_efficiency, X_with_efficiency
            y_train, y_test = X_optimal, X_optimal
        else:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_with_efficiency, X_optimal, test_size=0.2, random_state=42
            )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models for each device parameter
        param_models = {}
        for param in device_params:
            models = {}
            for model_name in ML_MODEL_NAMES:
                model = ML_MODELS[model_name]
                model.fit(X_train_scaled, y_train[param])
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                r2 = r2_score(y_test[param], y_pred)
                rmse = np.sqrt(mean_squared_error(y_test[param], y_pred))
                
                models[model_name] = model
            
            # Save best model
            best_model_name = max(models.keys(), key=lambda m: r2_score(y_test[param], models[m].predict(X_test_scaled)))
            param_models[param] = models[best_model_name]
            
            # Save model
            model_path = f'results/train_optimization_models/models/inverse_{target}_{param}.joblib'
            joblib.dump(models[best_model_name], model_path)
        
        inverse_models[target] = param_models
        inverse_scalers[target] = scaler
        
        logging.info(f"Saved inverse models for {target}")
    
    return inverse_models, inverse_scalers

def create_optimization_plots(df, efficiency_models, recombination_models):
    """Create visualization plots for optimization analysis."""
    logging.info("\n=== Creating Optimization Plots ===")
    
    plots_dir = 'results/train_optimization_models/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Efficiency vs Recombination scatter plot
    if 'MPP' in df.columns and 'IntSRHn_mean' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['IntSRHn_mean'], df['MPP'], alpha=0.6)
        plt.xlabel('IntSRHn (A/m²)')
        plt.ylabel('MPP (W/m²)')
        plt.title('Efficiency vs Electron Recombination Rate')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{plots_dir}/efficiency_vs_recombination.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find optimal recombination range
        best_efficiency = df['MPP'].max()
        optimal_recombination = df.loc[df['MPP'].idxmax(), 'IntSRHn_mean']
        logging.info(f"Best efficiency: {best_efficiency:.2f} W/m²")
        logging.info(f"Optimal IntSRHn: {optimal_recombination:.2e} A/m²")
    
    # 2. Feature importance for efficiency prediction
    if 'MPP' in efficiency_models:
        rf_model = efficiency_models['MPP']
        if hasattr(rf_model, 'feature_importances_'):
            importance = rf_model.feature_importances_
            feature_names = [col for col in df.columns if col.startswith('L') and '_' in col]
            
            # Create DataFrame for importance
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
            plt.title('Feature Importance for Efficiency Prediction')
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/efficiency_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 3. Feature importance for recombination prediction
    if 'IntSRHn_mean' in recombination_models:
        rf_model = recombination_models['IntSRHn_mean']
        if hasattr(rf_model, 'feature_importances_'):
            importance = rf_model.feature_importances_
            feature_names = [col for col in df.columns if col.startswith('L') and '_' in col]
            
            # Create DataFrame for importance
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
            plt.title('Feature Importance for Recombination Prediction')
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/recombination_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix of All Parameters')
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Optimization plots saved to {plots_dir}")

def main():
    """Main function to train optimization models."""
    logging.info("Starting optimization model training...")
    
    # Load enhanced data
    df = load_enhanced_data()
    
    # Prepare data
    X, y_efficiency, y_recombination, device_params = prepare_optimization_data(df)
    
    # Train efficiency predictor
    efficiency_models, efficiency_scalers = train_efficiency_predictor(X, y_efficiency, device_params)
    
    # Train recombination predictor
    recombination_models, recombination_scalers = train_recombination_predictor(X, y_recombination, device_params)
    
    # Train inverse optimizer
    inverse_models, inverse_scalers = train_inverse_optimizer(X, y_efficiency, device_params)
    
    # Create optimization plots
    create_optimization_plots(df, efficiency_models, recombination_models)
    
    # Save model metadata
    metadata = {
        'device_params': device_params,
        'efficiency_targets': list(y_efficiency.columns),
        'recombination_targets': list(y_recombination.columns),
        'training_date': datetime.now().isoformat(),
        'data_shape': df.shape,
        'model_info': {
            'efficiency_models': list(efficiency_models.keys()),
            'recombination_models': list(recombination_models.keys()),
            'inverse_models': list(inverse_models.keys())
        }
    }
    
    metadata_path = 'results/train_optimization_models/models/metadata.json'
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info("\n=== Optimization Model Training Complete ===")
    logging.info(f"Models saved to: results/train_optimization_models/models/")
    logging.info(f"Plots saved to: results/train_optimization_models/plots/")
    logging.info(f"Metadata saved to: {metadata_path}")
    
    # Summary
    logging.info(f"\nSummary:")
    logging.info(f"- Device parameters: {len(device_params)}")
    logging.info(f"- Efficiency targets: {len(y_efficiency.columns)}")
    logging.info(f"- Recombination targets: {len(y_recombination.columns)}")
    logging.info(f"- Training samples: {len(X)}")

if __name__ == "__main__":
    main() 