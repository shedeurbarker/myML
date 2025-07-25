"""
Solar Cell Optimization Model Training Script
============================================

PURPOSE:
--------
Trains machine learning models for solar cell optimization using physics-based simulation data.
Implements multi-target learning to predict efficiency, recombination rates, and optimal device parameters.

MODELS TRAINED:
---------------
1. Efficiency Predictors: MPP, Jsc, Voc, FF from device parameters (4 models)
2. Recombination Predictors: IntSRHn_mean, IntSRHn_std, IntSRHp_mean, IntSRHp_std, IntSRH_total, IntSRH_ratio (6 models)
3. Optimal Recombination Analysis: Direct analysis of recombination-efficiency relationship

ALGORITHMS:
-----------
- Random Forest Regressor (ensemble, robust)
- Gradient Boosting Regressor (sequential boosting)
- Linear Regression (baseline, interpretable)

INPUT:
------
- ML-ready data: results/prepare_ml_data/X_full.csv, y_efficiency_full.csv, y_recombination_full.csv
- Contains derived features and prepared targets for machine learning

OUTPUT:
-------
- Models: results/train_optimization_models/models/ (~10 total models)
- Analysis: results/train_optimization_models/optimal_recombination_analysis.json
- Plots: results/train_optimization_models/plots/
- Logs: results/train_optimization_models/optimization_training.log

USAGE:
------
python scripts/5_train_optimization_models.py

PREREQUISITES:
--------------
1. Run scripts/2_generate_simulations_enhanced.py
2. Run scripts/3_extract_simulation_data.py
3. Run scripts/4_prepare_ml_data.py
"""

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
    """Load the ML-ready data prepared by the prepare_ml_data script."""
    # Load the full dataset prepared by script 4
    X_path = 'results/prepare_ml_data/X_full.csv'
    y_efficiency_path = 'results/prepare_ml_data/y_efficiency_full.csv'
    y_recombination_path = 'results/prepare_ml_data/y_recombination_full.csv'
    
    if not os.path.exists(X_path):
        logging.error(f"ML data not found at {X_path}")
        logging.error("Please run scripts/4_prepare_ml_data.py first")
        raise FileNotFoundError(f"ML data not found: {X_path}")
    
    # Load features and targets
    X = pd.read_csv(X_path)
    y_efficiency = pd.read_csv(y_efficiency_path) if os.path.exists(y_efficiency_path) else pd.DataFrame()
    y_recombination = pd.read_csv(y_recombination_path) if os.path.exists(y_recombination_path) else pd.DataFrame()
    
    logging.info(f"Loaded ML data: X={X.shape}, y_efficiency={y_efficiency.shape}, y_recombination={y_recombination.shape}")
    logging.info(f"Features: {list(X.columns)}")
    
    # Combine into a single dataframe for compatibility
    df = X.copy()
    for col in y_efficiency.columns:
        df[col] = y_efficiency[col]
    for col in y_recombination.columns:
        df[col] = y_recombination[col]
    
    return df

def prepare_optimization_data(df):
    """Prepare data for optimization models."""
    logging.info("Preparing optimization data...")
    
    # Define all features (both primary and derived parameters)
    # Exclude target columns and other performance metrics from features
    target_columns = [
        'MPP', 'Jsc', 'Voc', 'FF',  # Efficiency targets
        'IntSRHn_mean', 'IntSRHn_std', 'IntSRHn_min', 'IntSRHn_max',  # Recombination targets
        'PCE', 'IntSRHp_mean', 'IntSRHp_std', 'IntSRH_total', 'IntSRH_ratio'  # Other performance metrics
    ]
    all_features = [col for col in df.columns if col not in target_columns]
    logging.info(f"All features: {all_features}")
    logging.info(f"Total features: {len(all_features)}")
    
    # Define efficiency metrics (targets)
    efficiency_metrics = ['MPP', 'Jsc', 'Voc', 'FF']
    available_efficiency = [col for col in efficiency_metrics if col in df.columns]
    logging.info(f"Available efficiency metrics: {available_efficiency}")
    
    # Define recombination metrics (targets) - ALL recombination metrics
    recombination_metrics = [
        'IntSRHn_mean', 'IntSRHn_std',  # Electron recombination
        'IntSRHp_mean', 'IntSRHp_std',  # Hole recombination  
        'IntSRH_total', 'IntSRH_ratio'   # Total and ratio
    ]
    available_recombination = [col for col in recombination_metrics if col in df.columns]
    logging.info(f"Available recombination metrics: {available_recombination}")
    
    # Remove rows with missing data
    required_cols = all_features + available_efficiency + available_recombination
    df_clean = df[required_cols].dropna()
    logging.info(f"Clean data shape: {df_clean.shape}")
    
    # Split features and targets
    X = df_clean[all_features]
    y_efficiency = df_clean[available_efficiency]
    y_recombination = df_clean[available_recombination]
    
    logging.info(f"Features (X): {X.shape}")
    logging.info(f"Efficiency targets: {y_efficiency.shape}")
    logging.info(f"Recombination targets: {y_recombination.shape}")
    
    return X, y_efficiency, y_recombination, all_features

def train_efficiency_predictor(X, y_efficiency, all_features):
    """Train model to predict efficiency from all features (primary and derived)."""
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

def train_recombination_predictor(X, y_recombination, all_features):
    """Train model to predict recombination from all features (primary and derived)."""
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

def find_optimal_recombination_efficiency_relationship(X, y_efficiency, y_recombination):
    """Find the optimal recombination rate that maximizes efficiency."""
    logging.info("\n=== Finding Optimal Recombination-Efficiency Relationship ===")
    
    # Combine efficiency and recombination data
    combined_data = pd.concat([y_efficiency, y_recombination], axis=1)
    
    # Find the configuration with maximum efficiency
    max_efficiency_idx = combined_data['MPP'].idxmax()
    max_efficiency = combined_data.loc[max_efficiency_idx, 'MPP']
    
    # Get recombination rates for maximum efficiency configuration
    optimal_recombination_rates = {}
    for col in y_recombination.columns:
        optimal_recombination_rates[col] = combined_data.loc[max_efficiency_idx, col]
    
    logging.info(f"Maximum efficiency: {max_efficiency:.2f} W/m²")
    logging.info(f"Optimal recombination rates:")
    for metric, rate in optimal_recombination_rates.items():
        logging.info(f"  {metric}: {rate:.2e}")
    
    # Analyze recombination-efficiency relationship
    recombination_analysis = {}
    for col in y_recombination.columns:
        if col in combined_data.columns:
            # Find correlation with efficiency
            correlation = combined_data['MPP'].corr(combined_data[col])
            recombination_analysis[col] = {
                'correlation_with_efficiency': correlation,
                'optimal_rate': optimal_recombination_rates.get(col, np.nan),
                'min_rate': combined_data[col].min(),
                'max_rate': combined_data[col].max(),
                'mean_rate': combined_data[col].mean()
            }
    
    # Save analysis results
    analysis_path = 'results/train_optimization_models/optimal_recombination_analysis.json'
    import json
    with open(analysis_path, 'w') as f:
        json.dump({
            'max_efficiency': max_efficiency,
            'optimal_recombination_rates': optimal_recombination_rates,
            'recombination_analysis': recombination_analysis,
            'analysis_date': datetime.now().isoformat()
        }, f, indent=2, default=str)
    
    logging.info(f"Recombination analysis saved to: {analysis_path}")
    
    return optimal_recombination_rates, recombination_analysis

def create_optimization_plots(df, efficiency_models, recombination_models, X):
    """Create visualization plots for optimization analysis with focus on recombination."""
    logging.info("\n=== Creating Optimization Plots ===")
    
    plots_dir = 'results/train_optimization_models/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Efficiency vs Recombination scatter plots (MULTIPLE)
    recombination_metrics = ['IntSRHn_mean', 'IntSRHp_mean', 'IntSRH_total', 'IntSRH_ratio']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(recombination_metrics):
        if metric in df.columns and 'MPP' in df.columns:
            axes[i].scatter(df[metric], df['MPP'], alpha=0.6, color='blue')
            axes[i].set_xlabel(f'{metric} (A/m²)')
            axes[i].set_ylabel('MPP (W/m²)')
            axes[i].set_title(f'Efficiency vs {metric}')
            if metric != 'IntSRH_ratio':
                axes[i].set_xscale('log')
            axes[i].grid(True, alpha=0.3)
            
            # Find optimal recombination range
            best_efficiency = df['MPP'].max()
            optimal_recombination = df.loc[df['MPP'].idxmax(), metric]
            logging.info(f"Best efficiency: {best_efficiency:.2f} W/m²")
            logging.info(f"Optimal {metric}: {optimal_recombination:.2e}")
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/efficiency_vs_recombination_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Recombination correlation analysis
    if len([col for col in recombination_metrics if col in df.columns]) > 1:
        recombination_cols = [col for col in recombination_metrics if col in df.columns]
        recombination_data = df[recombination_cols]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(recombination_data.corr(), annot=True, cmap='coolwarm', center=0, 
                    square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Recombination Metrics Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/recombination_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Feature importance for efficiency prediction
    if 'MPP' in efficiency_models:
        rf_model = efficiency_models['MPP']
        if hasattr(rf_model, 'feature_importances_'):
            importance = rf_model.feature_importances_
            
            # Debug: Check if lengths match
            logging.info(f"Feature importance length: {len(importance)}")
            logging.info(f"X columns length: {len(X.columns)}")
            
            # Create generic feature names if lengths don't match
            if len(importance) != len(X.columns):
                logging.warning(f"Length mismatch! Creating generic feature names")
                feature_names = [f'Feature_{i}' for i in range(len(importance))]
            else:
                feature_names = X.columns.tolist()
            
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
            
            # Debug: Check if lengths match
            logging.info(f"Feature importance length: {len(importance)}")
            logging.info(f"X columns length: {len(X.columns)}")
            
            # Create generic feature names if lengths don't match
            if len(importance) != len(X.columns):
                logging.warning(f"Length mismatch! Creating generic feature names")
                feature_names = [f'Feature_{i}' for i in range(len(importance))]
            else:
                feature_names = X.columns.tolist()
            
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
    X, y_efficiency, y_recombination, all_features = prepare_optimization_data(df)
    
    # Train efficiency predictor
    efficiency_models, efficiency_scalers = train_efficiency_predictor(X, y_efficiency, all_features)
    
    # Train recombination predictor
    recombination_models, recombination_scalers = train_recombination_predictor(X, y_recombination, all_features)
    
    # Find optimal recombination-efficiency relationship (SIMPLIFIED!)
    optimal_recombination_rates, recombination_analysis = find_optimal_recombination_efficiency_relationship(X, y_efficiency, y_recombination)
    
    # Create optimization plots
    create_optimization_plots(df, efficiency_models, recombination_models, X)
    
    # Save model metadata
    metadata = {
        'all_features': all_features,
        'efficiency_targets': list(y_efficiency.columns),
        'recombination_targets': list(y_recombination.columns),
        'training_date': datetime.now().isoformat(),
        'data_shape': df.shape,
        'optimal_recombination_rates': optimal_recombination_rates,
        'recombination_analysis': recombination_analysis,
        'model_info': {
            'efficiency_models': list(efficiency_models.keys()),
            'recombination_models': list(recombination_models.keys())
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
    logging.info(f"- All features: {len(all_features)}")
    logging.info(f"- Efficiency targets: {len(y_efficiency.columns)}")
    logging.info(f"- Recombination targets: {len(y_recombination.columns)}")
    logging.info(f"- Training samples: {len(X)}")

if __name__ == "__main__":
    main() 