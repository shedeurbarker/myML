"""
Solar Cell Optimization Model Training Script - MODIFIED FOR RECOMBINATION + PCE
===============================================================================

PURPOSE:
--------
Trains machine learning models for solar cell optimization using physics-based simulation data.
MODIFIED VERSION: Focuses on recombination prediction and PCE (Power Conversion Efficiency).

MODELS TRAINED:
---------------
1. PCE Predictor: Power Conversion Efficiency from device parameters (1 model)
2. Recombination Predictors: IntSRHn_mean, IntSRHn_std, IntSRHp_mean, IntSRHp_std, IntSRH_total, IntSRH_ratio (6 models)
3. Optimal Recombination Analysis: Direct analysis of recombination-PCE relationship

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
- Models: results/train_optimization_models/models/ (~7 total models)
- Analysis: results/train_optimization_models/optimal_recombination_analysis.json
- Plots: results/train_optimization_models/plots/
- Logs: results/train_optimization_models/optimization_training.log

USAGE:
------
python scripts/5_train_optimization_models_modified.py

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
import json

# SHAP for comprehensive feature importance analysis
try:
    import shap
    SHAP_AVAILABLE = True
    logging.info("SHAP library available for feature importance analysis")
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP library not available. Install with: pip install shap")

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define ML models and names directly
ML_MODELS = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'LinearRegression': LinearRegression()
}

ML_MODEL_NAMES = {
    'RandomForest': 'Random Forest',
    'GradientBoosting': 'Gradient Boosting', 
    'LinearRegression': 'Linear Regression'
}

# Set up logging
log_dir = 'results/train_optimization_models'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'optimization_training_modified.log')

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
    """Load the enhanced simulation data."""
    try:
        # Try to load from the prepared ML data
        X_file = 'results/prepare_ml_data/X_full.csv'
        y_efficiency_file = 'results/prepare_ml_data/y_efficiency_full.csv'
        y_recombination_file = 'results/prepare_ml_data/y_recombination_full.csv'
        
        if os.path.exists(X_file) and os.path.exists(y_efficiency_file) and os.path.exists(y_recombination_file):
            logging.info("Loading prepared ML data...")
            X = pd.read_csv(X_file, index_col=0)
            y_efficiency = pd.read_csv(y_efficiency_file, index_col=0)
            y_recombination = pd.read_csv(y_recombination_file, index_col=0)
            
            # Reset indices to ensure they match before concatenation
            X = X.reset_index(drop=True)
            y_efficiency = y_efficiency.reset_index(drop=True)
            y_recombination = y_recombination.reset_index(drop=True)
            
            # Combine into single dataframe for processing
            df = pd.concat([X, y_efficiency, y_recombination], axis=1)
            logging.info(f"Loaded prepared data: {df.shape}")
            return df
        else:
            logging.warning("Prepared ML data not found. Loading from extracted data...")
            # Fallback to extracted data
            df = pd.read_csv('results/extract_simulation_data/combined_output_with_efficiency.csv')
            logging.info(f"Loaded extracted data: {df.shape}")
            return df
            
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Clean and prepare data for training."""
    logging.info("Cleaning data...")
    
    # Remove rows with missing values in key columns
    key_columns = ['PCE'] + [col for col in df.columns if 'IntSRH' in col]
    df_clean = df.dropna(subset=key_columns)
    
    # Remove infinite values
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna()
    
    logging.info(f"Clean data shape: {df_clean.shape}")
    
    return df_clean

def prepare_optimization_data(df):
    """Prepare data for optimization models - MODIFIED FOR RECOMBINATION + PCE."""
    logging.info("Preparing optimization data for recombination + PCE prediction...")
    
    # Define all features (both primary and derived parameters)
    # Exclude target columns and other performance metrics from features
    target_columns = [
        'MPP', 'Jsc', 'Voc', 'FF',  # Efficiency targets (excluding PCE)
        'IntSRHn_mean', 'IntSRHn_std', 'IntSRHn_min', 'IntSRHn_max',  # Recombination targets
        'PCE', 'IntSRHp_mean', 'IntSRHp_std', 'IntSRH_total', 'IntSRH_ratio'  # PCE and other recombination metrics
    ]
    all_features = [col for col in df.columns if col not in target_columns]
    logging.info(f"All features: {all_features}")
    logging.info(f"Total features: {len(all_features)}")
    
    # Define PCE metric (primary efficiency target)
    pce_metrics = ['PCE']
    available_pce = [col for col in pce_metrics if col in df.columns]
    logging.info(f"Available PCE metrics: {available_pce}")
    
    # Define recombination metrics (targets) - ALL recombination metrics
    recombination_metrics = [
        'IntSRHn_mean', 'IntSRHn_std',  # Electron recombination
        'IntSRHp_mean', 'IntSRHp_std',  # Hole recombination  
        'IntSRH_total', 'IntSRH_ratio'   # Total and ratio
    ]
    available_recombination = [col for col in recombination_metrics if col in df.columns]
    logging.info(f"Available recombination metrics: {available_recombination}")
    
    # Remove rows with missing data
    required_cols = all_features + available_pce + available_recombination
    df_clean = df[required_cols].dropna()
    logging.info(f"Clean data shape: {df_clean.shape}")
    
    # Split features and targets
    X = df_clean[all_features]
    y_pce = df_clean[available_pce]
    y_recombination = df_clean[available_recombination]
    
    logging.info(f"Features (X): {X.shape}")
    logging.info(f"PCE targets: {y_pce.shape}")
    logging.info(f"Recombination targets: {y_recombination.shape}")
    
    return X, y_pce, y_recombination, all_features

def train_pce_predictor(X, y_pce, all_features):
    """Train model to predict PCE from all features (primary and derived)."""
    logging.info("\n=== Training PCE Predictor ===")
    
    # Handle small datasets
    if len(X) < 5:
        logging.warning(f"Very small dataset ({len(X)} samples) - using all data for training")
        X_train, X_test = X, X
        y_train, y_test = y_pce, y_pce
    else:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_pce, test_size=0.2, random_state=42
        )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models for PCE
    pce_models = {}
    pce_scalers = {}
    
    for target in y_pce.columns:
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
        pce_models[target] = models[best_model_name]
        pce_scalers[target] = scaler
        
        # Save model
        model_path = f'results/train_optimization_models/models/pce_{target}.joblib'
        scaler_path = f'results/train_optimization_models/models/pce_{target}_scaler.joblib'
        joblib.dump(models[best_model_name], model_path)
        joblib.dump(scaler, scaler_path)
        logging.info(f"Saved best model for {target}: {best_model_name}")
    
    return pce_models, pce_scalers

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

def find_optimal_recombination_pce_relationship(X, y_pce, y_recombination):
    """Find optimal recombination rates for maximum PCE."""
    logging.info("\n=== Analyzing Recombination-PCE Relationship ===")
    
    # Combine PCE and recombination data
    combined_data = pd.concat([y_pce, y_recombination], axis=1)
    
    # Find optimal recombination rates for maximum PCE
    optimal_recombination_rates = {}
    recombination_analysis = {}
    
    for recombination_col in y_recombination.columns:
        if recombination_col in combined_data.columns:
            # Find the recombination rate that corresponds to maximum PCE
            max_pce_idx = combined_data['PCE'].idxmax()
            optimal_rate = combined_data.loc[max_pce_idx, recombination_col]
            
            # Also find the recombination rate that gives the best PCE/recombination ratio
            if recombination_col != 'IntSRH_ratio':  # Skip ratio column for this analysis
                # Calculate efficiency per recombination rate
                combined_data[f'PCE_per_{recombination_col}'] = combined_data['PCE'] / (combined_data[recombination_col] + 1e-10)
                best_ratio_idx = combined_data[f'PCE_per_{recombination_col}'].idxmax()
                optimal_rate_ratio = combined_data.loc[best_ratio_idx, recombination_col]
                
                optimal_recombination_rates[recombination_col] = {
                    'max_pce_rate': optimal_rate,
                    'best_ratio_rate': optimal_rate_ratio,
                    'max_pce_value': combined_data.loc[max_pce_idx, 'PCE'],
                    'best_ratio_pce': combined_data.loc[best_ratio_idx, 'PCE']
                }
            else:
                optimal_recombination_rates[recombination_col] = {
                    'max_pce_rate': optimal_rate,
                    'max_pce_value': combined_data.loc[max_pce_idx, 'PCE']
                }
            
            # Store analysis data
            recombination_analysis[recombination_col] = {
                'correlation_with_pce': combined_data['PCE'].corr(combined_data[recombination_col]),
                'mean_rate': combined_data[recombination_col].mean(),
                'std_rate': combined_data[recombination_col].std(),
                'min_rate': combined_data[recombination_col].min(),
                'max_rate': combined_data[recombination_col].max()
            }
    
    # Save analysis results
    analysis_results = {
        'optimal_recombination_rates': optimal_recombination_rates,
        'recombination_analysis': recombination_analysis,
        'data_points': len(combined_data)
    }
    
    with open('results/train_optimization_models/optimal_recombination_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    logging.info("Optimal recombination analysis saved")
    return optimal_recombination_rates, recombination_analysis

def perform_shap_analysis(X, y_pce, y_recombination, pce_models, recombination_models, feature_names):
    """Perform SHAP analysis for feature importance."""
    if not SHAP_AVAILABLE:
        logging.warning("SHAP not available - skipping feature importance analysis")
        return
    
    logging.info("\n=== Performing SHAP Analysis ===")
    
    # Scale features for SHAP
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create SHAP explainer for best models
    best_pce_model = list(pce_models.values())[0]  # Use first PCE model
    best_recombination_model = list(recombination_models.values())[0]  # Use first recombination model
    
    # SHAP analysis for PCE model
    try:
        explainer_pce = shap.TreeExplainer(best_pce_model)
        shap_values_pce = explainer_pce.shap_values(X_scaled)
        
        # Create SHAP summary plot for PCE
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_pce, X_scaled, feature_names=feature_names, show=False)
        plt.title('SHAP Feature Importance - PCE Prediction')
        plt.tight_layout()
        plt.savefig('results/train_optimization_models/plots/shap_summary_pce.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save SHAP values
        shap_df_pce = pd.DataFrame(shap_values_pce, columns=feature_names)
        shap_df_pce.to_csv('results/train_optimization_models/shap_values_pce.csv')
        
        logging.info("SHAP analysis completed for PCE model")
        
    except Exception as e:
        logging.error(f"Error in SHAP analysis for PCE: {e}")
    
    # SHAP analysis for recombination model
    try:
        explainer_recombination = shap.TreeExplainer(best_recombination_model)
        shap_values_recombination = explainer_recombination.shap_values(X_scaled)
        
        # Create SHAP summary plot for recombination
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_recombination, X_scaled, feature_names=feature_names, show=False)
        plt.title('SHAP Feature Importance - Recombination Prediction')
        plt.tight_layout()
        plt.savefig('results/train_optimization_models/plots/shap_summary_recombination.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save SHAP values
        shap_df_recombination = pd.DataFrame(shap_values_recombination, columns=feature_names)
        shap_df_recombination.to_csv('results/train_optimization_models/shap_values_recombination.csv')
        
        logging.info("SHAP analysis completed for recombination model")
        
    except Exception as e:
        logging.error(f"Error in SHAP analysis for recombination: {e}")

def create_optimal_recombination_visualizations(X, y_pce, y_recombination, optimal_recombination_rates, recombination_analysis):
    """Create visualizations for optimal recombination analysis."""
    logging.info("\n=== Creating Recombination-PCE Visualizations ===")
    
    # Combine data for plotting
    combined_data = pd.concat([y_pce, y_recombination], axis=1)
    
    # Create subplots for each recombination metric
    n_recombination = len(y_recombination.columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, recombination_col in enumerate(y_recombination.columns):
        if i < len(axes):
            ax = axes[i]
            
            # Scatter plot: PCE vs recombination
            ax.scatter(combined_data[recombination_col], combined_data['PCE'], alpha=0.6, s=20)
            
            # Highlight optimal points
            if recombination_col in optimal_recombination_rates:
                opt_data = optimal_recombination_rates[recombination_col]
                
                # Mark point with maximum PCE
                ax.scatter(opt_data['max_pce_rate'], opt_data['max_pce_value'], 
                          color='red', s=100, marker='*', label='Max PCE')
                
                # Mark point with best ratio (if available)
                if 'best_ratio_rate' in opt_data:
                    ax.scatter(opt_data['best_ratio_rate'], opt_data['best_ratio_pce'], 
                              color='green', s=100, marker='s', label='Best Ratio')
            
            ax.set_xlabel(f'{recombination_col}')
            ax.set_ylabel('PCE (%)')
            ax.set_title(f'PCE vs {recombination_col}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_recombination, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('results/train_optimization_models/plots/recombination_pce_relationship.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = combined_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f')
    plt.title('Correlation Matrix: PCE vs Recombination Metrics')
    plt.tight_layout()
    plt.savefig('results/train_optimization_models/plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Recombination-PCE visualizations created")

def create_optimization_plots(df, pce_models, recombination_models, X):
    """Create comprehensive optimization plots."""
    logging.info("\n=== Creating Optimization Plots ===")
    
    # Model performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # PCE prediction performance
    if 'PCE' in pce_models:
        model = pce_models['PCE']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_pred = model.predict(X_scaled)
        
        axes[0, 0].scatter(df['PCE'], y_pred, alpha=0.6)
        axes[0, 0].plot([df['PCE'].min(), df['PCE'].max()], [df['PCE'].min(), df['PCE'].max()], 'r--')
        axes[0, 0].set_xlabel('Actual PCE (%)')
        axes[0, 0].set_ylabel('Predicted PCE (%)')
        axes[0, 0].set_title('PCE Prediction Performance')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Recombination prediction performance (example with IntSRHn_mean)
    if 'IntSRHn_mean' in recombination_models:
        model = recombination_models['IntSRHn_mean']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_pred = model.predict(X_scaled)
        
        axes[0, 1].scatter(df['IntSRHn_mean'], y_pred, alpha=0.6)
        axes[0, 1].plot([df['IntSRHn_mean'].min(), df['IntSRHn_mean'].max()], 
                        [df['IntSRHn_mean'].min(), df['IntSRHn_mean'].max()], 'r--')
        axes[0, 1].set_xlabel('Actual IntSRHn_mean')
        axes[0, 1].set_ylabel('Predicted IntSRHn_mean')
        axes[0, 1].set_title('Recombination Prediction Performance')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Feature importance for PCE
    if 'PCE' in pce_models and hasattr(pce_models['PCE'], 'feature_importances_'):
        feature_importance = pce_models['PCE'].feature_importances_
        feature_names = X.columns
        sorted_idx = np.argsort(feature_importance)[::-1]
        
        axes[1, 0].barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        axes[1, 0].set_yticks(range(len(sorted_idx)))
        axes[1, 0].set_yticklabels([feature_names[i] for i in sorted_idx])
        axes[1, 0].set_xlabel('Feature Importance')
        axes[1, 0].set_title('Feature Importance - PCE Prediction')
    
    # Feature importance for recombination
    if 'IntSRHn_mean' in recombination_models and hasattr(recombination_models['IntSRHn_mean'], 'feature_importances_'):
        feature_importance = recombination_models['IntSRHn_mean'].feature_importances_
        feature_names = X.columns
        sorted_idx = np.argsort(feature_importance)[::-1]
        
        axes[1, 1].barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        axes[1, 1].set_yticks(range(len(sorted_idx)))
        axes[1, 1].set_yticklabels([feature_names[i] for i in sorted_idx])
        axes[1, 1].set_xlabel('Feature Importance')
        axes[1, 1].set_title('Feature Importance - Recombination Prediction')
    
    plt.tight_layout()
    plt.savefig('results/train_optimization_models/plots/model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Optimization plots created")

def save_model_metadata(pce_models, recombination_models, all_features):
    """Save metadata about trained models."""
    metadata = {
        'pce_targets': list(pce_models.keys()),
        'recombination_targets': list(recombination_models.keys()),
        'device_params': all_features,
        'total_features': len(all_features),
        'model_types': list(ML_MODEL_NAMES.keys()),
        'training_date': datetime.now().isoformat()
    }
    
    with open('results/train_optimization_models/models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info("Model metadata saved")

def main():
    """Main training function."""
    logging.info("=" * 60)
    logging.info("SOLAR CELL OPTIMIZATION MODEL TRAINING - MODIFIED VERSION")
    logging.info("Focus: Recombination Prediction + PCE")
    logging.info("=" * 60)
    
    # Load data
    df = load_enhanced_data()
    if df is None:
        logging.error("Failed to load data. Exiting.")
        return
    
    # Clean data
    df = clean_data(df)
    
    # Prepare optimization data
    X, y_pce, y_recombination, all_features = prepare_optimization_data(df)
    
    if len(X) == 0:
        logging.error("No data available after preparation. Exiting.")
        return
    
    # Train PCE predictor
    pce_models, pce_scalers = train_pce_predictor(X, y_pce, all_features)
    
    # Train recombination predictor
    recombination_models, recombination_scalers = train_recombination_predictor(X, y_recombination, all_features)
    
    # Find optimal recombination-PCE relationship
    optimal_recombination_rates, recombination_analysis = find_optimal_recombination_pce_relationship(X, y_pce, y_recombination)
    
    # Perform SHAP analysis
    perform_shap_analysis(X, y_pce, y_recombination, pce_models, recombination_models, all_features)
    
    # Create visualizations
    create_optimal_recombination_visualizations(X, y_pce, y_recombination, optimal_recombination_rates, recombination_analysis)
    create_optimization_plots(df, pce_models, recombination_models, X)
    
    # Save model metadata
    save_model_metadata(pce_models, recombination_models, all_features)
    
    logging.info("\n" + "=" * 60)
    logging.info("TRAINING COMPLETED SUCCESSFULLY")
    logging.info(f"PCE models trained: {len(pce_models)}")
    logging.info(f"Recombination models trained: {len(recombination_models)}")
    logging.info("=" * 60)

if __name__ == "__main__":
    main() 