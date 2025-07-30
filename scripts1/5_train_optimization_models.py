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

def perform_shap_analysis(X, y_efficiency, y_recombination, efficiency_models, recombination_models, feature_names):
    """Perform comprehensive SHAP analysis for feature importance with robust error handling."""
    if not SHAP_AVAILABLE:
        logging.warning("SHAP not available - skipping SHAP analysis")
        return
    
    logging.info("Performing SHAP analysis with robust error handling...")
    
    plots_dir = 'results/train_optimization_models/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Prepare data for SHAP - ensure no missing values and use smaller sample
    X_clean = X.copy()
    X_clean = X_clean.fillna(X_clean.median())
    
    # Use a smaller sample for SHAP to avoid memory/time issues
    sample_size = min(1000, len(X_clean))
    X_sample = X_clean.sample(n=sample_size, random_state=42)
    logging.info(f"Using {sample_size} samples for SHAP analysis")
    
    # SHAP analysis for efficiency prediction
    if efficiency_models and len(y_efficiency.columns) > 0:
        logging.info("Performing SHAP analysis for efficiency prediction...")
        
        # Use the first efficiency target for SHAP analysis
        target_col = y_efficiency.columns[0]
        
        # Find best model for SHAP (prefer Random Forest)
        best_model = None
        for model_name, model in efficiency_models.items():
            if isinstance(model, RandomForestRegressor):
                best_model = model
                break
        
        if best_model is None and efficiency_models:
            best_model = list(efficiency_models.values())[0]
        
        if best_model:
            try:
                logging.info(f"Using {type(best_model).__name__} for SHAP analysis")
                
                # Use a simpler SHAP approach
                if isinstance(best_model, RandomForestRegressor):
                    # For Random Forest, use TreeExplainer with background
                    background = X_sample.sample(n=min(100, len(X_sample)), random_state=42)
                    explainer = shap.TreeExplainer(best_model, background)
                    shap_values = explainer.shap_values(X_sample)
                else:
                    # For other models, use a simpler approach
                    explainer = shap.Explainer(best_model, X_sample)
                    shap_values = explainer(X_sample)
                    if hasattr(shap_values, 'values'):
                        shap_values = shap_values.values
                
                # Create summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=20)
                plt.title(f'SHAP Summary Plot - {target_col} Prediction')
                plt.tight_layout()
                plt.savefig(f'{plots_dir}/shap_summary_efficiency.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Create feature importance plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
                plt.title(f'SHAP Feature Importance - {target_col}')
                plt.tight_layout()
                plt.savefig(f'{plots_dir}/shap_importance_efficiency.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save SHAP values
                shap_values_df = pd.DataFrame(shap_values, columns=feature_names)
                shap_values_df.to_csv(f'{plots_dir}/shap_values_efficiency.csv', index=False)
                
                logging.info(f"SHAP analysis completed for {target_col}")
                
            except Exception as e:
                logging.error(f"SHAP analysis failed for efficiency: {e}")
                logging.info("Continuing without SHAP analysis for efficiency")
    
    # SHAP analysis for recombination prediction
    if recombination_models and len(y_recombination.columns) > 0:
        logging.info("Performing SHAP analysis for recombination prediction...")
        
        # Use the first recombination target for SHAP analysis
        target_col = y_recombination.columns[0]
        
        # Find best model for SHAP
        best_model = None
        for model_name, model in recombination_models.items():
            if isinstance(model, RandomForestRegressor):
                best_model = model
                break
        
        if best_model is None and recombination_models:
            best_model = list(recombination_models.values())[0]
        
        if best_model:
            try:
                logging.info(f"Using {type(best_model).__name__} for SHAP analysis")
                
                # Use a simpler SHAP approach
                if isinstance(best_model, RandomForestRegressor):
                    # For Random Forest, use TreeExplainer with background
                    background = X_sample.sample(n=min(100, len(X_sample)), random_state=42)
                    explainer = shap.TreeExplainer(best_model, background)
                    shap_values = explainer.shap_values(X_sample)
                else:
                    # For other models, use a simpler approach
                    explainer = shap.Explainer(best_model, X_sample)
                    shap_values = explainer(X_sample)
                    if hasattr(shap_values, 'values'):
                        shap_values = shap_values.values
                
                # Create summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=20)
                plt.title(f'SHAP Summary Plot - {target_col} Prediction')
                plt.tight_layout()
                plt.savefig(f'{plots_dir}/shap_summary_recombination.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Create feature importance plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
                plt.title(f'SHAP Feature Importance - {target_col}')
                plt.tight_layout()
                plt.savefig(f'{plots_dir}/shap_importance_recombination.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save SHAP values
                shap_values_df = pd.DataFrame(shap_values, columns=feature_names)
                shap_values_df.to_csv(f'{plots_dir}/shap_values_recombination.csv', index=False)
                
                logging.info(f"SHAP analysis completed for {target_col}")
                
            except Exception as e:
                logging.error(f"SHAP analysis failed for recombination: {e}")
                logging.info("Continuing without SHAP analysis for recombination")
    
    logging.info("SHAP analysis completed")

def create_optimal_recombination_visualizations(X, y_efficiency, y_recombination, optimal_recombination_rates, recombination_analysis):
    """Create visualizations for optimal recombination analysis."""
    logging.info("Creating optimal recombination visualizations...")
    
    plots_dir = 'results/train_optimization_models/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Combine efficiency and recombination data
    combined_data = pd.concat([y_efficiency, y_recombination], axis=1)
    
    # 1. Efficiency vs Recombination Scatter Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Efficiency vs Recombination Rates Analysis', fontsize=16, fontweight='bold')
    
    recombination_cols = list(y_recombination.columns)
    for i, col in enumerate(recombination_cols):
        row = i // 3
        col_idx = i % 3
        ax = axes[row, col_idx]
        
        # Create scatter plot
        scatter = ax.scatter(combined_data[col], combined_data['MPP'], 
                           alpha=0.6, s=20, c=combined_data['MPP'], cmap='viridis')
        
        # Highlight optimal point
        optimal_rate = optimal_recombination_rates[col]
        max_efficiency = combined_data['MPP'].max()
        ax.scatter(optimal_rate, max_efficiency, color='red', s=200, 
                  marker='*', edgecolors='black', linewidth=2, label='Optimal Point')
        
        # Add correlation info
        correlation = recombination_analysis[col]['correlation_with_efficiency']
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
               transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_xlabel(f'{col}')
        ax.set_ylabel('MPP (W/m²)')
        ax.set_title(f'{col} vs Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/efficiency_vs_recombination_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Optimal Recombination Rates Bar Chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for plotting (use log scale for large values)
    optimal_rates = list(optimal_recombination_rates.values())
    rate_names = list(optimal_recombination_rates.keys())
    
    # Use log scale for better visualization
    log_rates = np.log10([max(rate, 1e-30) for rate in optimal_rates])
    
    bars = ax.bar(range(len(rate_names)), log_rates, color='skyblue', alpha=0.7)
    
    # Highlight the optimal point for each metric
    for i, (name, rate) in enumerate(optimal_recombination_rates.items()):
        bars[i].set_color('lightcoral')
    
    ax.set_xlabel('Recombination Metrics')
    ax.set_ylabel('Log10(Optimal Rate)')
    ax.set_title('Optimal Recombination Rates for Maximum Efficiency')
    ax.set_xticks(range(len(rate_names)))
    ax.set_xticklabels(rate_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, optimal_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{rate:.2e}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/optimal_recombination_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Correlation Heatmap for Recombination vs Efficiency
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create correlation matrix
    efficiency_cols = ['MPP', 'Jsc', 'Voc', 'FF']
    available_efficiency = [col for col in efficiency_cols if col in combined_data.columns]
    
    correlation_matrix = combined_data[available_efficiency + recombination_cols].corr()
    
    # Create heatmap
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Add text annotations
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(correlation_matrix.columns)
    ax.set_title('Correlation Matrix: Efficiency vs Recombination Metrics')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/efficiency_recombination_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Recombination Rate Distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Recombination Rate Distributions', fontsize=16, fontweight='bold')
    
    for i, col in enumerate(recombination_cols):
        row = i // 3
        col_idx = i % 3
        ax = axes[row, col_idx]
        
        # Create histogram
        ax.hist(combined_data[col], bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        
        # Mark optimal point
        optimal_rate = optimal_recombination_rates[col]
        ax.axvline(optimal_rate, color='red', linestyle='--', linewidth=2, 
                  label=f'Optimal: {optimal_rate:.2e}')
        
        # Add statistics
        mean_rate = recombination_analysis[col]['mean_rate']
        ax.axvline(mean_rate, color='green', linestyle=':', linewidth=2, 
                  label=f'Mean: {mean_rate:.2e}')
        
        ax.set_xlabel(f'{col}')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {col}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Use log scale for x-axis if values are very large
        if combined_data[col].max() > 1e10:
            ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/recombination_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Optimal recombination visualizations created successfully")

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
    
    # Clean up previous training results
    training_dir = 'results/train_optimization_models'
    if os.path.exists(training_dir):
        import shutil
        try:
            shutil.rmtree(training_dir)
            logging.info(f"Deleted previous training directory: {training_dir}")
        except Exception as e:
            logging.warning(f"Could not delete previous training directory: {e}")
    
    # Recreate directories
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs('results/train_optimization_models/models', exist_ok=True)
    os.makedirs('results/train_optimization_models/plots', exist_ok=True)
    logging.info("Created fresh training directories")
    
    try:
        # Load data
        df = load_enhanced_data()
        logging.info(f"Loaded data with shape: {df.shape}")
        
        # Prepare optimization data
        X, y_efficiency, y_recombination, all_features = prepare_optimization_data(df)
        logging.info(f"Prepared data - X: {X.shape}, y_efficiency: {y_efficiency.shape}, y_recombination: {y_recombination.shape}")
        
        # Train efficiency predictors
        logging.info("\n=== Training Efficiency Predictors ===")
        efficiency_models, efficiency_scalers = train_efficiency_predictor(X, y_efficiency, all_features)
        logging.info(f"Trained {len(efficiency_models)} efficiency models")
        
        # Train recombination predictors
        logging.info("\n=== Training Recombination Predictors ===")
        recombination_models, recombination_scalers = train_recombination_predictor(X, y_recombination, all_features)
        logging.info(f"Trained {len(recombination_models)} recombination models")
        
        # Find optimal recombination-efficiency relationship
        logging.info("\n=== Finding Optimal Recombination-Efficiency Relationship ===")
        optimal_recombination_rates, recombination_analysis = find_optimal_recombination_efficiency_relationship(X, y_efficiency, y_recombination)
        
        # Perform SHAP analysis
        logging.info("\n=== Performing SHAP Analysis ===")
        perform_shap_analysis(X, y_efficiency, y_recombination, efficiency_models, recombination_models, all_features)
        
        # Create optimal recombination visualizations
        logging.info("\n=== Creating Optimal Recombination Visualizations ===")
        create_optimal_recombination_visualizations(X, y_efficiency, y_recombination, optimal_recombination_rates, recombination_analysis)
        
        # Create optimization plots
        logging.info("\n=== Creating Optimization Plots ===")
        create_optimization_plots(df, efficiency_models, recombination_models, X)
        
        logging.info("\n=== Model Training Complete ===")
        logging.info(f"Efficiency models saved: {len(efficiency_models)}")
        logging.info(f"Recombination models saved: {len(recombination_models)}")
        
        # Save model metadata for script 6
        metadata = {
            'efficiency_targets': list(y_efficiency.columns),
            'recombination_targets': list(y_recombination.columns),
            'device_params': all_features,
            'training_date': datetime.now().isoformat(),
            'data_shape': df.shape,
            'model_info': {
                'efficiency_models': list(efficiency_models.keys()),
                'recombination_models': list(recombination_models.keys())
            }
        }
        
        metadata_path = 'results/train_optimization_models/models/metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Model metadata saved to: {metadata_path}")
        
        # Print summary
        print("\n=== MODEL TRAINING SUMMARY ===")
        print(f"Data points: {len(X)}")
        print(f"Features: {len(all_features)}")
        print(f"Efficiency targets: {len(y_efficiency.columns)}")
        print(f"Recombination targets: {len(y_recombination.columns)}")
        print(f"Models trained: {len(efficiency_models) + len(recombination_models)}")
        print(f"Results saved to: results/train_optimization_models/")
        
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 