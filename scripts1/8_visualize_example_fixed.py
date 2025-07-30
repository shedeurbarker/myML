"""
===============================================================================
COMPREHENSIVE VISUALIZATION OF SOLAR CELL OPTIMIZATION RESULTS
===============================================================================

PURPOSE:
This script creates comprehensive visualizations of the entire solar cell optimization
pipeline results, including optimization outcomes, model performance, and feature
importance analysis. It provides a complete dashboard view of the project's success.

WHAT THIS SCRIPT DOES:
1. Loads optimization results from script 6 (optimal parameters and efficiency)
2. Loads model validation metrics from script 7 (prediction performance)
3. Loads SHAP feature importance analysis from script 5 (model interpretability)
4. Creates comprehensive visualizations including:
   - Optimal parameters visualization with value labels
   - Efficiency vs recombination relationship plots
   - SHAP feature importance plots for efficiency and recombination models
   - Comprehensive dashboard with project status overview
5. Saves all visualizations with detailed metadata

VISUALIZATIONS CREATED:
- Optimal Parameters: Bar charts showing optimal device parameters with values
- Efficiency vs Recombination: Scatter plots showing optimal performance points
- SHAP Feature Importance: Comprehensive feature importance analysis for all models
- Comprehensive Dashboard: Overall project status with key metrics and success indicators

INPUT FILES:
- results/optimize_efficiency/reports/optimization_report.json (from script 6)
- results/predict/model_validation_metrics.csv (from script 7)
- results/train_optimization_models/plots/shap_*.png (from script 5)

OUTPUT FILES:
- results/visualize/comprehensive_dashboard.png (overall project status)
- results/visualize/optimal_parameters.png (optimal device parameters)
- results/visualize/efficiency_vs_recombination_optimal.png (optimal performance)
- results/visualize/shap_importance_*.png (feature importance analysis)
- results/visualize/dashboard_metadata.json (visualization metadata)
- results/visualize/visualize.log (detailed execution log)

DASHBOARD COMPONENTS:
1. Optimization Status: Success/failure indicators
2. Model Validation Status: Performance metrics and accuracy
3. Key Optimization Metrics: Optimal efficiency and recombination values
4. Best Model Performance: Top performing model details

PREREQUISITES:
- Run 1_create_feature_names.py to define feature structure
- Run 2_generate_simulations_enhanced.py to generate simulation data
- Run 3_extract_simulation_data.py to extract simulation results
- Run 4_prepare_ml_data.py to prepare ML datasets
- Run 5_train_optimization_models.py to train models and create SHAP analysis
- Run 6_optimize_efficiency.py to find optimal parameters
- Run 7_predict.py to validate model performance

USAGE:
python scripts/8_visualize_example_fixed.py

AUTHOR: ML Solar Cell Optimization Pipeline
DATE: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import logging
import os
from datetime import datetime
import sys
import json
import joblib

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
log_dir = 'results/visualize'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'visualize.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with num_vars axes."""
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(theta, *args, closed=closed, **kwargs)

        def plot(self, *args, **kwargs):
            return super().plot(theta, *args, **kwargs)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

    register_projection(RadarAxes)
    return theta

def check_prerequisites():
    """Check if all required files and directories exist."""
    logging.info("\n=== Checking Prerequisites ===")
    
    required_files = [
        'results/optimize_efficiency/reports/optimization_report.json',
        'results/train_optimization_models/plots/shap_importance_efficiency.png'
    ]
    
    optional_files = [
        'results/predict/model_validation_metrics.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logging.error("Missing required files:")
        for file_path in missing_files:
            logging.error(f"  - {file_path}")
        logging.error("Run scripts 5 and 6 first to generate required data.")
        return False
    
    # Check optional files
    missing_optional = []
    for file_path in optional_files:
        if not os.path.exists(file_path):
            missing_optional.append(file_path)
    
    if missing_optional:
        logging.warning("Missing optional files (visualization will be limited):")
        for file_path in missing_optional:
            logging.warning(f"  - {file_path}")
    
    logging.info("All prerequisites satisfied")
    return True

def ensure_visualize_results_dir():
    os.makedirs('results/visualize', exist_ok=True)

def load_optimization_results():
    """Load optimization results from script 6."""
    results_dir = 'results/optimize_efficiency'
    
    # Check for optimization report
    report_path = os.path.join(results_dir, 'reports', 'optimization_report.json')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            return json.load(f)
    
    logging.warning("No optimization report found. Run script 6 first.")
    return None

def load_model_validation_metrics():
    """Load model validation metrics from script 7."""
    metrics_path = 'results/predict/model_validation_metrics.csv'
    if os.path.exists(metrics_path):
        return pd.read_csv(metrics_path)
    
    logging.warning("No validation metrics found. Run script 7 first.")
    return None

def create_optimization_results_visualization():
    """Create comprehensive visualization of optimization results."""
    logging.info("Creating optimization results visualization...")
    
    results = load_optimization_results()
    if not results:
        logging.warning("No optimization results to visualize")
        return
    
    plots_dir = 'results/visualize'
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Optimal Parameters Visualization
    if 'optimal_parameters' in results:
        optimal_params = results['optimal_parameters']
        
        # Create parameter categories
        thickness_params = {k: v for k, v in optimal_params.items() if 'L' in k and k.endswith('_L')}
        energy_params = {k: v for k, v in optimal_params.items() if 'E_' in k}
        doping_params = {k: v for k, v in optimal_params.items() if 'N_' in k}
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Thickness parameters
        if thickness_params:
            axes[0].bar(thickness_params.keys(), thickness_params.values(), color='skyblue')
            axes[0].set_title('Optimal Layer Thicknesses')
            axes[0].set_ylabel('Thickness (nm)')
            axes[0].tick_params(axis='x', rotation=45)
        
        # Energy parameters
        if energy_params:
            axes[1].bar(energy_params.keys(), energy_params.values(), color='lightcoral')
            axes[1].set_title('Optimal Energy Levels')
            axes[1].set_ylabel('Energy (eV)')
            axes[1].tick_params(axis='x', rotation=45)
        
        # Doping parameters
        if doping_params:
            axes[2].bar(doping_params.keys(), [v/1e20 for v in doping_params.values()], color='lightgreen')
            axes[2].set_title('Optimal Doping Concentrations')
            axes[2].set_ylabel('Concentration (×10²⁰ m⁻³)')
            axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/optimal_parameters.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Efficiency vs Recombination Trade-off
    if 'optimal_efficiency' in results and 'optimal_recombination' in results:
        plt.figure(figsize=(10, 8))
        
        # Create a scatter plot showing the trade-off
        efficiency = results['optimal_efficiency']
        recombination = results['optimal_recombination']
        
        plt.scatter(recombination, efficiency, s=200, color='red', alpha=0.7, 
                   label=f'Optimal Point\nEfficiency: {efficiency:.2f} W/m²\nRecombination: {recombination:.2e} A/m²')
        
        plt.xlabel('Recombination Rate (A/m²)')
        plt.ylabel('Efficiency (W/m²)')
        plt.title('Optimal Efficiency vs Recombination Trade-off')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/efficiency_vs_recombination_optimal.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Optimization Method Comparison
    if 'all_results' in results:
        methods = list(results['all_results'].keys())
        efficiencies = []
        
        for method in methods:
            if results['all_results'][method].success:
                efficiencies.append(-results['all_results'][method].fun)
            else:
                efficiencies.append(0)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, efficiencies, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('Optimization Method Performance Comparison')
        plt.ylabel('Achieved Efficiency (W/m²)')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, eff in zip(bars, efficiencies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{eff:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/optimization_methods_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info("Optimization results visualization completed")

def create_model_performance_visualization():
    """Create visualization of model performance metrics."""
    logging.info("Creating model performance visualization...")
    
    metrics_df = load_model_validation_metrics()
    if metrics_df is None:
        logging.warning("No model validation metrics to visualize")
        return
    
    plots_dir = 'results/visualize'
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Model Performance Comparison
    plt.figure(figsize=(12, 8))
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # R² comparison
    for target in ['IntSRHn', 'IntSRHp']:
        target_data = metrics_df[metrics_df['Target'] == target]
        if len(target_data) > 0:
            axes[0, 0].bar(target_data['Model'], target_data['R²'], 
                           label=target, alpha=0.7)
    
    axes[0, 0].set_title('R² Score Comparison')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # RMSE comparison
    for target in ['IntSRHn', 'IntSRHp']:
        target_data = metrics_df[metrics_df['Target'] == target]
        if len(target_data) > 0:
            axes[0, 1].bar(target_data['Model'], target_data['RMSE'], 
                           label=target, alpha=0.7)
    
    axes[0, 1].set_title('RMSE Comparison')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Accuracy comparison
    for target in ['IntSRHn', 'IntSRHp']:
        target_data = metrics_df[metrics_df['Target'] == target]
        if len(target_data) > 0:
            axes[1, 0].bar(target_data['Model'], target_data['Mean_Accuracy'], 
                           label=target, alpha=0.7)
    
    axes[1, 0].set_title('Mean Accuracy Comparison')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Within 90% accuracy
    for target in ['IntSRHn', 'IntSRHp']:
        target_data = metrics_df[metrics_df['Target'] == target]
        if len(target_data) > 0:
            axes[1, 1].bar(target_data['Model'], target_data['Within_90%'], 
                           label=target, alpha=0.7)
    
    axes[1, 1].set_title('Predictions Within 90% Accuracy')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Radar Chart for Best Model
    best_model_data = metrics_df.loc[metrics_df['Mean_Accuracy'].idxmax()]
    
    # Prepare radar chart data
    categories = ['R²', 'Mean_Accuracy', 'Within_90%', 'Within_80%', 'Within_70%']
    values = [best_model_data['R²'], best_model_data['Mean_Accuracy'], 
              best_model_data['Within_90%'], best_model_data['Within_80%'], 
              best_model_data['Within_70%']]
    
    # Normalize values to 0-1 range for radar chart
    normalized_values = [v/100 if 'Within' in cat or 'Accuracy' in cat else v for v, cat in zip(values, categories)]
    
    theta = radar_factory(len(categories))
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
    
    ax.plot(theta, normalized_values)
    ax.fill(theta, normalized_values, alpha=0.25)
    ax.set_varlabels(categories)
    ax.set_title(f'Performance Profile: {best_model_data["Model"]} ({best_model_data["Target"]})')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/best_model_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Model performance visualization completed")

def create_feature_importance_visualization():
    """Create visualization of feature importance from SHAP analysis."""
    logging.info("Creating feature importance visualization...")
    
    plots_dir = 'results/visualize'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Check for SHAP results
    shap_dir = 'results/train_optimization_models/plots'
    
    # Load SHAP values if available
    shap_files = {
        'efficiency': f'{shap_dir}/shap_values_efficiency.csv',
        'recombination': f'{shap_dir}/shap_values_recombination.csv'
    }
    
    for target_type, file_path in shap_files.items():
        if os.path.exists(file_path):
            shap_values = pd.read_csv(file_path)
            
            # Calculate mean absolute SHAP values for feature importance
            feature_importance = np.abs(shap_values).mean().sort_values(ascending=False)
            
            plt.figure(figsize=(12, 8))
            feature_importance.head(15).plot(kind='barh')
            plt.title(f'SHAP Feature Importance - {target_type.capitalize()} Prediction')
            plt.xlabel('Mean |SHAP Value|')
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/shap_importance_{target_type}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    logging.info("Feature importance visualization completed")

def create_comprehensive_dashboard():
    """Create a comprehensive dashboard combining all visualizations."""
    logging.info("Creating comprehensive dashboard...")
    
    # Load all available data
    optimization_results = load_optimization_results()
    validation_metrics = load_model_validation_metrics()
    
    # Create dashboard summary
    dashboard_data = {
        'optimization_results': optimization_results is not None,
        'validation_metrics': validation_metrics is not None,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save dashboard metadata
    dashboard_path = 'results/visualize/dashboard_metadata.json'
    with open(dashboard_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Optimization success indicator
    if optimization_results:
        axes[0, 0].text(0.5, 0.5, '✓ Optimization\nCompleted', 
                        ha='center', va='center', fontsize=16, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        axes[0, 0].set_title('Optimization Status')
    else:
        axes[0, 0].text(0.5, 0.5, '✗ Optimization\nNot Found', 
                        ha='center', va='center', fontsize=16,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        axes[0, 0].set_title('Optimization Status')
    
    # 2. Model validation status
    if validation_metrics is not None:
        best_accuracy = validation_metrics['Mean_Accuracy'].max()
        axes[0, 1].text(0.5, 0.5, f'✓ Model Validation\nBest Accuracy: {best_accuracy:.1f}%', 
                        ha='center', va='center', fontsize=16,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        axes[0, 1].set_title('Model Validation Status')
    else:
        axes[0, 1].text(0.5, 0.5, '✗ Model Validation\nNot Found', 
                        ha='center', va='center', fontsize=16,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        axes[0, 1].set_title('Model Validation Status')
    
    # 3. Key metrics summary
    if optimization_results:
        efficiency = optimization_results.get('optimal_efficiency', 0)
        recombination = optimization_results.get('optimal_recombination', 0)
        axes[1, 0].text(0.5, 0.5, f'Optimal Efficiency:\n{efficiency:.2f} W/m²\n\nOptimal Recombination:\n{recombination:.2e} A/m²', 
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 0].set_title('Key Optimization Metrics')
    else:
        axes[1, 0].text(0.5, 0.5, 'No optimization\nresults available', 
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 0].set_title('Key Optimization Metrics')
    
    # 4. Model performance summary
    if validation_metrics is not None:
        best_model = validation_metrics.loc[validation_metrics['Mean_Accuracy'].idxmax()]
        axes[1, 1].text(0.5, 0.5, f'Best Model:\n{best_model["Model"]}\n\nTarget: {best_model["Target"]}\nAccuracy: {best_model["Mean_Accuracy"]:.1f}%', 
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 1].set_title('Best Model Performance')
    else:
        axes[1, 1].text(0.5, 0.5, 'No validation\nmetrics available', 
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 1].set_title('Best Model Performance')
    
    # Remove axis labels
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('Solar Cell Optimization Dashboard', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/visualize/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Comprehensive dashboard created")

def save_visualization_metadata(optimization_results, validation_metrics):
    """Save comprehensive metadata about visualization results."""
    logging.info("\n=== Saving Visualization Metadata ===")
    
    metadata = {
        'visualization_date': datetime.now().isoformat(),
        'optimization_results_available': optimization_results is not None,
        'validation_metrics_available': validation_metrics is not None,
        'visualizations_created': [],
        'data_summary': {}
    }
    
    # Add optimization results summary if available
    if optimization_results:
        metadata['data_summary']['optimization'] = {
            'optimal_efficiency': optimization_results.get('optimal_efficiency', 0),
            'optimal_recombination': optimization_results.get('optimal_recombination', 0),
            'optimization_method': optimization_results.get('optimization_method', 'Unknown'),
            'parameter_count': len(optimization_results.get('optimal_parameters', {}))
        }
    
    # Add validation metrics summary if available
    if validation_metrics is not None:
        metadata['data_summary']['validation'] = {
            'total_targets': len(validation_metrics),
            'mean_accuracy': validation_metrics.get('Mean_Accuracy', []).mean() if hasattr(validation_metrics, 'get') else 0,
            'best_model': validation_metrics.loc[validation_metrics['Mean_Accuracy'].idxmax(), 'Model'] if len(validation_metrics) > 0 else 'None'
        }
    
    # List created visualizations
    plots_dir = 'results/visualize'
    if os.path.exists(plots_dir):
        for file in os.listdir(plots_dir):
            if file.endswith('.png'):
                metadata['visualizations_created'].append(file)
    
    # Save metadata
    metadata_path = 'results/visualize/visualization_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Visualization metadata saved to: {metadata_path}")
    return metadata

def main():
    """Main function for comprehensive visualization."""
    logging.info("Starting comprehensive visualization...")
    
    check_prerequisites() # Check prerequisites
    ensure_visualize_results_dir()
    
    # Load all available data
    optimization_results = load_optimization_results()
    validation_metrics = load_model_validation_metrics()
    
    # Create all visualizations
    create_optimization_results_visualization()
    create_model_performance_visualization()
    create_feature_importance_visualization()
    create_comprehensive_dashboard()
    
    # Save metadata
    metadata = save_visualization_metadata(optimization_results, validation_metrics)
    
    # Log comprehensive summary
    logging.info("\n=== VISUALIZATION SUMMARY ===")
    logging.info(f"Optimization results available: {optimization_results is not None}")
    logging.info(f"Validation metrics available: {validation_metrics is not None}")
    
    if optimization_results:
        logging.info("Optimization Results Summary:")
        logging.info(f"  Optimal Efficiency: {optimization_results.get('optimal_efficiency', 0):.2f} W/m²")
        logging.info(f"  Optimal Recombination: {optimization_results.get('optimal_recombination', 0):.2e} A/m²")
        logging.info(f"  Optimization Method: {optimization_results.get('optimization_method', 'Unknown')}")
        logging.info(f"  Parameters Optimized: {len(optimization_results.get('optimal_parameters', {}))}")
    
    if metadata.get('visualizations_created'):
        logging.info(f"Visualizations Created: {len(metadata['visualizations_created'])} files")
        for viz_file in metadata['visualizations_created']:
            logging.info(f"  - {viz_file}")
    
    logging.info("Comprehensive visualization completed!")
    logging.info("All plots saved to: results/visualize/")

if __name__ == "__main__":
    main() 