import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Removed seaborn import to avoid compatibility issues
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

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_models import ML_MODEL_NAMES

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

def ensure_visualize_results_dir():
    os.makedirs('results/visualize', exist_ok=True)

def load_metrics_from_csv():
    """Load metrics from the CSV file."""
    csv_path = 'results/predict/model_validation_metrics.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metrics file not found at {csv_path}. Please run predict.py first.")
    
    metrics_df = pd.read_csv(csv_path)
    metrics = {}
    
    for target in ['IntSRHn', 'IntSRHp']:
        metrics[target] = {}
        target_data = metrics_df[metrics_df['Target'] == target]
        
        for model in ML_MODEL_NAMES:
            model_data = target_data[target_data['Model'] == model].iloc[0]
            metrics[target][model] = {
                'Mean Accuracy': model_data['Mean_Accuracy'],
                'Median Accuracy': model_data['Median_Accuracy'],
                'Within 90%': model_data['Within_90%'],
                'Within 80%': model_data['Within_80%'],
                'Within 70%': model_data['Within_70%'],
                'R²': model_data['R²']
            }
    
    return metrics

def create_model_comparison_plots():
    ensure_visualize_results_dir()
    """Create comparison plots for all models."""
    # Load metrics from CSV
    metrics = load_metrics_from_csv()
    
    # Set style - using matplotlib default instead of seaborn
    plt.style.use('default')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Bar plots for Mean Accuracy
    ax1 = plt.subplot(2, 1, 1)
    for target in ['IntSRHn', 'IntSRHp']:
        accuracies = [metrics[target][model]['Mean Accuracy'] for model in ML_MODEL_NAMES]
        x = np.arange(len(ML_MODEL_NAMES))
        width = 0.35
        
        if target == 'IntSRHn':
            bars1 = ax1.bar(x - width/2, accuracies, width, label=target, alpha=0.8)
        else:
            bars2 = ax1.bar(x + width/2, accuracies, width, label=target, alpha=0.8)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Mean Accuracy')
    ax1.set_title('Model Performance Comparison - Mean Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ML_MODEL_NAMES, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Bar plots for R² Score
    ax2 = plt.subplot(2, 1, 2)
    for target in ['IntSRHn', 'IntSRHp']:
        r2_scores = [metrics[target][model]['R²'] for model in ML_MODEL_NAMES]
        x = np.arange(len(ML_MODEL_NAMES))
        width = 0.35
        
        if target == 'IntSRHn':
            bars1 = ax2.bar(x - width/2, r2_scores, width, label=target, alpha=0.8)
        else:
            bars2 = ax2.bar(x + width/2, r2_scores, width, label=target, alpha=0.8)
    
    ax2.set_xlabel('Models')
    ax2.set_ylabel('R² Score')
    ax2.set_title('Model Performance Comparison - R² Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ML_MODEL_NAMES, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/visualize/model_comparison.png', dpi=300, bbox_inches='tight')
    logging.info("Model comparison plot saved to results/visualize/model_comparison.png")
    
    # Create radar charts for detailed comparison
    create_radar_charts(metrics)
    
    plt.show()

def create_radar_charts(metrics):
    """Create radar charts for detailed model comparison."""
    # Define metrics for radar chart
    radar_metrics = ['Mean Accuracy', 'Median Accuracy', 'Within 90%', 'Within 80%', 'Within 70%', 'R²']
    
    for target in ['IntSRHn', 'IntSRHp']:
        fig = plt.figure(figsize=(15, 10))
        
        # Create radar chart
        theta = radar_factory(len(radar_metrics))
        ax = plt.subplot(1, 1, 1, projection='radar')
        
        # Plot data for each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(ML_MODEL_NAMES)))
        
        for i, model in enumerate(ML_MODEL_NAMES):
            values = [metrics[target][model][metric] for metric in radar_metrics]
            ax.plot(theta, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(theta, values, alpha=0.1, color=colors[i])
        
        ax.set_varlabels(radar_metrics)
        ax.set_title(f'Model Performance Radar Chart - {target}', size=15, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(f'results/visualize/radar_chart_{target}.png', dpi=300, bbox_inches='tight')
        logging.info(f"Radar chart for {target} saved to results/visualize/radar_chart_{target}.png")

def create_heatmap(metrics):
    """Create a heatmap of model performance."""
    # Prepare data for heatmap
    targets = ['IntSRHn', 'IntSRHp']
    metrics_list = ['Mean Accuracy', 'Median Accuracy', 'Within 90%', 'Within 80%', 'Within 70%', 'R²']
    
    # Create data matrix
    data = []
    for target in targets:
        for metric in metrics_list:
            row = [metrics[target][model][metric] for model in ML_MODEL_NAMES]
            data.append(row)
    
    data = np.array(data)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
    
    # Set labels
    ax.set_xticks(range(len(ML_MODEL_NAMES)))
    ax.set_xticklabels(ML_MODEL_NAMES, rotation=45)
    
    y_labels = []
    for target in targets:
        for metric in metrics_list:
            y_labels.append(f"{target} - {metric}")
    
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Performance Score', rotation=-90, va="bottom")
    
    # Add text annotations
    for i in range(len(y_labels)):
        for j in range(len(ML_MODEL_NAMES)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color="black")
    
    ax.set_title('Model Performance Heatmap')
    plt.tight_layout()
    plt.savefig('results/visualize/performance_heatmap.png', dpi=300, bbox_inches='tight')
    logging.info("Performance heatmap saved to results/visualize/performance_heatmap.png")

def main():
    """Main function to create all visualizations."""
    try:
        logging.info("Starting visualization process...")
        
        # Create comparison plots
        create_model_comparison_plots()
        
        # Load metrics for heatmap
        metrics = load_metrics_from_csv()
        create_heatmap(metrics)
        
        logging.info("All visualizations completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during visualization: {str(e)}")
        raise

if __name__ == "__main__":
    main() 