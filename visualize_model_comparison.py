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
        
        for model in ['RandomForest', 'GradientBoosting', 'LinearRegression']:
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
    
    # Set style
    plt.style.use('default')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Bar plots for Mean Accuracy
    ax1 = plt.subplot(2, 1, 1)
    for target in ['IntSRHn', 'IntSRHp']:
        accuracies = [metrics[target][model]['Mean Accuracy'] for model in ['RandomForest', 'GradientBoosting', 'LinearRegression']]
        x = np.arange(len(accuracies))
        width = 0.35
        ax1.bar(x + (width if target == 'IntSRHp' else 0), accuracies, width, label=target)
    
    ax1.set_ylabel('Mean Accuracy (%)')
    ax1.set_title('Mean Accuracy Comparison')
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels(['Random Forest', 'Gradient Boosting', 'Linear Regression'])
    ax1.legend()
    
    # 2. Bar plots for R² scores
    ax2 = plt.subplot(2, 1, 2)
    for target in ['IntSRHn', 'IntSRHp']:
        r2_scores = [metrics[target][model]['R²'] for model in ['RandomForest', 'GradientBoosting', 'LinearRegression']]
        x = np.arange(len(r2_scores))
        width = 0.35
        ax2.bar(x + (width if target == 'IntSRHp' else 0), r2_scores, width, label=target)
    
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score Comparison')
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels(['Random Forest', 'Gradient Boosting', 'Linear Regression'])
    ax2.legend()
    
    # Register radar projection before using it
    try:
        theta = radar_factory(6)
        labels = ['Mean Accuracy', 'Median Accuracy', 'Within 90%', 'Within 80%', 'Within 70%', 'R²']
        # 3. Radar chart for IntSRHn
        #ax3 = plt.subplot(2, 2, 3, projection='radar')
        for model in ['RandomForest', 'GradientBoosting', 'LinearRegression']:
            values = [metrics['IntSRHn'][model][label] for label in labels]
            #ax3.plot(theta, values, label=model)
            #ax3.fill(theta, values, alpha=0.1)
        #ax3.set_varlabels(labels)
        #ax3.set_title('IntSRHn Model Comparison')
        #ax3.legend(loc='upper right', bbox_to_anchor=(0.3, 0.3))
        # 4. Radar chart for IntSRHp
        #ax4 = plt.subplot(2, 2, 4, projection='radar')
        #for model in ['RandomForest', 'GradientBoosting', 'LinearRegression']:
        #    values = [metrics['IntSRHp'][model][label] for label in labels]
        #    ax4.plot(theta, values, label=model)
        #    ax4.fill(theta, values, alpha=0.1)
        #ax4.set_varlabels(labels)
        #ax4.set_title('IntSRHp Model Comparison')
        #ax4.legend(loc='upper right', bbox_to_anchor=(0.3, 0.3))
    except Exception as e:
        logging.warning(f"Radar plots could not be generated: {e}")
    
    # Adjust layout and save
    #plt.tight_layout()
    plt.savefig('results/visualize/model_comparison.png', dpi=300, bbox_inches='tight')
    logging.info("Model comparison plot saved to results/visualize/model_comparison.png")
    
    # Create separate plots for each target
    for target in ['IntSRHn', 'IntSRHp']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot for accuracy metrics
        metrics_to_plot = ['Mean Accuracy', 'Median Accuracy', 'Within 90%', 'Within 80%', 'Within 70%']
        x = np.arange(len(metrics_to_plot))
        width = 0.25
        
        for i, model in enumerate(['RandomForest', 'GradientBoosting', 'LinearRegression']):
            values = [metrics[target][model][m] for m in metrics_to_plot]
            ax1.bar(x + i*width, values, width, label=model)
        
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title(f'{target} - Accuracy Metrics')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(metrics_to_plot, rotation=45)
        ax1.legend()
        
        # Bar plot for R²
        r2_values = [metrics[target][model]['R²'] for model in ['RandomForest', 'GradientBoosting', 'LinearRegression']]
        ax2.bar(['Random Forest', 'Gradient Boosting', 'Linear Regression'], r2_values)
        ax2.set_ylabel('R² Score')
        ax2.set_title(f'{target} - R² Scores')
        
        #plt.tight_layout()
        plt.savefig(f'results/visualize/{target}_comparison.png', dpi=300, bbox_inches='tight')
        logging.info(f"{target} comparison plot saved to results/visualize/{target}_comparison.png")

if __name__ == "__main__":
    create_model_comparison_plots() 