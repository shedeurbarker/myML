import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("=== ML EFFICIENCY PREDICTION WITH INTERFACIAL RECOMBINATION ===")

# Load the ML-ready data
print("Loading ML-ready data...")
df = pd.read_csv('results/extract/ml_ready_data.csv')

print(f"Data shape: {df.shape}")

# Data preprocessing
print("\n=== DATA PREPROCESSING ===")

# Define features and target
feature_cols = [col for col in df.columns if col not in ['P', 'P_abs']]
target_col = 'P_abs'  # Power density as efficiency metric

print(f"Features: {len(feature_cols)}")
print(f"Target: {target_col}")

# Remove extreme outliers (top and bottom 1%)
q_low = df[target_col].quantile(0.01)
q_high = df[target_col].quantile(0.99)
df_filtered = df[(df[target_col] >= q_low) & (df[target_col] <= q_high)]

print(f"Data after outlier removal: {len(df_filtered):,} points")
print(f"Target range: {df_filtered[target_col].min():.2e} to {df_filtered[target_col].max():.2e}")

# Prepare features and target
X = df_filtered[feature_cols]
y = df_filtered[target_col]

# Log-transform the target for better ML performance
y_log = np.log10(y + 1e-30)

print(f"Log-transformed target range: {y_log.min():.2f} to {y_log.max():.2f}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

print(f"Training set: {len(X_train):,} samples")
print(f"Test set: {len(X_test):,} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
}

# Train and evaluate models
print("\n=== MODEL TRAINING AND EVALUATION ===")

results = {}
feature_importance = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    if name in ['Random Forest', 'Gradient Boosting']:
        model.fit(X_train_scaled, y_train)
        # Get feature importance
        feature_importance[name] = dict(zip(feature_cols, model.feature_importances_))
    else:
        model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'predictions': y_pred
    }
    
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

# Compare models
print("\n=== MODEL COMPARISON ===")
comparison_df = pd.DataFrame(results).T[['R²', 'RMSE', 'MAE']].round(4)
print(comparison_df)

# Find best model
best_model_name = comparison_df['R²'].idxmax()
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} (R² = {comparison_df.loc[best_model_name, 'R²']:.4f})")

# Feature importance analysis
print("\n=== FEATURE IMPORTANCE ANALYSIS ===")

if best_model_name in feature_importance:
    # Get feature importance for best model
    importance_dict = feature_importance[best_model_name]
    importance_df = pd.DataFrame(list(importance_dict.items()), 
                                columns=['Feature', 'Importance'])
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    print("Top 10 most important features:")
    print(importance_df.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('results/extract/feature_importance.png', dpi=300, bbox_inches='tight')
    print("Feature importance plot saved to: results/extract/feature_importance.png")

# Interfacial recombination specific analysis
print("\n=== INTERFACIAL RECOMBINATION ANALYSIS ===")

# Find interfacial recombination features
int_features = [col for col in feature_cols if 'IntSRH' in col]
print(f"Interfacial recombination features: {int_features}")

if best_model_name in feature_importance:
    int_importance = {k: v for k, v in importance_dict.items() if 'IntSRH' in k}
    int_importance_df = pd.DataFrame(list(int_importance.items()), 
                                    columns=['Feature', 'Importance'])
    int_importance_df = int_importance_df.sort_values('Importance', ascending=False)
    
    print("\nInterfacial recombination feature importance:")
    print(int_importance_df)

# Create efficiency prediction plots
print("\n=== EFFICIENCY PREDICTION VISUALIZATION ===")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Efficiency Prediction Analysis', fontsize=16, fontweight='bold')

# 1. Actual vs Predicted
ax1 = axes[0, 0]
y_pred_best = results[best_model_name]['predictions']
ax1.scatter(y_test, y_pred_best, alpha=0.6, s=1)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Log10(Power)')
ax1.set_ylabel('Predicted Log10(Power)')
ax1.set_title(f'Actual vs Predicted - {best_model_name}')
ax1.text(0.05, 0.95, f'R² = {results[best_model_name]["R²"]:.4f}', 
         transform=ax1.transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Residuals plot
ax2 = axes[0, 1]
residuals = y_test - y_pred_best
ax2.scatter(y_pred_best, residuals, alpha=0.6, s=1)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel('Predicted Log10(Power)')
ax2.set_ylabel('Residuals')
ax2.set_title('Residuals Plot')

# 3. Model comparison
ax3 = axes[1, 0]
model_names = list(results.keys())
r2_scores = [results[name]['R²'] for name in model_names]
bars = ax3.bar(model_names, r2_scores, alpha=0.7)
ax3.set_ylabel('R² Score')
ax3.set_title('Model Performance Comparison')
ax3.tick_params(axis='x', rotation=45)

# Color the best model
best_idx = model_names.index(best_model_name)
bars[best_idx].set_color('red')

# 4. Interfacial recombination vs efficiency
ax4 = axes[1, 1]
if 'log_IntSRHn' in X.columns:
    scatter = ax4.scatter(X_test['log_IntSRHn'], y_test, 
                         c=y_pred_best, cmap='viridis', alpha=0.6, s=1)
    ax4.set_xlabel('Log10(IntSRHn)')
    ax4.set_ylabel('Actual Log10(Power)')
    ax4.set_title('Interfacial Recombination vs Efficiency\n(colored by predictions)')
    plt.colorbar(scatter, ax=ax4, label='Predicted Log10(Power)')

plt.tight_layout()
plt.savefig('results/extract/efficiency_prediction_analysis.png', dpi=300, bbox_inches='tight')
print("Efficiency prediction analysis saved to: results/extract/efficiency_prediction_analysis.png")

# Optimization recommendations
print("\n=== OPTIMIZATION RECOMMENDATIONS ===")

# Analyze optimal interfacial recombination ranges
if best_model_name in feature_importance:
    # Find top interfacial recombination features
    top_int_features = int_importance_df.head(3)['Feature'].tolist()
    
    print("Top interfacial recombination features for optimization:")
    for feature in top_int_features:
        feature_data = X_test[feature]
        target_data = y_test
        
        # Find optimal range (where efficiency is highest)
        high_efficiency_mask = target_data > target_data.quantile(0.8)
        optimal_range = feature_data[high_efficiency_mask]
        
        print(f"\n{feature}:")
        print(f"  Optimal range: {optimal_range.min():.2f} to {optimal_range.max():.2f}")
        print(f"  Current range: {feature_data.min():.2f} to {feature_data.max():.2f}")
        print(f"  Importance: {importance_dict[feature]:.4f}")

# Save model results
results_summary = {
    'best_model': best_model_name,
    'best_r2': comparison_df.loc[best_model_name, 'R²'],
    'feature_importance': importance_dict if best_model_name in feature_importance else None,
    'model_comparison': comparison_df.to_dict()
}

# Save to file
import json
with open('results/extract/ml_results_summary.json', 'w') as f:
    # Convert numpy types to native Python types for JSON serialization
    json_results = {}
    for key, value in results_summary.items():
        if isinstance(value, dict):
            json_results[key] = {k: float(v) if isinstance(v, np.floating) else v 
                               for k, v in value.items()}
        else:
            json_results[key] = value
    json.dump(json_results, f, indent=2)

print(f"\nML results summary saved to: results/extract/ml_results_summary.json")

print("\n=== SUMMARY ===")
print(f"Best model: {best_model_name}")
print(f"R² Score: {comparison_df.loc[best_model_name, 'R²']:.4f}")
print(f"RMSE: {comparison_df.loc[best_model_name, 'RMSE']:.4f}")
print(f"Key insights for interfacial recombination optimization identified")
print("Visualization plots generated for analysis")

print("\nML efficiency prediction analysis complete!") 