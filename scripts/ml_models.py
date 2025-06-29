from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#from sklearn.linear_model import LinearRegression  # Uncomment to enable

# Centralized model dictionary for the ML pipeline
ML_MODELS = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    # 'LinearRegression': LinearRegression(),  # Uncomment to enable
}

# Centralized model list for iteration order
ML_MODEL_NAMES = [k for k in ML_MODELS.keys()] 