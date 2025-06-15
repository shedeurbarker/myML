# Machine Learning for Interface Recombination Analysis

This project implements a machine learning pipeline for predicting interface recombination rates (IntSRHn and IntSRHp) based on semiconductor device simulation data.

## Project Structure

```
myML/
├── data/                  # Raw simulation data
├── models/               # Trained ML models
├── results/             # Output results
│   ├── predict/        # Prediction outputs
│   └── visualize/      # Visualization outputs
├── extract_interface_data.py    # Extract interface data from simulations
├── prepare_ml_data.py          # Prepare data for ML training
├── train_ml_models.py          # Train ML models
├── predict.py                  # Make predictions with trained models
└── visualize_model_comparison.py # Visualize model comparisons
```

## Prerequisites

-   Python 3.8 or higher
-   Required Python packages (install using `pip install -r requirements.txt`):
    -   numpy
    -   pandas
    -   scikit-learn
    -   matplotlib
    -   seaborn
    -   joblib

## Workflow

### 1. Data Generation (Simulations)

1. Run semiconductor device simulations using your preferred simulation tool
2. Ensure simulations output data in a format compatible with the extraction script
3. Place simulation results in the `data/` directory

### 2. Extract Interface Data

Run the interface data extraction script:

```bash
python extract_interface_data.py
```

This script:

-   Reads simulation data from the `data/` directory
-   Extracts interface-related parameters
-   Handles missing values by padding with zeros
-   Saves processed data to `interface_data_padded.csv`

### 3. Prepare Data for Machine Learning

Run the data preparation script:

```bash
python prepare_ml_data.py
```

This script:

-   Loads the padded interface data
-   Splits data into training and testing sets
-   Performs feature scaling
-   Saves prepared data as numpy arrays:
    -   `X_train.npy`
    -   `X_test.npy`
    -   `y_train.npy`
    -   `y_test.npy`

### 4. Train Machine Learning Models

Run the training script:

```bash
python train_ml_models.py
```

This script:

-   Trains three different models for each target variable (IntSRHn and IntSRHp):
    -   Random Forest
    -   Gradient Boosting
    -   Linear Regression
-   Evaluates model performance using:
    -   RMSE (Root Mean Square Error)
    -   MAE (Mean Absolute Error)
    -   R² Score
    -   Prediction accuracy metrics
-   Saves trained models to the `models/` directory:
    -   `RandomForest_IntSRHn.joblib`
    -   `GradientBoosting_IntSRHn.joblib`
    -   `LinearRegression_IntSRHn.joblib`
    -   `RandomForest_IntSRHp.joblib`
    -   `GradientBoosting_IntSRHp.joblib`
    -   `LinearRegression_IntSRHp.joblib`

### 5. Make Predictions

Run the prediction script:

```bash
python predict.py
```

This script:

-   Loads trained models
-   Makes predictions for both IntSRHn and IntSRHp
-   Validates predictions against test data
-   Saves validation metrics to `results/predict/model_validation_metrics.csv`
-   Generates accuracy plots in `results/predict/`

### 6. Visualize Model Comparisons

Run the visualization script:

```bash
python visualize_model_comparison.py
```

This script:

-   Reads validation metrics from the CSV file
-   Generates comparison plots:
    -   Combined model comparison (`results/visualize/model_comparison.png`)
    -   Individual target comparisons:
        -   `results/visualize/IntSRHn_comparison.png`
        -   `results/visualize/IntSRHp_comparison.png`

## Model Performance

The models are evaluated based on:

-   Mean Accuracy
-   Median Accuracy
-   Percentage of predictions within accuracy thresholds (70%, 80%, 90%)
-   R² Score

## Output Files

### Results Directory Structure

```
results/
├── predict/
│   ├── model_validation_metrics.csv
│   ├── validation_IntSRHn.png
│   ├── validation_IntSRHp.png
│   ├── accuracy_distribution_IntSRHn.png
│   └── accuracy_distribution_IntSRHp.png
└── visualize/
    ├── model_comparison.png
    ├── IntSRHn_comparison.png
    └── IntSRHp_comparison.png
```

## Notes

-   The Random Forest model typically performs best for both IntSRHn and IntSRHp predictions
-   All models are saved in the `models/` directory for future use
-   Validation metrics and visualizations are automatically generated and saved
-   The workflow is designed to be modular and can be extended with additional models or features

## Troubleshooting

If you encounter any issues:

1. Ensure all required packages are installed
2. Check that simulation data is in the correct format
3. Verify that all required directories exist
4. Check the logs for specific error messages

## Future Improvements

-   Add support for more ML models
-   Implement cross-validation
-   Add hyperparameter tuning
-   Include feature importance analysis
-   Add support for ensemble methods
