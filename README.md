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
├── extracted_data.py    # Extract interface data from simulations
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
python extracted_data.py
```

This script:

-   Reads simulation data from the `data/` directory
-   Extracts interface-related parameters
-   Handles missing values by padding with zeros
-   Saves processed data to `interface_extracted_data.csv`

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

# Solar Cell Simulation Project

This project uses SimSS (version 5.22) to simulate solar cell performance with different material parameters and configurations, followed by machine learning analysis of the results.

## Project Structure

```
myML/
├── sim/
│   ├── simulations/          # Simulation outputs (gitignored)
│   ├── Data/                 # Material data files
│   ├── parameters.txt        # Parameter ranges for simulations
│   ├── simulation_setup.txt  # Simulation configuration
│   └── generate_simulations.py # Script to generate and run simulations
├── scripts/                  # Machine learning and analysis scripts
│   ├── extracted_data.py    # Extract interface data from simulation results
│   ├── prepare_ml_data.py          # Prepare and preprocess data for ML models
│   ├── train_ml_models.py          # Train and evaluate ML models
│   ├── visualize_model_comparison.py # Visualize and compare model performance
│   └── predict.py                   # Make predictions using trained models
├── run_all.py               # Master script to run multiple simulation scripts
├── .gitignore
└── README.md
```

## Setup

1. Ensure Python 3.x is installed
2. Create and activate virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate     # Windows
    ```
3. Install required packages:
    ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn
    ```

## Simulation Parameters

The simulation uses a three-layer structure:

1. Layer 1 (PCBM): Electron Transport Layer

    - Band energies: E_c = 3.7-4.0 eV, E_v = 5.7-5.9 eV
    - Thickness: 20-50 nm

2. Layer 2 (MAPI): Active Layer

    - Band energies: E_c = 4.4-4.6 eV, E_v = 5.6-5.8 eV
    - Thickness: 200-500 nm

3. Layer 3 (PEDOT): Hole Transport Layer
    - Band energies: E_c = 3.4-3.6 eV, E_v = 5.3-5.5 eV
    - Thickness: 20-50 nm

## Running Simulations

### Option 1: Run Individual Scripts

1. Configure parameters in `sim/parameters.txt`
2. Adjust simulation settings in `sim/simulation_setup.txt`
3. Run the simulation generator:
    ```bash
    python sim/generate_simulations.py
    ```

### Option 2: Run All Scripts

Use the master script to run multiple simulation scripts in sequence:

```bash
python run_all.py
```

The master script will:

-   Run all configured scripts in order
-   Log output to both console and a timestamped log file
-   Provide a summary of successful/failed scripts

The script will:

-   Generate parameter combinations
-   Create simulation directories
-   Run simulations
-   Combine results into `simulations/combined_output.csv`

## Machine Learning Pipeline

The project includes a complete ML pipeline for analyzing simulation results:

1. **Data Extraction** (`extracted_data.py`):

    - Extracts interface data from simulation results
    - Processes raw simulation outputs
    - Creates structured datasets for ML

2. **Data Preparation** (`prepare_ml_data.py`):

    - Preprocesses extracted data
    - Handles missing values
    - Performs feature engineering
    - Splits data into training and testing sets

3. **Model Training** (`train_ml_models.py`):

    - Trains multiple ML models
    - Performs cross-validation
    - Evaluates model performance
    - Saves trained models

4. **Visualization** (`visualize_model_comparison.py`):

    - Compares model performance
    - Generates performance plots
    - Creates feature importance visualizations
    - Saves analysis results

5. **Prediction** (`predict.py`):
    - Loads trained models
    - Makes predictions on new data
    - Generates prediction reports

## Output

-   Individual simulation results are stored in `sim/simulations/sim_XXXX/`
-   Combined results are saved in `sim/simulations/combined_output.csv`
-   ML model outputs and visualizations are saved in `scripts/output/`
-   Log file contains simulation progress and errors

## Notes

-   Maximum number of simulations is set to 20 (adjustable in `generate_simulations.py`)
-   Non-converging simulations (return code 95) are considered successful
-   The simulations directory is gitignored to avoid committing large output files
-   ML models and their outputs are version controlled for reproducibility

## Data Preparation

Before training the models, the following steps are taken to prepare the data:

1. **Data Cleaning**:

    - Handle missing values.
    - Remove duplicate rows.
    - Detect and handle outliers.

2. **Data Transformation**:

    - Normalize or standardize features.
    - Apply log transformations to skewed data.
    - Encode categorical variables.

3. **Feature Engineering**:

    - Create new features from existing ones.
    - Select relevant features using techniques like correlation analysis.

4. **Data Splitting**:

    - Split the dataset into training and testing sets.
    - Use cross-validation for robust model evaluation.

5. **Data Augmentation**:

    - Generate synthetic data if necessary.

6. **Handling Imbalanced Data**:

    - Use resampling techniques to balance classes.

7. **Data Validation**:

    - Ensure data integrity and consistency.

8. **Documentation and Logging**:

    - Log data preparation steps for reproducibility.

9. **Exploratory Data Analysis (EDA)**:

    - Visualize data to understand distributions and relationships.

10. **Data Privacy and Security**:
    - Anonymize sensitive information.

## Usage

To prepare the data for training, run:

```bash
python scripts/prepare_ml_data.py
```

## Training Models

To train the models, run:

```bash
python scripts/train_ml_models.py
```

## Making Predictions

To make predictions using the trained models, run:

```bash
python scripts/predict.py
```

## Logs

Logs are stored in the `logs` directory for tracking the data preparation and model training processes.
