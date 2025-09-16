# Solar Cell Optimization Workflow

## Overview

This project implements a **machine learning-driven optimization pipeline** to find the optimal electron recombination rates (IntSRHn) and device parameters that maximize solar cell efficiency. The workflow combines physics-based simulations with multi-target machine learning to predict and optimize solar cell performance.

## Key Features

-   **Enhanced Simulation Generation**: Captures both recombination rates and efficiency metrics
-   **Multi-Target ML Models**: Predicts efficiency, recombination, and optimal parameters
-   **SHAP Analysis**: Comprehensive feature importance analysis using SHAP values
-   **Global Optimization**: Uses multiple optimization algorithms to find optimal configurations
-   **Physics Validation**: Ensures results are physically meaningful
-   **Comprehensive Reporting**: Generates detailed reports and visualizations
-   **Advanced Visualization**: Dashboard showing optimization results and model performance

## Workflow Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Enhanced      │    │  Multi-Target    │    │   Optimization  │
│  Simulations    │───▶│   ML Models      │───▶│   Pipeline      │
│                 │    │                  │    │                 │
│ • Efficiency    │    │ • Efficiency     │    │ • Find optimal  │
│ • Recombination │    │ • Recombination  │    │   recombination │
│ • Device params │    │ • Inverse models │    │ • Optimize      │
└─────────────────┘    └──────────────────┘    │   parameters    │
                                               └─────────────────┘
```

## Target Parameters

### Primary Optimization Targets

-   **Optimal Efficiency**: Maximum power point (MPP) in W/cm²
-   **Optimal Recombination**: Mean electron interfacial recombination rate (IntSRHn_mean)
-   **Optimal Device Parameters**: Layer thicknesses, energy levels, doping concentrations

### Device Parameters (Optimization Variables)

```python
# Layer 1 (PCBM - Electron Transport Layer)
L1_L: Layer thickness (20-50 nm)
L1_E_c: Conduction band energy (3.7-4.0 eV)
L1_E_v: Valence band energy (5.7-5.9 eV)
L1_N_D: Donor concentration (1E20-1E21 m⁻³)
L1_N_A: Acceptor concentration (1E20-1E21 m⁻³)

# Layer 2 (MAPI - Active Layer)
L2_L: Layer thickness (200-500 nm)
L2_E_c: Conduction band energy (4.4-4.6 eV)
L2_E_v: Valence band energy (5.6-5.8 eV)
L2_N_D: Donor concentration (1E20-1E21 m⁻³)
L2_N_A: Acceptor concentration (1E20-1E21 m⁻³)

# Layer 3 (PEDOT - Hole Transport Layer)
L3_L: Layer thickness (20-50 nm)
L3_E_c: Conduction band energy (3.4-3.6 eV)
L3_E_v: Valence band energy (5.3-5.5 eV)
L3_N_D: Donor concentration (1E20-1E21 m⁻³)
L3_N_A: Acceptor concentration (1E20-1E21 m⁻³)
```

## Quick Start

### 1. Run Complete Workflow

```bash
python run_optimization_workflow.py
```

This will run the entire optimization pipeline:

1. Define feature structure and parameter bounds
2. Generate enhanced simulations
3. Train optimization models
4. Run optimization algorithms
5. Generate reports and visualizations

### 2. Run ML and Optimization Only

```bash
python run_ml_optimization_workflow.py
```

This runs only the ML and optimization portion (scripts 4-8), assuming data preparation is complete:

1. Prepare ML data (derived features, cleaning, splits)
2. Train optimization models (efficiency & recombination)
3. Run optimization (find optimal device parameters)
4. Make predictions (validate and predict)
5. Visualize results (comprehensive dashboard)

**Prerequisites:** Scripts 1-3 must be run first.

### 3. Important Note on Target Variables

The optimization focuses on two key targets:

-   **MPP (Maximum Power Point)**: The fundamental efficiency metric in W/cm²
-   **IntSRHn_mean**: Mean electron interfacial recombination rate

PCE can be calculated from MPP when needed: `PCE = (MPP / 1000) × 100`

### 4. Run Individual Steps

#### Step 1: Define Feature Structure

```bash
python scripts/1_create_feature_names.py
```

-   Defines all feature names and parameter bounds
-   Reads bounds from `sim/parameters.txt` for consistency
-   Creates centralized feature definitions for all scripts
-   Output: `results/feature_definitions.json`

#### Step 2: Generate Enhanced Simulations

```bash
python scripts/2_generate_simulations_enhanced.py
```

-   Runs physics simulations for all parameter combinations
-   **Does NOT extract or combine data** (see next step)
-   Output: simulation result files in `sim/simulations/`

#### Step 3: Extract Simulation Data

```bash
python scripts/3_extract_simulation_data.py
```

-   Extracts device parameters, efficiency metrics, and recombination data from simulation outputs
-   Combines all data into a single CSV for ML
-   Output: `results/generate_enhanced/combined_output_with_efficiency.csv`

#### Step 4: Prepare ML Data

```bash
python scripts/4_prepare_ml_data.py [--remove-outliers] [--enhanced-features]
```

-   Creates enhanced physics-based derived features from primary parameters
-   Validates physics constraints (energy gaps, thicknesses, doping concentrations)
-   Handles missing values and outliers with enhanced detection
-   Prepares train/test splits for ML models
-   Creates inverse optimization dataset for parameter prediction
-   Output: `results/prepare_ml_data/`

**Enhanced Features Added:**

-   Physics-based features: recombination_efficiency_ratio, interface_quality_index
-   Carrier transport features: conduction_band_alignment_quality, valence_band_alignment_quality
-   Thickness optimization: thickness_balance_quality, transport_layer_balance
-   Doping optimization: average_doping_ratio, doping_consistency
-   Energy level features: energy_gap_progression, energy_gap_uniformity

#### Step 5: Train Optimization Models

```bash
python scripts/5_train_optimization_models.py
```

-   Trains efficiency prediction models
-   Trains recombination prediction models
-   Performs SHAP analysis for feature importance
-   Output: `results/train_optimization_models/models/`

#### Step 6: Run Optimization

```bash
python scripts/6_optimize_efficiency.py
```

-   Finds optimal device parameters
-   Predicts optimal recombination rates
-   Validates results with physics constraints
-   Output: `results/optimize_efficiency/reports/optimization_report.json`

#### Step 7: Make Predictions

```bash
python scripts/7_predict.py
```

-   Makes predictions using trained optimization models
-   Validates predictions against experimental data
-   Output: `results/predict/`

#### Step 8: Visualize Results

```bash
python scripts/8_visualize_example_fixed.py
```

-   Creates comprehensive dashboard of optimization results
-   Shows model performance and feature importance
-   Output: `results/visualize/`

## Output Files

### Enhanced Simulation Data

-   **Location**: `results/generate_enhanced/combined_output_with_efficiency.csv`
-   **Contains**: Device parameters, efficiency metric (MPP), recombination rate (IntSRHn_mean)
-   **How to generate**: Run `2_generate_simulations_enhanced.py` to produce simulation outputs, then run `3_extract_simulation_data.py` to extract and combine the data

### ML Data Preparation

-   **Location**: `results/prepare_ml_data/`
-   **Files**:
    -   `X_train_efficiency.csv`: Training features for efficiency prediction
    -   `X_test_efficiency.csv`: Test features for efficiency prediction
    -   `y_train_efficiency.csv`: Training targets for efficiency prediction
    -   `y_test_efficiency.csv`: Test targets for efficiency prediction
    -   `X_train_recombination.csv`: Training features for recombination prediction
    -   `X_test_recombination.csv`: Test features for recombination prediction
    -   `y_train_recombination.csv`: Training targets for recombination prediction
    -   `y_test_recombination.csv`: Test targets for recombination prediction
    -   `X_train_inverse.csv`: Training features for inverse optimization
    -   `X_test_inverse.csv`: Test features for inverse optimization
    -   `y_train_inverse.csv`: Training targets for inverse optimization
    -   `y_test_inverse.csv`: Test targets for inverse optimization
    -   `X_full.csv`: Full feature dataset for optimization
    -   `y_efficiency_full.csv`: Full efficiency targets
    -   `y_recombination_full.csv`: Full recombination targets
    -   `dataset_metadata.json`: Enhanced dataset information and statistics
-   **Enhanced Features**: Physics-based features for recombination-efficiency relationship, carrier transport optimization, and device parameter optimization

### Trained Models

-   **Location**: `results/train_optimization_models/models/`
-   **Models**:
    -   `efficiency_MPP.joblib`: Predicts MPP from device parameters
    -   `efficiency_Jsc.joblib`: Predicts Jsc from device parameters
    -   `recombination_IntSRHn_mean.joblib`: Predicts recombination from device parameters
    -   `inverse_MPP_*.joblib`: Predicts optimal device parameters for target efficiency
-   **SHAP Analysis**:
    -   `shap_summary_efficiency.png`: SHAP summary plot for efficiency prediction
    -   `shap_importance_efficiency.png`: Feature importance from SHAP analysis
    -   `shap_values_efficiency.csv`: Raw SHAP values for further analysis

### Optimization Results

-   **Location**: `results/optimize_efficiency/`
-   **Files**:
    -   `reports/optimization_report.json`: Comprehensive optimization results
    -   `plots/optimization_results.png`: Visualization of optimal parameters
    -   `plots/optimization_methods.png`: Comparison of optimization algorithms

### Visualization Results

-   **Location**: `results/visualize/`
-   **Files**:
    -   `comprehensive_dashboard.png`: Complete dashboard of all results
    -   `optimal_parameters.png`: Visualization of optimal device parameters
    -   `efficiency_vs_recombination_optimal.png`: Trade-off analysis
    -   `model_performance_comparison.png`: Model performance metrics
    -   `shap_importance_efficiency.png`: SHAP feature importance plots

## Key Insights

### 1. Optimal Recombination is NOT Zero

The optimization reveals that **minimum recombination does not always lead to maximum efficiency**. There's often an optimal recombination rate that balances:

-   Carrier transport efficiency
-   Interface quality
-   Energy level alignment

### 2. Layer Thickness Optimization

-   **PCBM Layer**: Optimal thickness balances electron transport vs. recombination
-   **MAPI Layer**: Thickness affects light absorption and bulk recombination
-   **PEDOT Layer**: Thickness influences hole transport and interface recombination

### 3. Energy Level Engineering

-   **Conduction Band**: Affects electron injection and recombination barriers
-   **Valence Band**: Influences hole transport and recombination rates
-   **Band Alignment**: Critical for efficient carrier transport

### 4. Doping Concentration Effects

-   **Donor Concentration**: Affects electron transport and recombination
-   **Acceptor Concentration**: Influences hole transport and recombination
-   **Optimal Balance**: Required for efficient device operation

## Optimization Algorithms

### 1. L-BFGS-B (Local Optimization)

-   **Purpose**: Fine-tune parameters near good initial guesses
-   **Advantages**: Fast convergence, handles constraints
-   **Use Case**: Refinement of parameter values

### 2. Differential Evolution (Global Optimization)

-   **Purpose**: Explore the entire parameter space
-   **Advantages**: Finds global optima, robust to local minima
-   **Use Case**: Initial parameter discovery

### 3. Constraint Handling

-   **Recombination Constraint**: Limits maximum recombination rate
-   **Parameter Bounds**: Ensures physically meaningful values
-   **Validation**: Checks results against physics constraints

## Model Performance

### Efficiency Prediction

-   **Target**: MPP (Maximum Power Point) in W/cm²
-   **Accuracy**: R² > 0.8 typically achieved
-   **Features**: All device parameters (thickness, energy levels, doping)

### Recombination Prediction

-   **Target**: IntSRHn_mean (Mean electron interfacial recombination rate)
-   **Accuracy**: R² > 0.7 typically achieved
-   **Features**: Device parameters affecting interface recombination

### Inverse Optimization

-   **Target**: Device parameters for given efficiency
-   **Purpose**: Find optimal configurations
-   **Method**: Trained on high-efficiency configurations

## Validation and Physics

### 1. Parameter Bounds Validation

-   Ensures all parameters are within physically reasonable ranges
-   Checks against material property limits
-   Validates against experimental constraints

### 2. Recombination Constraint

-   Limits maximum recombination rate to prevent excessive losses
-   Ensures device remains functional
-   Balances efficiency vs. recombination

### 3. Efficiency Prediction Validation

-   Cross-validates ML predictions with physics simulations
-   Checks for prediction consistency
-   Validates optimization results

## Recommendations

### For High Efficiency

1. **Optimize Layer Thicknesses**: Balance transport vs. recombination
2. **Engineer Energy Levels**: Ensure proper band alignment
3. **Control Doping**: Find optimal carrier concentrations
4. **Monitor Recombination**: Keep within optimal range

### For Manufacturing

1. **Parameter Sensitivity**: Identify critical parameters
2. **Tolerance Analysis**: Understand parameter variations
3. **Robust Design**: Choose parameter ranges for stability
4. **Cost Optimization**: Balance performance vs. cost

## Troubleshooting

### Common Issues

1. **Simulation Failures**

    - Check parameter bounds in `sim/parameters.txt`
    - Verify simulation setup in `sim/simulation_setup.txt`
    - Review logs in `results/generate_enhanced/simulation_enhanced.log`

2. **Model Training Issues**

    - Ensure sufficient data points (>100 recommended)
    - Check for missing values in enhanced data
    - Verify feature scaling and preprocessing

3. **Optimization Failures**
    - Adjust parameter bounds if too restrictive
    - Modify recombination constraints if too strict
    - Try different optimization methods

### Performance Tips

1. **Data Quality**: Ensure high-quality simulation data
2. **Feature Engineering**: Consider adding interaction features
3. **Model Selection**: Try different ML algorithms
4. **Hyperparameter Tuning**: Optimize model parameters

## Future Improvements

### 1. Advanced Optimization

-   Multi-objective optimization (efficiency + stability)
-   Bayesian optimization for better exploration
-   Constraint relaxation for broader search

### 2. Enhanced Models

-   Deep learning models for better accuracy
-   Ensemble methods for robust predictions
-   Uncertainty quantification for predictions

### 3. Extended Physics

-   Temperature effects on recombination
-   Light intensity dependence
-   Degradation modeling

### 4. Manufacturing Integration

-   Process variation modeling
-   Cost optimization
-   Yield prediction

## Contact and Support

For questions or issues with the optimization workflow:

1. Check the log files in `logs/` directory
2. Review the documentation in each script
3. Examine the output files for detailed results

The optimization workflow provides a comprehensive approach to finding optimal solar cell configurations that balance efficiency with practical constraints.
