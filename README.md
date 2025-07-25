# Solar Cell Optimization Workflow

## Overview

This project implements a **machine learning-driven optimization pipeline** to find the optimal electron recombination rates (IntSRHn) and device parameters that maximize solar cell efficiency. The workflow combines physics-based simulations with multi-target machine learning to predict and optimize solar cell performance.

## Key Features

-   **Enhanced Simulation Generation**: Captures both recombination rates and efficiency metrics
-   **Multi-Target ML Models**: Predicts efficiency, recombination, and optimal parameters
-   **Global Optimization**: Uses multiple optimization algorithms to find optimal configurations
-   **Physics Validation**: Ensures results are physically meaningful
-   **Comprehensive Reporting**: Generates detailed reports and visualizations

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

-   **Optimal IntSRHn**: Electron recombination rate for maximum efficiency
-   **Optimal Efficiency**: Predicted maximum power point (MPP)
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

1. Generate enhanced simulations
2. Train optimization models
3. Run optimization algorithms
4. Generate reports and visualizations

### 2. Important Note on PCE Calculation

The Power Conversion Efficiency (PCE) is calculated as:

```
PCE = (MPP / 1000 W/m²) × 100
```

where MPP is the Maximum Power Point in W/m² and 1000 W/m² is the standard AM1.5G solar irradiance.

### 2. Run Individual Steps

#### Step 1: Generate Enhanced Simulations

```bash
python scripts/2_generate_simulations_enhanced.py
```

-   Runs physics simulations for all parameter combinations
-   **Does NOT extract or combine data** (see next step)
-   Output: simulation result files in `sim/simulations/`

#### Step 2: Extract Simulation Data

```bash
python scripts/3_extract_simulation_data.py
```

-   Extracts device parameters, efficiency metrics, and recombination data from simulation outputs
-   Combines all data into a single CSV for ML
-   Output: `results/generate_enhanced/combined_output_with_efficiency.csv`

#### Step 3: Train Optimization Models

```bash
python scripts/5_train_optimization_models.py
```

-   Trains efficiency prediction models
-   Trains recombination prediction models
-   Trains inverse optimization models
-   Output: `results/train_optimization_models/models/`

#### Step 4: Run Optimization

```bash
python scripts/6_optimize_efficiency.py
```

-   Finds optimal device parameters
-   Predicts optimal recombination rates
-   Validates results with physics constraints
-   Output: `results/optimize_efficiency/reports/optimization_report.json`

## Output Files

### Enhanced Simulation Data

-   **Location**: `results/generate_enhanced/combined_output_with_efficiency.csv`
-   **Contains**: Device parameters, efficiency metrics (MPP, Jsc, Voc, FF), recombination rates
-   **How to generate**: Run `2_generate_simulations_enhanced.py` to produce simulation outputs, then run `3_extract_simulation_data.py` to extract and combine the data

### Trained Models

-   **Location**: `results/train_optimization_models/models/`
-   **Models**:
    -   `efficiency_MPP.joblib`: Predicts MPP from device parameters
    -   `efficiency_Jsc.joblib`: Predicts Jsc from device parameters
    -   `recombination_IntSRHn_mean.joblib`: Predicts recombination from device parameters
    -   `inverse_MPP_*.joblib`: Predicts optimal device parameters for target efficiency

### Optimization Results

-   **Location**: `results/optimize_efficiency/`
-   **Files**:
    -   `reports/optimization_report.json`: Comprehensive optimization results
    -   `plots/optimization_results.png`: Visualization of optimal parameters
    -   `plots/optimization_methods.png`: Comparison of optimization algorithms

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

-   **Target**: MPP (Maximum Power Point)
-   **Accuracy**: R² > 0.8 typically achieved
-   **Features**: All device parameters (thickness, energy levels, doping)

### Recombination Prediction

-   **Target**: IntSRHn_mean (Average electron recombination)
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
