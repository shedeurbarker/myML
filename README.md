# Solar Cell Optimization Workflow

## Overview

This project implements a **machine learning-driven optimization pipeline** to find optimal solar cell device parameters that maximize efficiency (MPP) while controlling electron interfacial recombination rates (IntSRHn_mean). The workflow combines physics-based simulations with machine learning to predict and optimize solar cell performance.

## Key Features

-   **Focused Target Optimization**: Specifically targets MPP (Maximum Power Point) and IntSRHn_mean (mean electron interfacial recombination rate)
-   **15-Parameter Device Optimization**: Optimizes layer thicknesses, energy levels, and doping concentrations across 3 device layers
-   **Physics-Based Simulations**: Uses detailed drift-diffusion simulations for accurate device modeling
-   **Enhanced Feature Engineering**: Creates physics-informed derived features from primary parameters
-   **ML-Ready Data Pipeline**: Automated data extraction, cleaning, and preparation for machine learning
-   **Comprehensive Logging**: Detailed tracking of all workflow steps with progress monitoring

## Device Architecture

The solar cell consists of three layers with 5 parameters each (15 total optimization variables):

```
┌─────────────────────────────────────────────────────┐
│                    Layer 3 (PEDOT)                  │  ← Hole Transport Layer
│                 Hole Transport Layer                 │    (5 parameters: L, E_c, E_v, N_D, N_A)
├─────────────────────────────────────────────────────┤
│                    Layer 2 (MAPI)                   │  ← Active Layer
│                    Active Layer                      │    (5 parameters: L, E_c, E_v, N_D, N_A)
├─────────────────────────────────────────────────────┤
│                    Layer 1 (PCBM)                   │  ← Electron Transport Layer
│                Electron Transport Layer              │    (5 parameters: L, E_c, E_v, N_D, N_A)
└─────────────────────────────────────────────────────┘
```

## Optimization Variables (15 Primary Parameters)

### Layer 1 (PCBM - Electron Transport Layer)

-   `L1_L`: Layer thickness (20-50 nm)
-   `L1_E_c`: Conduction band energy (3.7-4.0 eV)
-   `L1_E_v`: Valence band energy (5.7-5.9 eV)
-   `L1_N_D`: Donor concentration (1E20-1E21 m⁻³)
-   `L1_N_A`: Acceptor concentration (1E20-1E21 m⁻³)

### Layer 2 (MAPI - Active Layer)

-   `L2_L`: Layer thickness (200-500 nm)
-   `L2_E_c`: Conduction band energy (4.4-4.6 eV)
-   `L2_E_v`: Valence band energy (5.6-5.8 eV)
-   `L2_N_D`: Donor concentration (1E20-1E21 m⁻³)
-   `L2_N_A`: Acceptor concentration (1E20-1E21 m⁻³)

### Layer 3 (PEDOT - Hole Transport Layer)

-   `L3_L`: Layer thickness (20-50 nm)
-   `L3_E_c`: Conduction band energy (3.4-3.6 eV)
-   `L3_E_v`: Valence band energy (5.3-5.5 eV)
-   `L3_N_D`: Donor concentration (1E20-1E21 m⁻³)
-   `L3_N_A`: Acceptor concentration (1E20-1E21 m⁻³)

## Target Variables

### Primary Targets (Defined in Script 1)

-   **MPP**: Maximum Power Point (W/cm²) - efficiency optimization target
-   **IntSRHn_mean**: Mean electron interfacial recombination rate - recombination control target

## Workflow Steps

### Step 1: Define Feature Structure

```bash
python scripts/1_create_feature_names.py
```

**Purpose**: Establishes the foundation for the entire ML pipeline by defining all features and parameters.

**What it does**:

-   Defines the 15 primary device parameters (optimization variables)
-   Specifies the 2 target variables (MPP and IntSRHn_mean)
-   Loads parameter bounds from `sim/parameters.txt` for optimization constraints
-   Creates centralized feature definitions for consistency across all scripts

**Output files**:

-   `results/features/feature_definitions.json` - Complete feature definitions
-   `results/features/primary_parameters.json` - List of the 15 optimization variables
-   `results/features/parameter_bounds.json` - Min/max bounds for each parameter
-   `results/features/target_variables.json` - Target variables (MPP and IntSRHn_mean)

### Step 2: Generate Physics-Validated Simulations

```bash
python scripts/2_generate_simulations.py
```

**Purpose**: Runs physics-based drift-diffusion simulations with comprehensive solar cell physics validation to ensure only realistic device structures are simulated.

**What it does**:

-   Reads physics-compliant parameter ranges from `sim/parameters.txt`
-   Generates parameter combinations (grid sampling or random sampling)
-   **VALIDATES COMPREHENSIVE SOLAR CELL PHYSICS** before running simulations
-   Rejects unphysical combinations based on rigorous device physics constraints
-   **ENFORCES ELECTRODE WORK FUNCTION COMPATIBILITY** during parameter validation
-   Creates simulation directories only for validated parameter sets
-   Executes physics simulations (`simss.exe`) for valid configurations only
-   Logs validation statistics and simulation progress

**Comprehensive Physics Validation**:

### 1. Energy Band Alignment

-   **Electron Transport**: ETL conduction band ≥ Active layer conduction band (downhill energy cascade)
-   **Hole Transport**: HTL valence band ≤ Active layer valence band (downhill energy cascade)
-   **Internal Consistency**: Positive bandgaps (1.0-4.0 eV) for all layers

### 2. Doping and Carrier Type

-   **ETL (L1)**: n-type semiconductor (N_D >> N_A, ratio ≥ 10:1)
-   **Active Layer (L2)**: Intrinsic/undoped (very low doping < 1e18 m⁻³)
-   **HTL (L3)**: p-type semiconductor (N_A >> N_D, ratio ≥ 10:1)

### 3. Layer Thickness

-   **Transport Layers (ETL/HTL)**: 10-50 nm (thin for low resistance)
-   **Active Layer**: 300-600 nm (thick for light absorption)

### 4. Electrode Work Function Compatibility

-   **Left Electrode (W_L)**: Fixed at 4.05 eV (ITO electrode)
-   **Right Electrode (W_R)**: Fixed at 5.2 eV (Au electrode)
-   **Physics Constraint**: Parameter ranges constrained to ensure W_L ≥ E_c(ETL) and W_R ≤ E_v(HTL)
-   **Implementation**: Validation during parameter generation rejects incompatible combinations

**Device Structure**: ETL/Active/HTL (PCBM/MAPI/PEDOT:PSS)

-   L1: Electron Transport Layer (PCBM) - n-type
-   L2: Active Layer (MAPI Perovskite) - intrinsic
-   L3: Hole Transport Layer (PEDOT:PSS) - p-type

**Input files**:

-   `sim/parameters.txt` - Parameter ranges and bounds
-   `sim/simulation_setup.txt` - Simulation configuration
-   `sim/L1_parameters.txt`, `sim/L2_parameters.txt`, `sim/L3_parameters.txt` - Layer configurations
-   `sim/Data/` - Material optical and electrical properties
-   `sim/simss.exe` - Physics simulation executable

**Output files**:

-   `sim/simulations/sim_XXXX/` - Individual simulation result folders
-   `results/generated_simulations/generated_simulations.log` - Simulation log

**Important**: This script only runs simulations. Data extraction happens in Step 3.

### Step 3: Extract Simulation Data

```bash
python scripts/3_extract_simulation_data.py
```

**Purpose**: Extracts the essential data (device parameters + MPP + IntSRHn_mean) from simulation results.

**What it does**:

-   Processes each simulation directory in `sim/simulations/`
-   Extracts device parameters from `parameters.json`
-   Calculates MPP (Maximum Power Point) from J-V curves in `output_JV.dat`
-   Extracts IntSRHn_mean from recombination data in `output_Var.dat`
-   Combines all data into a single CSV file for machine learning
-   Handles failed simulations gracefully (skips and logs)

**Input files**:

-   `sim/simulations/sim_XXXX/output_JV.dat` - J-V curve data
-   `sim/simulations/sim_XXXX/output_Var.dat` - Recombination rate data
-   `sim/simulations/sim_XXXX/parameters.json` - Device parameters
-   `results/features/feature_definitions.json` - Feature definitions from Step 1

**Output files**:

-   `results/extract_simulation_data/extracted_simulation_data.csv` - Combined dataset
-   `results/extract_simulation_data/extraction.log` - Extraction log

**Data structure**: Each row contains 15 device parameters + MPP + IntSRHn_mean (17 columns total)

### Step 4: Prepare ML Data

```bash
python scripts/4_prepare_ml_data.py [--remove-outliers] [--enhanced-features]
```

**Purpose**: Transforms raw simulation data into ML-ready datasets with enhanced features and proper train/test splits.

**What it does**:

-   Loads extracted data from Step 3
-   Creates enhanced derived features from the 15 primary parameters
-   Handles missing values using median/mode imputation
-   Optionally removes outliers using IQR method (disabled by default)
-   Creates separate datasets for efficiency and recombination prediction
-   Splits data into 80/20 train/test sets
-   Saves ML-ready datasets with comprehensive metadata

**Enhanced derived features created**:

-   **Thickness features**: total_thickness, thickness_ratio_L2, thickness_ratio_ETL, thickness_ratio_HTL
-   **Energy gap features**: energy_gap_L1, energy_gap_L2, energy_gap_L3
-   **Band alignment features**: band_offset_L1_L2, band_offset_L2_L3, conduction_band_offset, valence_band_offset
-   **Doping features**: doping_ratio_L1, doping_ratio_L2, doping_ratio_L3, total_donor_concentration, total_acceptor_concentration
-   **Material property features**: average_energy_gap, energy_gap_variance, thickness_variance, doping_variance
-   **Physics-based features**: recombination_efficiency_ratio, interface_quality_index, carrier_transport_efficiency

**Input files**:

-   `results/extract_simulation_data/extracted_simulation_data.csv` - From Step 3
-   `results/features/feature_definitions.json` - From Step 1

**Output files**:

-   `results/prepare_ml_data/X_train_efficiency.csv` - Training features for efficiency prediction
-   `results/prepare_ml_data/X_test_efficiency.csv` - Test features for efficiency prediction
-   `results/prepare_ml_data/y_train_efficiency.csv` - Training targets (MPP) for efficiency prediction
-   `results/prepare_ml_data/y_test_efficiency.csv` - Test targets (MPP) for efficiency prediction
-   `results/prepare_ml_data/X_train_recombination.csv` - Training features for recombination prediction
-   `results/prepare_ml_data/X_test_recombination.csv` - Test features for recombination prediction
-   `results/prepare_ml_data/y_train_recombination.csv` - Training targets (IntSRHn_mean) for recombination prediction
-   `results/prepare_ml_data/y_test_recombination.csv` - Test targets (IntSRHn_mean) for recombination prediction
-   `results/prepare_ml_data/X_full.csv` - Complete feature dataset for optimization
-   `results/prepare_ml_data/y_efficiency_full.csv` - Complete efficiency targets
-   `results/prepare_ml_data/y_recombination_full.csv` - Complete recombination targets
-   `results/prepare_ml_data/dataset_metadata.json` - Dataset statistics and information

**Command line options**:

-   `--remove-outliers`: Enable outlier removal (disabled by default to preserve all data)
-   `--enhanced-features`: Enable enhanced physics-based features (enabled by default)

### Step 5: Train Machine Learning Models

```bash
python scripts/5_train_models.py
```

**Purpose**: Trains machine learning models to predict MPP and IntSRHn_mean from device parameters.

**What it does**:

-   Loads ML-ready data from Step 4
-   Trains models to predict MPP (Maximum Power Point) from device parameters
-   Trains models to predict IntSRHn_mean (mean electron interfacial recombination rate)
-   Uses robust scaling and 5-fold cross-validation for reliable model performance
-   Compares multiple algorithms and selects the best-performing models
-   Saves trained models and scalers for optimization use

**Algorithms used**:

-   **Random Forest Regressor**: Ensemble method, robust to outliers
-   **XGBoost Regressor**: Advanced gradient boosting (if XGBoost is installed)
-   **Gradient Boosting Regressor**: Fallback if XGBoost is unavailable

**Key features**:

-   RobustScaler for both features and targets (handles outliers)
-   5-fold cross-validation for model selection
-   Comprehensive evaluation metrics (R², MAE, RMSE)
-   Automatic best model selection for each target variable
-   **NEW**: Comprehensive training visualizations and performance analysis

**Training Visualizations Created**:

-   `model_performance_comparison.png` - Algorithm R² and error comparison charts
-   `detailed_metrics_heatmap.png` - Performance metrics heatmap for all algorithms
-   `cross_validation_stability.png` - Cross-validation stability analysis
-   `best_models_summary.png` - Best model selection summary with performance thresholds
-   `error_analysis.png` - Detailed error breakdown (MAE, RMSE, MAPE)
-   `training_summary_report.png` - Complete training report with methodology and results

**Input files**:

-   `results/prepare_ml_data/X_full.csv` - Features from Step 4
-   `results/prepare_ml_data/y_efficiency_full.csv` - MPP targets
-   `results/prepare_ml_data/y_recombination_full.csv` - IntSRHn_mean targets

**Output files**:

-   `results/train_optimization_models/models/efficiency_MPP_*.joblib` - MPP prediction models
-   `results/train_optimization_models/models/recombination_IntSRHn_mean_*.joblib` - Recombination models
-   `results/train_optimization_models/scalers/` - Feature and target scalers
-   `results/train_optimization_models/training_metadata.json` - Training statistics
-   `results/train_optimization_models/training.log` - Detailed training log
-   `results/train_optimization_models/plots/` - **NEW**: Comprehensive training visualizations

### Step 6: Predict and Validate Models

```bash
python scripts/6_predict.py
```

**Purpose**: Uses trained ML models to make predictions and validate model performance against experimental data.

**What it does**:

-   Loads trained MPP and IntSRHn_mean prediction models from Step 5
-   Creates example predictions using configurable device parameters
-   Validates model predictions against the full simulation dataset
-   Calculates comprehensive performance metrics (R², RMSE, MAE, MAPE)
-   Generates validation plots with proper number formatting
-   Provides model performance assessment for optimization readiness

**Key features**:

-   **Configurable Example Parameters**: Uses external JSON file for easy parameter adjustment
-   **Comprehensive Validation**: Tests models against 10,000+ device simulations
-   **Professional Visualizations**: Clean plots with scientific notation for large numbers
-   **Performance Metrics**: R², RMSE, MAE, and MAPE for both efficiency and recombination models
-   **Error Analysis**: Residual plots and statistical analysis

**Input files**:

-   `results/train_optimization_models/models/efficiency_MPP.joblib` - MPP prediction model
-   `results/train_optimization_models/models/recombination_IntSRHn_mean.joblib` - Recombination model
-   `results/train_optimization_models/models/*_scalers.joblib` - Feature and target scalers
-   `results/extract_simulation_data/extracted_simulation_data.csv` - Data for validation
-   `example_device_parameters.json` - **NEW**: Configurable example device parameters

**Output files**:

-   `results/predict/predictions.log` - Detailed execution log with parameter loading
-   `results/predict/efficiency_MPP_validation.png` - MPP prediction accuracy plot
-   `results/predict/efficiency_PCE_validation.png` - Power conversion efficiency plot
-   `results/predict/recombination_IntSRHn_mean_validation.png` - Recombination prediction plot

**Example device configuration** (`example_device_parameters.json`):

```json
{
    "device_type": "Perovskite Solar Cell (PCBM/MAPI/PEDOT)",
    "parameters": {
        "L1_L": 3.0e-8, // ETL thickness (30 nm)
        "L1_E_c": 3.8, // ETL conduction band (3.8 eV)
        "L2_L": 3.0e-7, // Active layer thickness (300 nm)
        "L2_E_c": 3.7, // Active conduction band (3.7 eV)
        "L3_E_v": 5.3 // HTL valence band (5.3 eV)
    }
}
```

**Validation Results** (typical performance on 10,000-device dataset):

-   **MPP Model**: R² = 80.46%, MAE = 14.7% (good predictive power)
-   **Recombination Model**: R² = 83.92%, MAE = 14.8% (good predictive power)

## Quick Start

### Run Complete Data Pipeline

```bash
# Step 1: Define features and parameters
python scripts/1_create_feature_names.py

# Step 2: Generate physics simulations
python scripts/2_generate_simulations.py

# Step 3: Extract simulation data
python scripts/3_extract_simulation_data.py

# Step 4: Prepare ML-ready datasets
python scripts/4_prepare_ml_data.py --enhanced-features

# Step 5: Train machine learning models
python scripts/5_train_models.py

# Step 6: Predict and validate models
python scripts/6_predict.py
```

### Verify Results

After running all steps, you should have:

-   `results/extract_simulation_data/extracted_simulation_data.csv` - Raw simulation data (15 parameters + 2 targets)
-   `results/prepare_ml_data/` - ML-ready datasets with enhanced features
-   `results/train_optimization_models/models/` - Trained ML models for MPP and IntSRHn_mean prediction
-   `results/train_optimization_models/scalers/` - Feature and target scalers
-   `results/train_optimization_models/plots/` - Training performance visualizations
-   `results/predict/` - Model validation results and prediction plots
-   `example_device_parameters.json` - Configurable example device parameters
-   Comprehensive logs in each results subdirectory

## Data Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Script 1       │    │  Script 2        │    │  Script 3       │    │  Script 4        │    │  Script 5        │    │  Script 6        │
│                 │    │                  │    │                 │    │                  │    │                  │    │                  │
│ Define Features │───▶│ Generate Physics │───▶│ Extract Data    │───▶│ Prepare ML Data  │───▶│ Train ML Models  │───▶│ Predict & Validate│
│                 │    │  Simulations     │    │                 │    │                  │    │                  │    │                  │
│ • 15 parameters │    │ • Parameter      │    │ • MPP           │    │ • Enhanced       │    │ • MPP predictor  │    │ • Example preds  │
│ • 2 targets     │    │   combinations   │    │ • IntSRHn_mean  │    │   features       │    │ • IntSRHn_mean   │    │ • Model validation│
│ • Bounds        │    │ • Physics sims   │    │ • Device params │    │ • Train/test     │    │   predictor      │    │ • Performance    │
└─────────────────┘    └──────────────────┘    └─────────────────┘    │   splits         │    │ • Scalers        │    │   plots          │
                                                                      └──────────────────┘    │ • Visualizations │    └──────────────────┘
                                                                                              └──────────────────┘
```

## Key Insights

### 1. Focused Target Optimization

The workflow specifically targets two key metrics:

-   **MPP (Maximum Power Point)**: Direct measure of device efficiency
-   **IntSRHn_mean**: Controls interfacial recombination losses

### 2. Physics-Informed Feature Engineering

Enhanced features capture important physics relationships:

-   Energy band alignment between layers
-   Thickness ratios affecting transport
-   Doping concentration effects on carrier dynamics

### 3. Comprehensive Data Quality

-   Handles missing values and simulation failures gracefully
-   Optional outlier removal preserves data by default
-   Comprehensive logging for reproducibility

### 4. ML-Ready Output

-   Proper train/test splits for model validation
-   Standardized feature naming and scaling
-   Rich metadata for understanding dataset characteristics

### 5. Configurable Example Parameters

-   External JSON configuration for easy device parameter adjustment
-   No code modification needed - just edit the parameter file
-   Built-in physics constraints documentation
-   Transparent parameter loading and validation

## File Structure

```
myML/
├── scripts/
│   ├── 1_create_feature_names.py     # Define features and parameters
│   ├── 2_generate_simulations.py     # Run physics simulations
│   ├── 3_extract_simulation_data.py  # Extract MPP and IntSRHn_mean
│   ├── 4_prepare_ml_data.py          # Create ML-ready datasets
│   ├── 5_train_models.py             # Train ML models
│   └── 6_predict.py                  # Predict and validate models
├── sim/
│   ├── parameters.txt                # Parameter bounds
│   ├── simss.exe                     # Physics simulation executable
│   ├── Data/                         # Material properties
│   └── simulations/                  # Simulation results
├── results/
│   ├── features/                     # Feature definitions from Script 1
│   ├── extract_simulation_data/      # Extracted simulation data
│   ├── prepare_ml_data/              # ML-ready datasets
│   ├── train_optimization_models/    # Trained ML models and scalers
│   └── predict/                      # Model validation and prediction results
├── example_device_parameters.json    # Configurable example device parameters
└── README.md                         # This file
```

## Prerequisites

-   **Python 3.7+** with pandas, numpy, scikit-learn
-   **Physics Simulation**: `sim/simss.exe` (drift-diffusion solver)
-   **Material Data**: Optical and electrical properties in `sim/Data/`
-   **Parameter Configuration**: Bounds defined in `sim/parameters.txt`

## Next Steps

After completing the full ML workflow (Steps 1-6), you can:

1. **Customize Example Parameters**: Edit `example_device_parameters.json` to test different device configurations
2. **Run Optimization**: Apply optimization algorithms using the validated models to find optimal device parameters
3. **Analyze Trade-offs**: Study the relationship between efficiency and recombination using the trained models
4. **Scale to New Materials**: Extend the workflow to other solar cell material systems
5. **Deploy Models**: Use the trained models in production optimization systems

## Troubleshooting

### Common Issues

1. **Missing Feature Definitions**

    - **Error**: "Feature definitions not found"
    - **Solution**: Run `python scripts/1_create_feature_names.py` first

2. **Simulation Failures**

    - **Check**: `results/generated_simulations/generated_simulations.log`
    - **Solution**: Verify parameter bounds in `sim/parameters.txt`

3. **Empty Extraction Results**

    - **Check**: `results/extract_simulation_data/extraction.log`
    - **Solution**: Ensure simulations completed successfully

4. **Missing Data in ML Preparation**
    - **Check**: `results/prepare_ml_data/preparation.log`
    - **Solution**: Verify extraction step completed successfully

### Performance Tips

1. **Simulation Speed**: Adjust number of parameter combinations in script 2
2. **Memory Usage**: Process simulations in batches if memory is limited
3. **Feature Selection**: Disable enhanced features if not needed: `--no-enhanced-features`
4. **Data Quality**: Enable outlier removal if needed: `--remove-outliers`

## Contact and Support

For questions about the workflow:

1. Check log files in each `results/` subdirectory
2. Review script documentation and comments
3. Verify input file formats and parameter bounds

This workflow provides a comprehensive approach to solar cell optimization using physics-based simulations and machine learning.
