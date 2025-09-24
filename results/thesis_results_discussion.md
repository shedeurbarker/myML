## Results and Discussion

### Results

-   **Dataset preparation**: 9,950 samples; 38 base features plus 18 engineered physics-informed features. Physics validation passed (missing values handled, outliers clipped, constraints validated).

-   **Model performance**

    -   **Efficiency (target `MPP`)**: Validation R²=0.9987, RMSE=6.98, MAE=2.06. The selected XGBoost model achieved Test R²=0.9998, RMSE=2.58, MAE=1.40.
    -   **Recombination (`IntSRHn_mean`)**: Validation R²=0.8735 with large absolute errors due to target magnitude; XGBoost Test R²=0.9967. Interpreted comparatively (relative changes) rather than by absolute error.

-   **Experimental device prediction (PCBM/MAPI/PEDOT)**: All physics constraints satisfied. Predicted performance: `PCE`=19.92% and `MPP`≈199.17 (per-area power, consistent units across runs). Recombination category: Low.

-   **Optimization outcome**: Preserving manufacturability and physics validity, optimization improved performance to `PCE`=21.69% and `MPP`≈216.86, corresponding to **+8.88%** uplift. Recombination changed by +0.15% (negligible).
    -   Key parameter changes (original → optimized):
        -   ETL thickness `L1_L`: 20 nm → ~37.4 nm
        -   Absorber thickness `L2_L`: 350 nm → ~268.6 nm
        -   HTL thickness `L3_L`: 20 nm → ~37.4 nm
        -   ETL `E_c`: 3.80 eV → ~3.92 eV
    -   Energy alignment and electrode compatibility remained satisfied; minor W_R vs. HTL E_v warnings were within tolerance.

#### Figures and Table

-   **Fig. 1a**: `results/8_optimize_device/1a_mpp_comparison.png` — Comparison of `MPP` before and after optimization for the PCBM/MAPI/PEDOT device.
-   **Fig. 1b**: `results/8_optimize_device/1b_pce_comparison.png` — Comparison of `PCE` before and after optimization.
-   **Fig. 1c**: `results/8_optimize_device/1c_recombination_comparison.png` — Comparison of `IntSRHn_mean` before and after optimization.
-   **Fig. 2**: `results/8_optimize_device/2_thickness_optimization.png` — Sensitivity of `PCE` to layer thickness parameters with the optimized point highlighted.
-   **Fig. 3**: `results/8_optimize_device/3_energy_optimization.png` — Sensitivity of `PCE` to energy-level parameters (band offsets and alignment).
-   **Fig. 4**: `results/8_optimize_device/4_doping_optimization.png` — Sensitivity of `PCE` to doping parameters across layers.
-   **Fig. 5**: `results/8_optimize_device/5_performance_improvements.png` — Summary of performance improvements following the multi-stage optimization.
-   **Table 1**: `results/8_optimize_device/6_optimization_table.png` — Optimized parameters versus original settings with relative changes.

### Discussion

-   **Model fidelity and uncertainty**: The `MPP` model shows near-ideal predictive fidelity on synthetic data, supporting its use for optimization. The recombination model reaches high R² on test data but with large absolute errors given the target scale; it is best used for relative comparisons and physics consistency checks.

-   **Physics consistency**: Optimized configurations satisfy energy alignment (ETL_Ec ≥ Active_Ec; Active_Ev ≥ HTL_Ev) and electrode compatibility. Adjusting ETL `E_c` (~3.92 eV) and thicknesses improved transport balance without violating constraints.

-   **Design insights**: Thickness tuning is the primary lever: slightly thicker transport layers (~20 → ~37 nm) and a moderately thinner absorber (~350 → ~269 nm) improve extraction and reduce parasitic losses, raising `MPP` and `PCE`. Small energy-level nudges complement this by improving interfacial alignment.

-   **Practical implications**: The **+8.88%** `PCE` gain (to 21.69%) is below the Shockley–Queisser limit (33.7%) and maintains manufacturability. A ready-to-use configuration is provided at `results/8_optimize_device/example_device_parameters_optimized.json`.

-   **Limitations and future work**: Results derive from high-fidelity simulations; experimental transfer may require recalibration for material quality, interfacial recombination, and contact resistances. Extending to multi-objective optimization (e.g., performance vs. stability, tolerance to process variations) and incorporating uncertainty quantification would improve robustness. Adding physics such as ion migration, interfacial recombination velocities, and grain-boundary effects could further refine guidance.

### Reproducibility

-   Optimized parameters: `results/8_optimize_device/optimized_device_parameters.json`
-   Ready-to-use example: `results/8_optimize_device/example_device_parameters_optimized.json`
-   Optimization report and log: `results/8_optimize_device/optimization_report.json`, `results/8_optimize_device/optimization_log.txt`
