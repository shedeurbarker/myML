import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def calculate_grid_points(total_length, num_points, grad):
    """Calculate the position of grid points with exponential distribution"""
    x = np.linspace(0, 1, num_points)
    # Exponential distribution
    positions = total_length * (np.exp(grad * x) - 1) / (np.exp(grad) - 1)
    return positions

def plot_grid_comparison():
    # Device parameters
    total_length = 565e-9  # 565 nm
    grad = 5
    
    # Different grid numbers to compare
    grid_numbers = [150, 300, 600]
    
    # Layer thicknesses
    pcbm_thickness = 25e-9  # 25 nm
    mapi_thickness = 500e-9  # 500 nm
    pedot_thickness = 40e-9  # 40 nm

    # Create figure
    fig, axes = plt.subplots(len(grid_numbers), 2, figsize=(15, 4*len(grid_numbers)))
    fig.suptitle('Grid Number Comparison Analysis', fontsize=16)

    for idx, num_points in enumerate(grid_numbers):
        # Calculate grid positions
        grid_positions = calculate_grid_points(total_length, num_points, grad)
        spacing = np.diff(grid_positions)

        # Plot 1: Grid points and layer structure
        ax1 = axes[idx, 0]
        # Draw layers
        ax1.add_patch(Rectangle((0, 0), pcbm_thickness, 1, facecolor='lightblue', alpha=0.3, label='PCBM'))
        ax1.add_patch(Rectangle((pcbm_thickness, 0), mapi_thickness, 1, facecolor='lightgreen', alpha=0.3, label='MAPI'))
        ax1.add_patch(Rectangle((pcbm_thickness + mapi_thickness, 0), pedot_thickness, 1, facecolor='lightcoral', alpha=0.3, label='PEDOT'))

        # Plot grid points
        ax1.scatter(grid_positions, np.ones_like(grid_positions), c='black', s=10, label='Grid Points')
        
        # Add interface markers
        ax1.axvline(x=pcbm_thickness, color='red', linestyle='--', label='PCBM/MAPI Interface')
        ax1.axvline(x=pcbm_thickness + mapi_thickness, color='red', linestyle='--', label='MAPI/PEDOT Interface')

        ax1.set_xlabel('Position (m)')
        ax1.set_ylabel('Layer Structure')
        ax1.set_title(f'Grid Points Distribution (NP = {num_points})')
        ax1.legend(loc='upper right')
        ax1.grid(True)

        # Plot 2: Spacing between points
        ax2 = axes[idx, 1]
        ax2.plot(grid_positions[:-1], spacing, 'b-', label='Point Spacing')
        ax2.set_xlabel('Position (m)')
        ax2.set_ylabel('Spacing (m)')
        ax2.set_title(f'Grid Point Spacing (NP = {num_points})')
        ax2.grid(True)
        ax2.legend()

        # Add interface markers to spacing plot
        ax2.axvline(x=pcbm_thickness, color='red', linestyle='--')
        ax2.axvline(x=pcbm_thickness + mapi_thickness, color='red', linestyle='--')

        # Calculate statistics
        interface_region = 5e-9  # 5 nm around interfaces
        pcbm_mapi_points = np.where(np.abs(grid_positions - pcbm_thickness) < interface_region)[0]
        mapi_pedot_points = np.where(np.abs(grid_positions - (pcbm_thickness + mapi_thickness)) < interface_region)[0]
        bulk_points = np.where((grid_positions > pcbm_thickness + interface_region) & 
                              (grid_positions < pcbm_thickness + mapi_thickness - interface_region))[0]

        stats_text = f"""
        Grid Statistics (NP = {num_points}):
        Points near PCBM/MAPI interface: {len(pcbm_mapi_points)}
        Points near MAPI/PEDOT interface: {len(mapi_pedot_points)}
        Points in bulk MAPI: {len(bulk_points)}
        Avg. interface spacing: {np.mean(spacing[pcbm_mapi_points]):.2e} m
        Avg. bulk spacing: {np.mean(spacing[bulk_points]):.2e} m
        """
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('grid_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_grid_impact():
    """Analyze the impact of different grid numbers on simulation accuracy"""
    grid_numbers = [150, 300, 600]
    interface_region = 5e-9  # 5 nm around interfaces
    
    print("\nGrid Number Impact Analysis:")
    print("-" * 50)
    
    for num_points in grid_numbers:
        # Calculate grid positions
        grid_positions = calculate_grid_points(565e-9, num_points, 5)
        spacing = np.diff(grid_positions)
        
        # Calculate interface points
        pcbm_mapi_points = np.where(np.abs(grid_positions - 25e-9) < interface_region)[0]
        mapi_pedot_points = np.where(np.abs(grid_positions - 525e-9) < interface_region)[0]
        
        # Calculate statistics
        interface_spacing = np.mean(spacing[pcbm_mapi_points])
        bulk_spacing = np.mean(spacing[100:200])  # Middle of MAPI layer
        
        print(f"\nNP = {num_points}:")
        print(f"Interface resolution: {interface_spacing:.2e} m")
        print(f"Bulk resolution: {bulk_spacing:.2e} m")
        print(f"Interface points: {len(pcbm_mapi_points)}")
        print(f"Points per nm at interface: {1e9/interface_spacing:.1f}")
        print(f"Points per nm in bulk: {1e9/bulk_spacing:.1f}")
        
        # Estimate computation time (rough estimate)
        comp_time = num_points * 0.01  # 0.01 seconds per point
        print(f"Estimated computation time: {comp_time:.1f} seconds")

if __name__ == "__main__":
    plot_grid_comparison()
    analyze_grid_impact() 