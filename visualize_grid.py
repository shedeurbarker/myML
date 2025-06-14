import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def calculate_grid_points(total_length, num_points, grad):
    """Calculate the position of grid points with exponential distribution"""
    x = np.linspace(0, 1, num_points)
    # Exponential distribution
    positions = total_length * (np.exp(grad * x) - 1) / (np.exp(grad) - 1)
    return positions

def plot_grid_distribution():
    # Device parameters
    total_length = 565e-9  # 565 nm
    num_points = 300
    grad = 5

    # Layer thicknesses
    pcbm_thickness = 25e-9  # 25 nm
    mapi_thickness = 500e-9  # 500 nm
    pedot_thickness = 40e-9  # 40 nm

    # Calculate grid positions
    grid_positions = calculate_grid_points(total_length, num_points, grad)
    spacing = np.diff(grid_positions)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
    fig.suptitle('Solar Cell Grid Distribution Analysis', fontsize=16)

    # Plot 1: Grid points and layer structure
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
    ax1.set_title('Grid Points Distribution Across Device')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Plot 2: Spacing between points
    ax2.plot(grid_positions[:-1], spacing, 'b-', label='Point Spacing')
    ax2.set_xlabel('Position (m)')
    ax2.set_ylabel('Spacing (m)')
    ax2.set_title('Grid Point Spacing')
    ax2.grid(True)
    ax2.legend()

    # Add interface markers to spacing plot
    ax2.axvline(x=pcbm_thickness, color='red', linestyle='--')
    ax2.axvline(x=pcbm_thickness + mapi_thickness, color='red', linestyle='--')

    # Calculate and display statistics
    interface_region = 5e-9  # 5 nm around interfaces
    pcbm_mapi_points = np.where(np.abs(grid_positions - pcbm_thickness) < interface_region)[0]
    mapi_pedot_points = np.where(np.abs(grid_positions - (pcbm_thickness + mapi_thickness)) < interface_region)[0]
    bulk_points = np.where((grid_positions > pcbm_thickness + interface_region) & 
                          (grid_positions < pcbm_thickness + mapi_thickness - interface_region))[0]

    stats_text = f"""
    Grid Statistics:
    Total Points: {num_points}
    Points near PCBM/MAPI interface: {len(pcbm_mapi_points)}
    Points near MAPI/PEDOT interface: {len(mapi_pedot_points)}
    Points in bulk MAPI: {len(bulk_points)}
    Average spacing near interfaces: {np.mean(spacing[pcbm_mapi_points]):.2e} m
    Average spacing in bulk: {np.mean(spacing[bulk_points]):.2e} m
    """
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('grid_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_grid_distribution() 