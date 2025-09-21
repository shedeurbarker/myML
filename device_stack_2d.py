#!/usr/bin/env python3
"""
2D Device Stack Cross-Section Visualizer

This script creates a technical 2D cross-sectional view of the solar cell device stack
showing layer interfaces, electric field distribution, energy band diagram, and 
detailed dimensional analysis.

The device structure (from top to bottom):
- TCO (ITO): 50 nm - Transparent front contact
- ETL (PCBM): 20-50 nm - Electron transport layer  
- Active (MAPI): 210-350 nm - Perovskite absorber layer
- HTL (PEDOT): 20-50 nm - Hole transport layer
- Back Electrode (Au): 101 nm - Reflective back contact

Light enters from the top through the transparent ITO contact.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json

def load_device_parameters():
    """Load device parameters from example_device_parameters.json"""
    try:
        with open('example_device_parameters.json', 'r') as f:
            device_config = json.load(f)
        
        params = device_config['parameters']
        
        # Convert to nm for visualization
        l1_thickness = params['L1_L'] * 1e9  # ETL thickness in nm
        l2_thickness = params['L2_L'] * 1e9  # Active layer thickness in nm  
        l3_thickness = params['L3_L'] * 1e9  # HTL thickness in nm
        
        # Energy levels (eV)
        energy_levels = {
            'tco_work_function': 4.05,  # ITO work function
            'etl_lumo': params['L1_E_c'],  # ETL conduction band
            'etl_homo': params['L1_E_v'],  # ETL valence band
            'active_lumo': params['L2_E_c'],  # Active conduction band
            'active_homo': params['L2_E_v'],  # Active valence band
            'htl_lumo': params['L3_E_c'],  # HTL conduction band
            'htl_homo': params['L3_E_v'],  # HTL valence band
            'back_work_function': 5.2,  # Au work function
        }
        
        return l1_thickness, l2_thickness, l3_thickness, device_config, energy_levels
        
    except FileNotFoundError:
        # Use default values if file not found
        energy_levels = {
            'tco_work_function': 4.05,
            'etl_lumo': -4.0, 'etl_homo': -6.0,
            'active_lumo': -3.8, 'active_homo': -5.4,
            'htl_lumo': -2.2, 'htl_homo': -5.1,
            'back_work_function': 5.2,
        }
        return 25.0, 280.0, 40.0, {"device_type": "Default Perovskite Solar Cell"}, energy_levels

def create_2d_device_stack():
    """Create a technical 2D cross-sectional view of the device stack."""
    
    # Load actual device parameters
    l1_thickness, l2_thickness, l3_thickness, device_config, energy_levels = load_device_parameters()
    
    # Fixed layer thicknesses from simulation setup
    tco_thickness = 50.0  # nm (ITO)
    back_electrode_thickness = 101.0  # nm (Au)
    
    # Layer properties for 2D visualization
    layers = [
        {
            'name': 'TCO\n(ITO)',
            'thickness': tco_thickness,
            'color': '#E8F4FD',  # Light blue (transparent)
            'edge_color': '#2196F3',
            'material': 'Indium Tin Oxide',
            'conductivity': 'Conductive',
            'transparency': 'Transparent',
            'pattern': None
        },
        {
            'name': 'ETL\n(PCBM)', 
            'thickness': l1_thickness,
            'color': '#E3F2FD',  # Light blue
            'edge_color': '#1976D2',
            'material': 'PC₆₁BM',
            'conductivity': 'n-type',
            'transparency': 'Semi-transparent',
            'pattern': '///'
        },
        {
            'name': 'Active\n(MAPI)',
            'thickness': l2_thickness, 
            'color': '#C8E6C9',  # Light green
            'edge_color': '#388E3C',
            'material': 'CH₃NH₃PbI₃',
            'conductivity': 'Intrinsic',
            'transparency': 'Absorbing',
            'pattern': '...'
        },
        {
            'name': 'HTL\n(PEDOT)',
            'thickness': l3_thickness,
            'color': '#FFCDD2',  # Light red
            'edge_color': '#D32F2F', 
            'material': 'PEDOT:PSS',
            'conductivity': 'p-type',
            'transparency': 'Semi-transparent',
            'pattern': '\\\\\\'
        },
        {
            'name': 'Back Contact\n(Au)',
            'thickness': back_electrode_thickness,
            'color': '#FFF9C4',  # Light yellow (gold)
            'edge_color': '#F57F17',
            'material': 'Gold',
            'conductivity': 'Metallic',
            'transparency': 'Reflective',
            'pattern': '---'
        }
    ]
    
    # Calculate positions and total thickness
    total_thickness = sum(layer['thickness'] for layer in layers)
    positions = []
    y_pos = 0
    for layer in layers:
        positions.append(y_pos)
        y_pos += layer['thickness']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'2D Cross-Sectional View: {device_config.get("device_type", "Perovskite Solar Cell")}', 
                 fontsize=18, fontweight='bold')
    
    # Define grid layout
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[2, 2, 1.5])
    
    # === MAIN 2D CROSS-SECTION ===
    ax_main = fig.add_subplot(gs[0, :2])
    ax_main.set_title('Device Cross-Section (2D View)', fontsize=14, fontweight='bold')
    
    # Device dimensions for 2D view
    device_width = 10  # μm (arbitrary for visualization)
    width_nm = device_width * 1000  # Convert to nm
    
    # Draw layers
    for i, (layer, y_pos) in enumerate(zip(layers, positions)):
        # Main layer rectangle
        rect = patches.Rectangle(
            (0, y_pos), width_nm, layer['thickness'],
            facecolor=layer['color'], 
            edgecolor=layer['edge_color'],
            linewidth=2,
            hatch=layer['pattern'],
            alpha=0.8
        )
        ax_main.add_patch(rect)
        
        # Layer labels on the right
        text_y = y_pos + layer['thickness'] / 2
        ax_main.text(width_nm + 200, text_y, 
                    f"{layer['name']}\n{layer['thickness']:.1f} nm\n{layer['material']}", 
                    va='center', ha='left', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=layer['color'], alpha=0.7))
        
        # Interface lines
        if i > 0:
            ax_main.axhline(y=y_pos, color='black', linewidth=1, linestyle='--', alpha=0.5)
    
    # Add light rays
    light_x = np.linspace(0, width_nm, 20)
    for i, x in enumerate(light_x[::3]):
        if i % 2 == 0:  # Alternate rays
            ax_main.arrow(x, -100, 0, 80, head_width=100, head_length=15, 
                         fc='orange', ec='orange', alpha=0.7)
    
    # Add light label
    ax_main.text(width_nm/2, -150, '☀️ Incident Light (AM1.5G)', 
                ha='center', fontsize=12, fontweight='bold', color='orange')
    
    # Add charge collection arrows
    # Electrons (blue arrows going up)
    for x_pos in [width_nm*0.2, width_nm*0.5, width_nm*0.8]:
        ax_main.arrow(x_pos, positions[2] + l2_thickness/2, 0, -(positions[2] + l2_thickness/2 + 50), 
                     head_width=150, head_length=30, fc='blue', ec='blue', alpha=0.6)
    
    ax_main.text(width_nm*0.1, positions[0] - 50, 'e⁻', fontsize=14, color='blue', fontweight='bold')
    
    # Holes (red arrows going down)  
    for x_pos in [width_nm*0.3, width_nm*0.6, width_nm*0.9]:
        ax_main.arrow(x_pos, positions[2] + l2_thickness/2, 0, 
                     positions[4] + back_electrode_thickness/2 - (positions[2] + l2_thickness/2), 
                     head_width=150, head_length=30, fc='red', ec='red', alpha=0.6)
    
    ax_main.text(width_nm*0.9, positions[4] + back_electrode_thickness + 30, 'h⁺', 
                fontsize=14, color='red', fontweight='bold')
    
    # Set axis properties
    ax_main.set_xlim(-500, width_nm + 2000)
    ax_main.set_ylim(-200, total_thickness + 100)
    ax_main.set_xlabel('Width (nm)', fontsize=12)
    ax_main.set_ylabel('Depth (nm)', fontsize=12)
    ax_main.grid(True, alpha=0.3)
    
    # === ENERGY BAND DIAGRAM ===
    ax_energy = fig.add_subplot(gs[0, 2])
    ax_energy.set_title('Energy Band Diagram', fontsize=12, fontweight='bold')
    
    # Energy positions for band diagram
    energy_positions = [0]
    for thickness in [layer['thickness'] for layer in layers]:
        energy_positions.append(energy_positions[-1] + thickness)
    
    # Draw energy bands
    vacuum_level = 0  # Reference
    
    # Conduction and valence bands
    cb_levels = [
        energy_levels['tco_work_function'],  # ITO (work function)
        energy_levels['etl_lumo'],           # ETL CB
        energy_levels['active_lumo'],        # Active CB
        energy_levels['htl_lumo'],           # HTL CB
        energy_levels['back_work_function']  # Au (work function)
    ]
    
    vb_levels = [
        energy_levels['tco_work_function'],  # ITO (work function)
        energy_levels['etl_homo'],           # ETL VB
        energy_levels['active_homo'],        # Active VB
        energy_levels['htl_homo'],           # HTL VB
        energy_levels['back_work_function']  # Au (work function)
    ]
    
    # Plot conduction band
    for i in range(len(layers)):
        y_start = energy_positions[i]
        y_end = energy_positions[i+1]
        if i == 0 or i == 4:  # Contacts (only work function)
            ax_energy.plot([cb_levels[i], cb_levels[i]], [y_start, y_end], 
                          'b-', linewidth=3, label='Fermi Level' if i == 0 else '')
        else:  # Semiconductors (CB and VB)
            ax_energy.plot([cb_levels[i], cb_levels[i]], [y_start, y_end], 
                          'b-', linewidth=3, label='Conduction Band' if i == 1 else '')
            ax_energy.plot([vb_levels[i], vb_levels[i]], [y_start, y_end], 
                          'r-', linewidth=3, label='Valence Band' if i == 1 else '')
            
            # Fill bandgap
            ax_energy.fill_betweenx([y_start, y_end], vb_levels[i], cb_levels[i], 
                                   alpha=0.2, color=layers[i]['color'])
    
    # Add energy level labels
    for i, (cb, vb) in enumerate(zip(cb_levels, vb_levels)):
        y_mid = (energy_positions[i] + energy_positions[i+1]) / 2
        if i == 0 or i == 4:  # Contacts
            ax_energy.text(cb + 0.1, y_mid, f'{cb:.2f} eV', 
                          fontsize=8, va='center')
        else:  # Semiconductors
            ax_energy.text(cb + 0.1, y_mid + 20, f'{cb:.2f} eV', 
                          fontsize=8, va='center', color='blue')
            ax_energy.text(vb - 0.3, y_mid - 20, f'{vb:.2f} eV', 
                          fontsize=8, va='center', color='red')
    
    ax_energy.set_ylabel('Depth (nm)', fontsize=10)
    ax_energy.set_xlabel('Energy (eV)', fontsize=10)
    ax_energy.legend(fontsize=8)
    ax_energy.grid(True, alpha=0.3)
    ax_energy.set_ylim(0, total_thickness)
    
    # === TECHNICAL SPECIFICATIONS TABLE ===
    ax_specs = fig.add_subplot(gs[1, :])
    ax_specs.set_title('Technical Specifications & Analysis', fontsize=14, fontweight='bold')
    ax_specs.axis('off')
    
    # Create specifications table
    specs_data = []
    headers = ['Layer', 'Material', 'Thickness\n(nm)', 'Conductivity', 'Optical\nProperty', 'Energy Levels\n(eV)']
    
    for i, layer in enumerate(layers):
        if i == 0 or i == 4:  # Contacts
            energy_info = f"Φ = {cb_levels[i]:.2f}"
        else:  # Semiconductors
            bandgap = abs(cb_levels[i] - vb_levels[i])
            energy_info = f"Eg = {bandgap:.2f}\nCB: {cb_levels[i]:.2f}\nVB: {vb_levels[i]:.2f}"
        
        specs_data.append([
            layer['name'].replace('\n', ' '),
            layer['material'],
            f"{layer['thickness']:.1f}",
            layer['conductivity'],
            layer['transparency'],
            energy_info
        ])
    
    # Create table
    table = ax_specs.table(cellText=specs_data, colLabels=headers, 
                          cellLoc='center', loc='center',
                          colWidths=[0.12, 0.18, 0.08, 0.12, 0.12, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#E3F2FD')
        table[(0, i)].set_text_props(weight='bold')
    
    for i in range(len(layers)):
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(layers[i]['color'])
    
    # Add device performance summary
    performance_text = f"""
Device Performance Summary:
• Total Stack Thickness: {total_thickness:.1f} nm
• Active Layer Fraction: {l2_thickness/total_thickness*100:.1f}%
• Transport Layer Balance: ETL={l1_thickness:.1f}nm / HTL={l3_thickness:.1f}nm
• Contact Resistance: TCO + Back = {tco_thickness + back_electrode_thickness:.1f} nm
• Optical Path: {total_thickness/1000:.3f} μm
• Active Layer Bandgap: {abs(energy_levels['active_lumo'] - energy_levels['active_homo']):.2f} eV
• Built-in Potential: ≈{abs(energy_levels['back_work_function'] - energy_levels['tco_work_function']):.2f} V
    """
    
    ax_specs.text(0.65, 0.02, performance_text, transform=ax_specs.transAxes, fontsize=10,
                 verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.5", 
                 facecolor='lightblue', alpha=0.3))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('device_stack_2d_view.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("✅ 2D Device stack cross-section created successfully!")
    print(f"📊 Device: {device_config.get('device_type', 'Perovskite Solar Cell')}")
    print(f"📏 Cross-sectional dimensions:")
    print(f"   • Width: {device_width} μm ({width_nm:.0f} nm)")
    print(f"   • Total depth: {total_thickness:.1f} nm")
    print(f"   • Active layer: {l2_thickness:.1f} nm ({l2_thickness/total_thickness*100:.1f}% of stack)")
    print(f"🔋 Energy characteristics:")
    print(f"   • Active bandgap: {abs(energy_levels['active_lumo'] - energy_levels['active_homo']):.2f} eV")
    print(f"   • Built-in potential: ~{abs(energy_levels['back_work_function'] - energy_levels['tco_work_function']):.2f} V")
    print(f"💾 Saved as: device_stack_2d_view.png")

if __name__ == "__main__":
    create_2d_device_stack()
