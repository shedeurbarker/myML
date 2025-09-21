#!/usr/bin/env python3
"""
Device Stack Visualizer

This script creates a visual diagram of the solar cell device stack showing
all 5 layers with proper scaling, materials, thicknesses, and light path.

The device structure is:
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
        
        return l1_thickness, l2_thickness, l3_thickness, device_config
        
    except FileNotFoundError:
        # Use default values if file not found
        return 25.0, 280.0, 40.0, {"device_type": "Default Perovskite Solar Cell"}

def create_device_stack_diagram():
    """Create a detailed device stack diagram with proper scaling."""
    
    # Load actual device parameters
    l1_thickness, l2_thickness, l3_thickness, device_config = load_device_parameters()
    
    # Fixed layer thicknesses from simulation setup
    tco_thickness = 50.0  # nm (ITO)
    back_electrode_thickness = 101.0  # nm (Au)
    
    # Layer properties
    layers = [
        {
            'name': 'TCO (ITO)',
            'thickness': tco_thickness,
            'color': '#E8F4FD',  # Light blue (transparent)
            'edge_color': '#2196F3',
            'material': 'Indium Tin Oxide',
            'function': 'Transparent Front Contact',
            'work_function': '4.05 eV'
        },
        {
            'name': 'ETL (PCBM)', 
            'thickness': l1_thickness,
            'color': '#E3F2FD',  # Light blue
            'edge_color': '#1976D2',
            'material': 'PC₆₁BM',
            'function': 'Electron Transport Layer',
            'type': 'n-type'
        },
        {
            'name': 'Active (MAPI)',
            'thickness': l2_thickness, 
            'color': '#C8E6C9',  # Light green
            'edge_color': '#388E3C',
            'material': 'CH₃NH₃PbI₃',
            'function': 'Light Absorber Layer',
            'type': 'Intrinsic'
        },
        {
            'name': 'HTL (PEDOT)',
            'thickness': l3_thickness,
            'color': '#FFCDD2',  # Light red
            'edge_color': '#D32F2F', 
            'material': 'PEDOT:PSS',
            'function': 'Hole Transport Layer',
            'type': 'p-type'
        },
        {
            'name': 'Back Electrode (Au)',
            'thickness': back_electrode_thickness,
            'color': '#FFF9C4',  # Light yellow (gold)
            'edge_color': '#F57F17',
            'material': 'Gold',
            'function': 'Reflective Back Contact', 
            'work_function': '5.2 eV'
        }
    ]
    
    # Calculate total thickness and positions
    total_thickness = sum(layer['thickness'] for layer in layers)
    
    # Create figure with proper aspect ratio
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle(f'Solar Cell Device Stack - {device_config.get("device_type", "Perovskite Solar Cell")}', 
                 fontsize=16, fontweight='bold')
    
    # === LEFT PLOT: Scaled Device Stack ===
    ax1.set_title('Device Stack (Proper Scale)', fontsize=14, fontweight='bold')
    
    # Draw layers from top to bottom
    y_position = 0
    width = 8  # Arbitrary width for visualization
    
    for i, layer in enumerate(layers):
        # Create rectangle for each layer
        rect = patches.Rectangle(
            (0, y_position), width, layer['thickness'],
            facecolor=layer['color'], 
            edgecolor=layer['edge_color'],
            linewidth=2
        )
        ax1.add_patch(rect)
        
        # Add layer labels
        text_y = y_position + layer['thickness'] / 2
        ax1.text(width + 0.5, text_y, f"{layer['name']}\n{layer['thickness']:.1f} nm", 
                va='center', ha='left', fontsize=10, fontweight='bold')
        
        # Add material info
        ax1.text(-0.5, text_y, f"{layer['material']}", 
                va='center', ha='right', fontsize=9, style='italic')
        
        y_position += layer['thickness']
    
    # Add light arrow
    arrow_props = dict(arrowstyle='->', lw=3, color='orange')
    ax1.annotate('☀️ LIGHT', xy=(width/2, -50), xytext=(width/2, -150),
                arrowprops=arrow_props, ha='center', fontsize=12, fontweight='bold', color='orange')
    
    # Add substrate
    substrate_rect = patches.Rectangle(
        (0, y_position), width, 50,
        facecolor='#F5F5F5', edgecolor='#757575', linewidth=1, alpha=0.7
    )
    ax1.add_patch(substrate_rect)
    ax1.text(width + 0.5, y_position + 25, "Substrate (Glass)", 
            va='center', ha='left', fontsize=10, alpha=0.7)
    
    # Set axis properties
    ax1.set_xlim(-3, width + 6)
    ax1.set_ylim(-200, total_thickness + 100)
    ax1.set_ylabel('Thickness (nm)', fontsize=12)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Remove x-axis ticks
    ax1.set_xticks([])
    
    # === RIGHT PLOT: Layer Information Table ===
    ax2.set_title('Layer Properties & Functions', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Create information table
    table_data = []
    headers = ['Layer', 'Material', 'Thickness\n(nm)', 'Function', 'Properties']
    
    for layer in layers:
        properties = []
        if 'type' in layer:
            properties.append(layer['type'])
        if 'work_function' in layer:
            properties.append(f"Φ = {layer['work_function']}")
        
        table_data.append([
            layer['name'],
            layer['material'],
            f"{layer['thickness']:.1f}",
            layer['function'],
            '\n'.join(properties) if properties else '-'
        ])
    
    # Create table
    table = ax2.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     colWidths=[0.15, 0.2, 0.1, 0.3, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#E3F2FD')
        table[(0, i)].set_text_props(weight='bold')
    
    for i in range(len(layers)):
        # Color code rows to match the diagram
        table[(i+1, 0)].set_facecolor(layers[i]['color'])
        table[(i+1, 1)].set_facecolor(layers[i]['color'])
        table[(i+1, 2)].set_facecolor(layers[i]['color'])
        table[(i+1, 3)].set_facecolor(layers[i]['color'])
        table[(i+1, 4)].set_facecolor(layers[i]['color'])
    
    # Add device statistics
    stats_text = f"""
Device Statistics:
• Total Active Thickness: {total_thickness:.1f} nm
• Active Layer: {l2_thickness:.1f} nm ({l2_thickness/total_thickness*100:.1f}% of stack)
• Transport Layers: {l1_thickness + l3_thickness:.1f} nm ({(l1_thickness + l3_thickness)/total_thickness*100:.1f}% of stack)
• Contact Layers: {tco_thickness + back_electrode_thickness:.1f} nm

Light Path:
☀️ → ITO → PCBM → MAPI (absorption) → PEDOT → Au (reflection)

Charge Collection:
• Electrons: MAPI → PCBM → ITO (front contact)
• Holes: MAPI → PEDOT → Au (back contact)
    """
    
    ax2.text(0.02, 0.02, stats_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.5", 
            facecolor='lightgray', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('device_stack_diagram.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("✅ Device stack diagram created successfully!")
    print(f"📊 Device: {device_config.get('device_type', 'Perovskite Solar Cell')}")
    print(f"📏 Layer thicknesses:")
    print(f"   • TCO (ITO): {tco_thickness:.1f} nm")
    print(f"   • ETL (PCBM): {l1_thickness:.1f} nm") 
    print(f"   • Active (MAPI): {l2_thickness:.1f} nm")
    print(f"   • HTL (PEDOT): {l3_thickness:.1f} nm")
    print(f"   • Back Electrode (Au): {back_electrode_thickness:.1f} nm")
    print(f"📐 Total stack thickness: {total_thickness:.1f} nm")
    print(f"💾 Saved as: device_stack_diagram.png")

if __name__ == "__main__":
    create_device_stack_diagram()
