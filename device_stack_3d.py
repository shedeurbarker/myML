#!/usr/bin/env python3
"""
3D Device Stack Visualizer

This script creates a detailed 3D visualization of the solar cell device stack
showing layers in true 3D perspective with depth, materials, transparency,
and interactive viewing capabilities.

The device structure (from top to bottom):
- TCO (ITO): 50 nm - Transparent front contact
- ETL (PCBM): 20-50 nm - Electron transport layer  
- Active (MAPI): 210-350 nm - Perovskite absorber layer
- HTL (PEDOT): 20-50 nm - Hole transport layer
- Back Electrode (Au): 101 nm - Reflective back contact

Light enters from the top through the transparent ITO contact.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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

def create_3d_layer(ax, x_range, y_range, z_bottom, z_top, color, alpha=0.8, edge_color='black'):
    """Create a 3D rectangular layer (cuboid) between z_bottom and z_top."""
    
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # Define the 8 vertices of the cuboid
    vertices = np.array([
        [x_min, y_min, z_bottom],  # 0: bottom face
        [x_max, y_min, z_bottom],  # 1
        [x_max, y_max, z_bottom],  # 2
        [x_min, y_max, z_bottom],  # 3
        [x_min, y_min, z_top],     # 4: top face
        [x_max, y_min, z_top],     # 5
        [x_max, y_max, z_top],     # 6
        [x_min, y_max, z_top],     # 7
    ])
    
    # Define the 6 faces of the cuboid
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front face
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back face
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right face
        [vertices[4], vertices[7], vertices[3], vertices[0]],  # left face
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top face
        [vertices[0], vertices[3], vertices[2], vertices[1]],  # bottom face
    ]
    
    # Create the 3D polygon collection with low z-order so labels appear on top
    poly3d = Poly3DCollection(faces, facecolors=color, alpha=alpha, 
                             edgecolors=edge_color, linewidths=0.5, zorder=1)
    ax.add_collection3d(poly3d)
    
    return poly3d

def add_light_rays_3d(ax, x_range, y_range, z_top, num_rays=12):
    """Add 3D light rays with clear arrows coming from above."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # Create a grid of light ray positions
    rows = int(np.sqrt(num_rays))
    cols = int(np.sqrt(num_rays))
    x_positions = np.linspace(x_min + 1000, x_max - 1000, cols)
    y_positions = np.linspace(y_min + 1000, y_max - 1000, rows)
    
    for x in x_positions:
        for y in y_positions:
            # Light ray from above with arrow
            z_start = z_top + 300
            z_end = z_top + 50
            
            # Draw the light ray line
            ax.plot([x, x], [y, y], [z_start, z_end], 
                   color='gold', linewidth=3, alpha=0.8)
            
            # Add a clear downward arrow at the end
            arrow_length = 80
            ax.quiver(x, y, z_end, 0, 0, -arrow_length, 
                     color='orange', arrow_length_ratio=0.3, 
                     linewidth=3, alpha=0.9)
    
    # Add a central bright light beam
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    ax.plot([center_x, center_x], [center_y, center_y], [z_top + 400, z_top + 50], 
           color='yellow', linewidth=5, alpha=0.9)
    
    # Add central arrow
    ax.quiver(center_x, center_y, z_top + 50, 0, 0, -100, 
             color='red', arrow_length_ratio=0.3, 
             linewidth=4, alpha=1.0)

def add_charge_paths_3d(ax, x_range, y_range, z_positions, active_layer_idx):
    """Add 3D charge transport paths."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    # Active layer center
    active_z = (z_positions[active_layer_idx] + z_positions[active_layer_idx + 1]) / 2
    
    # Electron paths (to top contact)
    for i in range(5):
        x_offset = (i - 2) * 800
        y_offset = (i - 2) * 600
        x_pos = x_center + x_offset
        y_pos = y_center + y_offset
        
        # Electron path (blue, upward)
        ax.plot([x_pos, x_pos], [y_pos, y_pos], [active_z, z_positions[0] - 50], 
               color='blue', linewidth=3, alpha=0.7)
        ax.scatter([x_pos], [y_pos], [z_positions[0] - 50], 
                  color='blue', s=50, marker='^', alpha=0.8)
    
    # Hole paths (to bottom contact)
    for i in range(5):
        x_offset = (i - 2) * 700
        y_offset = (i - 2) * 500
        x_pos = x_center + x_offset
        y_pos = y_center + y_offset
        
        # Hole path (red, downward)
        ax.plot([x_pos, x_pos], [y_pos, y_pos], [active_z, z_positions[-1] + 50], 
               color='red', linewidth=3, alpha=0.7)
        ax.scatter([x_pos], [y_pos], [z_positions[-1] + 50], 
                  color='red', s=50, marker='v', alpha=0.8)

def create_3d_device_stack():
    """Create a comprehensive 3D visualization of the device stack."""
    
    # Load actual device parameters
    l1_thickness, l2_thickness, l3_thickness, device_config = load_device_parameters()
    
    # Fixed layer thicknesses from simulation setup
    tco_thickness = 50.0  # nm (ITO)
    back_electrode_thickness = 101.0  # nm (Au)
    
    # Layer properties for 3D visualization
    layers = [
        {
            'name': 'TCO (ITO)',
            'thickness': tco_thickness,
            'color': '#E8F4FD',  # Light blue (transparent)
            'alpha': 0.3,  # Very transparent
            'edge_color': '#2196F3',
            'material': 'Indium Tin Oxide'
        },
        {
            'name': 'ETL (PCBM)', 
            'thickness': l1_thickness,
            'color': '#E3F2FD',  # Light blue
            'alpha': 0.6,
            'edge_color': '#1976D2',
            'material': 'PCBM'
        },
        {
            'name': 'Active (MAPI)',
            'thickness': l2_thickness, 
            'color': '#C8E6C9',  # Light green
            'alpha': 0.8,  # Less transparent (absorbing layer)
            'edge_color': '#388E3C',
            'material': 'CH₃NH₃PbI₃'
        },
        {
            'name': 'HTL (PEDOT)',
            'thickness': l3_thickness,
            'color': '#FFCDD2',  # Light red
            'alpha': 0.6,
            'edge_color': '#D32F2F', 
            'material': 'PEDOT:PSS'
        },
        {
            'name': 'Back Electrode (Au)',
            'thickness': back_electrode_thickness,
            'color': '#FFF9C4',  # Light yellow (gold)
            'alpha': 0.9,  # Nearly opaque (metallic)
            'edge_color': '#F57F17',
            'material': 'Gold'
        }
    ]
    
    # Calculate total thickness and z-positions (flip the stack - light enters from top)
    total_thickness = sum(layer['thickness'] for layer in layers)
    z_positions = [0]  # Start from z=0 (bottom)
    
    # Reverse the layer order so TCO is at the top and substrate at bottom
    layers_reversed = layers[::-1]  # Reverse the layer list
    for layer in layers_reversed:
        z_positions.append(z_positions[-1] + layer['thickness'])
    z_positions.reverse()  # Reverse z_positions so top layer has highest z
    
    # Adjust z_positions to start from 0 at bottom
    max_z = max(z_positions)
    z_positions = [max_z - z for z in z_positions]
    
    # Device lateral dimensions (μm converted to nm for consistency)
    device_width = 8000   # 8 μm in nm
    device_length = 6000  # 6 μm in nm
    
    # Create 3D figure with single plot
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'3D Device Stack Visualization: {device_config.get("device_type", "Perovskite Solar Cell")}', 
                 fontsize=18, fontweight='bold')
    
    # === MAIN 3D VIEW ===
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_title('3D Isometric View with Layer Details', fontsize=16, fontweight='bold')
    
    x_range = (0, device_width)
    y_range = (0, device_length)
    
    # Draw each layer as a 3D cuboid (using reversed layer order)
    for i, layer in enumerate(layers_reversed):
        z_bottom = z_positions[i]
        z_top = z_positions[i + 1]
        
        create_3d_layer(ax1, x_range, y_range, z_bottom, z_top, 
                       layer['color'], layer['alpha'], layer['edge_color'])
        
        # Add layer labels with extreme vertical positioning for maximum spread
        # First 3 layers (0,1,2) on the left, last 2 layers (3,4) on the right
        if i < 3:  # Left side labels - top, middle, bottom positions
            label_x = -3500  # Position labels further left (negative x)
            ha_alignment = 'right'
            line_start_x = label_x + 300
            # Extreme vertical positioning: top, middle, bottom
            if i == 0:  # Bottom position
                label_y = -1000
            elif i == 1:  # Middle position
                label_y = device_length / 2
            else:  # i == 2, Top position
                label_y = device_length + 3000
        else:  # Right side labels - top, middle, bottom positions
            label_x = device_width + 2500  # Position labels further right (positive x)
            ha_alignment = 'left'
            line_start_x = label_x - 300
            # Extreme vertical positioning for right side
            right_label_index = i - 3  # 0, 1 for the right side labels
            if right_label_index == 0:  # Top position
                label_y = device_length + 3000
            else:  # right_label_index == 1, Bottom position
                label_y = -1000
        
        label_z = (z_bottom + z_top) / 2
        
        # Create label text with more information
        label_text = f"{layer['name']}\n{layer['material']}\n{layer['thickness']:.1f} nm"
        
        # Use a high zorder to ensure labels are always on top
        text_obj = ax1.text(label_x, label_y, label_z, label_text, 
                           fontsize=11, fontweight='bold', ha=ha_alignment,
                           bbox=dict(boxstyle="round,pad=0.5", facecolor=layer['color'], 
                                    alpha=0.95, edgecolor=layer['edge_color'], linewidth=2),
                           zorder=1000)  # Very high z-order to be on top
        
        # Add connecting line from label to the specific layer edge with high z-order
        layer_edge_x = device_width / 2
        layer_edge_y = device_length / 2
        
        # Determine which edge of the device to connect to based on label position
        if i < 3:  # Left side labels - connect to left edge
            layer_edge_x = 0
        else:  # Right side labels - connect to right edge
            layer_edge_x = device_width
            
        # Connect from label position to the actual layer edge at its z-position
        line = ax1.plot([line_start_x, layer_edge_x], [label_y, layer_edge_y], 
                       [label_z, label_z], '--', color=layer['edge_color'], 
                       linewidth=2, alpha=0.8, zorder=999)  # High z-order for lines
    
    # Add light rays from above (coming down to the top layer)
    add_light_rays_3d(ax1, x_range, y_range, z_positions[0])  # Top of the stack
    
    # Add light source label positioned higher and to the side to avoid covering device
    light_label_x = device_width / 2 - 1500  # Offset to the side
    light_label_y = device_length / 2
    light_label_z = z_positions[0] + 600  # Higher position
    ax1.text(light_label_x, light_label_y, light_label_z, 
            "☀️ SOLAR LIGHT\n(AM1.5G)", fontsize=12, fontweight='bold', 
            ha='center', va='center', color='darkorange',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', 
                     alpha=0.9, edgecolor='orange', linewidth=2),
            zorder=1000)
    
    # Add charge transport paths
    add_charge_paths_3d(ax1, x_range, y_range, z_positions, 2)  # Active layer is index 2
    
    # Add substrate
    substrate_thickness = 100
    create_3d_layer(ax1, x_range, y_range, z_positions[-1], z_positions[-1] + substrate_thickness, 
                   '#F5F5F5', 0.4, '#757575')
    
    # Add substrate label on the right side with extreme positioning (middle)
    substrate_z = z_positions[-1] + substrate_thickness/2
    substrate_label_x = device_width + 2500  # Position further right
    # Position substrate label in middle position on right side
    substrate_label_y = device_length / 2  # Middle position
    
    substrate_text = ax1.text(substrate_label_x, substrate_label_y, substrate_z, 
                             "Substrate\n(Glass)\n100.0 nm", fontsize=11, fontweight='bold', ha='left',
                             bbox=dict(boxstyle="round,pad=0.5", facecolor='#F5F5F5', 
                                      alpha=0.95, edgecolor='#757575', linewidth=2),
                             zorder=1000)  # High z-order to be on top
    
    # Add connecting line for substrate with high z-order (connect to right edge)
    substrate_edge_x = device_width  # Right edge for substrate
    substrate_edge_y = device_length / 2
    substrate_line = ax1.plot([substrate_label_x - 300, substrate_edge_x], [substrate_label_y, substrate_edge_y], 
                             [substrate_z, substrate_z], '--', color='#757575', 
                             linewidth=2, alpha=0.8, zorder=999)
    
    # Set axis properties
    ax1.set_xlabel('Width (nm)', fontsize=14)
    ax1.set_ylabel('Length (nm)', fontsize=14)
    ax1.set_zlabel('Height (nm)', fontsize=14)
    
    # Set viewing limits to show labels on both sides with improved spacing and light arrows
    ax1.set_xlim(-4000, device_width + 3000)  # Extended x-range for wider label spacing
    ax1.set_ylim(-2000, device_length + 4000)  # Extended y-range for better vertical spacing
    ax1.set_zlim(-100, total_thickness + 700)  # Extended z-range for higher light label
    ax1.view_init(elev=25, azim=45)
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Add text annotations
    info_text = f"""
3D Device Characteristics:
• Lateral dimensions: {device_width/1000:.1f} × {device_length/1000:.1f} μm
• Total stack height: {total_thickness:.1f} nm
• Volume: {device_width * device_length * total_thickness / 1e9:.2f} μm³
• Active layer volume: {device_width * device_length * l2_thickness / 1e9:.2f} μm³
• Aspect ratio: {total_thickness/(device_width/1000):.1f} (height/width)

Light Collection:
☀️ → TCO (transparent) → ETL → Active (absorption) → HTL → Au (reflection)

Charge Transport:
• Electrons: Active → ETL → TCO (front contact)
• Holes: Active → HTL → Au (back contact)
    """
    
    fig.text(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('device_stack_3d_view.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("✅ 3D Device stack visualization created successfully!")
    print(f"📊 Device: {device_config.get('device_type', 'Perovskite Solar Cell')}")
    print(f"📏 3D dimensions:")
    print(f"   • Width: {device_width/1000:.1f} μm")
    print(f"   • Length: {device_length/1000:.1f} μm") 
    print(f"   • Height: {total_thickness:.1f} nm")
    print(f"   • Volume: {device_width * device_length * total_thickness / 1e9:.2f} μm³")
    print(f"🎯 View configuration:")
    print(f"   • Single clean 3D isometric view: 25° elevation, 45° azimuth")
    print(f"   • Labels balanced: 3 on left, 3 on right")
    print(f"   • Light entering from top (physically correct)")
    print(f"   • Connecting lines link labels to layers")
    print(f"💾 Saved as: device_stack_3d_view.png")

def create_multiple_3d_views(layers, device_config, device_width, device_length, z_positions):
    """Create multiple 3D viewing angles in one figure."""
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'3D Multi-View Analysis: {device_config.get("device_type", "Perovskite Solar Cell")}', 
                 fontsize=16, fontweight='bold')
    
    # Different viewing angles
    views = [
        {'elev': 90, 'azim': 0, 'title': 'Top View (Plan)'},
        {'elev': 0, 'azim': 0, 'title': 'Front View (Elevation)'},
        {'elev': 0, 'azim': 90, 'title': 'Side View (Profile)'},
        {'elev': 30, 'azim': 225, 'title': 'Perspective View'}
    ]
    
    x_range = (0, device_width)
    y_range = (0, device_length)
    
    for i, view in enumerate(views):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.set_title(view['title'], fontsize=12, fontweight='bold')
        
        # Draw layers
        for j, layer in enumerate(layers):
            z_bottom = z_positions[j]
            z_top = z_positions[j + 1]
            
            create_3d_layer(ax, x_range, y_range, z_bottom, z_top,
                           layer['color'], layer['alpha'], layer['edge_color'])
        
        # Set view angle
        ax.view_init(elev=view['elev'], azim=view['azim'])
        
        # Set axis properties
        ax.set_xlabel('Width (nm)', fontsize=10)
        ax.set_ylabel('Length (nm)', fontsize=10)
        ax.set_zlabel('Height (nm)', fontsize=10)
        
        ax.set_xlim(0, device_width)
        ax.set_ylim(0, device_length)
        ax.set_zlim(0, sum(layer['thickness'] for layer in layers))
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('device_stack_3d_multi_view.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')

if __name__ == "__main__":
    create_3d_device_stack()
