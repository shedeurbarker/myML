import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def create_3d_layer_visualization():
    # Real thicknesses (nm)
    real_thicknesses = {
        'Glass': 80,           # Visual only, not physical
        'ITO': 50,    # L_TCO = 50E-9 m
        'PEDOT': 40,    # L = 40E-9 m
        'MAPI': 500, # L = 500E-9 m
        'PCBM': 25,     # L = 25E-9 m
        'Au': 101   # L_BE = 101E-9 m
    }
    # Descriptions for each layer
    descriptions = {
        'Glass': 'Substrate',
        'ITO': 'Top Electrode',
        'PEDOT': 'HTL',
        'MAPI': 'Active Layer',
        'PCBM': 'ETL',
        'Au': 'Cathode'
    }
    # Colors for each layer
    colors = {
        'Glass': '#C0C0C0',  # Silver
        'ITO': '#87CEEB',  # Sky Blue
        'PEDOT': '#FF4500',  # Orange Red
        'MAPI': '#32CD32',  # Lime Green
        'PCBM': '#4169E1',  # Royal Blue
        'Au': '#FFD700'  # Gold
    }
    # Make the panel and figure much larger
    panel_x = 36
    panel_y = 16
    fig = plt.figure(figsize=(28, 14))
    ax = fig.add_subplot(111, projection='3d')
    z_pos = 0
    label_positions = []
    for layer in real_thicknesses:
        dz = real_thicknesses[layer]
        # Define the 8 corners of the box
        x = [0, panel_x, panel_x, 0, 0, panel_x, panel_x, 0]
        y = [0, 0, panel_y, panel_y, 0, 0, panel_y, panel_y]
        z = [z_pos, z_pos, z_pos, z_pos, z_pos+dz, z_pos+dz, z_pos+dz, z_pos+dz]
        # Vertices for each face
        verts = [
            [ [x[0],y[0],z[0]], [x[1],y[1],z[1]], [x[2],y[2],z[2]], [x[3],y[3],z[3]] ], # bottom
            [ [x[4],y[4],z[4]], [x[5],y[5],z[5]], [x[6],y[6],z[6]], [x[7],y[7],z[7]] ], # top
            [ [x[0],y[0],z[0]], [x[1],y[1],z[1]], [x[5],y[5],z[5]], [x[4],y[4],z[4]] ], # front
            [ [x[2],y[2],z[2]], [x[3],y[3],z[3]], [x[7],y[7],z[7]], [x[6],y[6],z[6]] ], # back
            [ [x[1],y[1],z[1]], [x[2],y[2],z[2]], [x[6],y[6],z[6]], [x[5],y[5],z[5]] ], # right
            [ [x[4],y[4],z[4]], [x[7],y[7],z[7]], [x[3],y[3],z[3]], [x[0],y[0],z[0]] ]  # left
        ]
        box = Poly3DCollection(verts, facecolors=colors[layer], edgecolors='k', linewidths=2, alpha=0.90)
        ax.add_collection3d(box)
        # Store label position (center of layer)
        label_positions.append((panel_x, panel_y/2, z_pos+dz/2, layer, dz, descriptions[layer]))
        z_pos += dz
    # Draw labels and lines outside the box, staggered vertically
    label_y_base = panel_y + 5
    label_y_step = 2.5
    for i, (lx, ly, lz, layer, dz, desc) in enumerate(label_positions):
        label_y = label_y_base + i * label_y_step
        # Draw a short line from the right face to the label
        ax.plot([panel_x, panel_x+2.5], [ly, label_y], [lz, lz], color='black', lw=2)
        # Combine name, description, and thickness in one label
        label_text = f"{layer} ({desc} – {dz} nm)"
        ax.text(panel_x+3, label_y, lz, label_text, fontsize=22, va='center', ha='left', fontweight='bold', color='black')
    # Set limits and view
    ax.set_xlim(0, panel_x+30)
    ax.set_ylim(0, panel_y+20)
    ax.set_zlim(0, z_pos+200)
    ax.view_init(elev=18, azim=-35)
    ax.set_axis_off()
    plt.subplots_adjust(right=0.85)
    # Add title
    plt.title('Perovskite Solar Cell Layer Structure (n-i-p, 3D View)', pad=40, fontsize=28, fontweight='bold')
    # Add work function information
    work_function_text = (
        "Work Functions:\n"
        "ITO (Anode): 4.05 eV\n"
        "Au (Cathode): 5.2 eV"
    )
    plt.figtext(0.02, 0.02, work_function_text, fontsize=18)
    # Save and show
    plt.savefig('perovskite_layers_3d.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    create_3d_layer_visualization()
    print("3D layer visualization has been saved as 'perovskite_layers_3d.png'") 