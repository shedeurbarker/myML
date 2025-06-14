import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def create_3d_layer_visualization():
    # Real thicknesses (nm)
    real_thicknesses = {
        'Glass': 80,           # Visual only, not physical
        'ITO (Anode)': 50,    # L_TCO = 50E-9 m
        'PEDOT (HTL)': 40,    # L = 40E-9 m
        'MAPI (Active)': 500, # L = 500E-9 m
        'PCBM (ETL)': 25,     # L = 25E-9 m
        'Au (Cathode)': 101   # L_BE = 101E-9 m
    }
    # Colors for each layer
    colors = {
        'Glass': '#C0C0C0',  # Silver
        'ITO (Anode)': '#87CEEB',  # Sky Blue
        'PEDOT (HTL)': '#FF4500',  # Orange Red
        'MAPI (Active)': '#32CD32',  # Lime Green
        'PCBM (ETL)': '#4169E1',  # Royal Blue
        'Au (Cathode)': '#FFD700'  # Gold
    }
    # Panel size (larger for clarity)
    panel_x = 18
    panel_y = 8
    fig = plt.figure(figsize=(12, 8))
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
        box = Poly3DCollection(verts, facecolors=colors[layer], edgecolors='k', linewidths=1, alpha=0.85)
        ax.add_collection3d(box)
        # Store label position (right side, center of layer)
        label_positions.append((panel_x, panel_y/2, z_pos+dz/2, layer, dz))
        z_pos += dz
    # Draw labels and lines outside the box
    for i, (lx, ly, lz, layer, dz) in enumerate(label_positions):
        label_y = panel_y + 2
        # Draw a line from the box to the label
        ax.plot([lx, lx+2], [ly, label_y], [lz, lz], color='black', lw=1)
        # Add the label
        ax.text(lx+2.2, label_y, lz, f"{layer}\n{dz} nm", fontsize=11, va='center', ha='left', fontweight='bold', color='black')
    # Set limits and view
    ax.set_xlim(0, panel_x+8)
    ax.set_ylim(0, panel_y+8)
    ax.set_zlim(0, z_pos+100)
    ax.view_init(elev=18, azim=-35)
    ax.set_axis_off()
    # Add title
    plt.title('Perovskite Solar Cell Layer Structure (n-i-p, 3D View)', pad=20, fontsize=15, fontweight='bold')
    # Add work function information
    work_function_text = (
        "Work Functions:\n"
        "ITO (Anode): 4.05 eV\n"
        "Au (Cathode): 5.2 eV"
    )
    plt.figtext(0.02, 0.02, work_function_text, fontsize=10)
    # Save and show
    plt.savefig('perovskite_layers_3d.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_3d_layer_visualization()
    print("3D layer visualization has been saved as 'perovskite_layers_3d.png'") 