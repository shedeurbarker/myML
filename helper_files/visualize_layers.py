import matplotlib.pyplot as plt


def create_layer_visualization():
    # Real thicknesses (nm)
    real_thicknesses = {
        'Glass': 80,           # Visual only, not physical
        'ITO (Anode)': 50,    # L_TCO = 50E-9 m
        'PEDOT (HTL)': 40,    # L = 40E-9 m
        'MAPI (Active)': 500, # L = 500E-9 m
        'PCBM (ETL)': 25,     # L = 25E-9 m
        'Au (Cathode)': 101   # L_BE = 101E-9 m
    }
    # Visual thicknesses (arbitrary, for clarity)
    visual_thicknesses = {
        'Glass': 60,
        'ITO (Anode)': 60,
        'PEDOT (HTL)': 60,
        'MAPI (Active)': 120,
        'PCBM (ETL)': 60,
        'Au (Cathode)': 60
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
    # Descriptions for each layer
    descriptions = {
        'Glass': 'Substrate',
        'ITO (Anode)': 'Top Electrode',
        'PEDOT (HTL)': 'Hole Transport Layer',
        'MAPI (Active)': 'Perovskite Active Layer',
        'PCBM (ETL)': 'Electron Transport Layer',
        'Au (Cathode)': 'Bottom Electrode'
    }
    panel_width = 8
    fig, ax = plt.subplots(figsize=(8, 7))
    y_pos = 0
    for layer in real_thicknesses:
        vthick = visual_thicknesses[layer]
        rect = plt.Rectangle((0, y_pos), panel_width, vthick,
                             facecolor=colors[layer], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        # Add layer name and description (smaller font size)
        ax.text(panel_width/2, y_pos + vthick/2, f"{layer}\n({descriptions[layer]})",
                ha='center', va='center', fontsize=10, fontweight='bold', color='black')
        # Add real thickness label
        ax.text(panel_width + 0.15, y_pos + vthick/2, f'{real_thicknesses[layer]} nm',
                ha='left', va='center', fontsize=11, color='black')
        y_pos += vthick
    # Add light direction arrow at the top
    ax.arrow(panel_width/2, y_pos + 40, 0, -30, head_width=1, head_length=15, fc='black', ec='black')
    ax.text(panel_width/2, y_pos + 60, 'Light →', ha='center', va='bottom', fontsize=15, fontweight='bold')
    # Set plot limits and aspect
    ax.set_xlim(-0.5, panel_width + 2)
    ax.set_ylim(-30, y_pos + 70)
    ax.set_aspect('auto')
    ax.axis('off')
    # Add title
    plt.title('Perovskite Solar Cell Layer Structure (n-i-p, 2D Flat Panel)', pad=20, fontsize=17, fontweight='bold')
    # Add work function information
    work_function_text = (
        "Work Functions:\n"
        "ITO (Anode): 4.05 eV\n"
        "Au (Cathode): 5.2 eV"
    )
    plt.figtext(0.02, 0.02, work_function_text, fontsize=11)
    # Save and show
    plt.savefig('perovskite_layers_2d_flat.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_layer_visualization()
    print("Layer visualization has been saved as 'perovskite_layers_2d_flat.png'") 