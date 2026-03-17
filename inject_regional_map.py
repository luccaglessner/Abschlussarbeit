import json

nb_path = r'c:\Users\lucca\Desktop\Abschlussarbeit HTWK\abschlussarbeit\Abschlussarbeit Bearbeitung\Jupyter Notebooks\3_Machine-Learning\3.2_Machine-Learning\MiniSom\MiniSom_Machine-Learning.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# --- Define the new plot function for the Regional Density Map ---
# We'll inject this helper logic or just the plot logic.
# Let's define a reusable plot block.

regional_plot_code = """
            # ------------------------------- NEU: Regionaler Dichtevergleich (Seite 4) -------------------------------
            REGION_MODE = os.environ.get('SOM_REGION')
            if REGION_MODE and REGION_MODE != 'ALL':
                REGION_NAME = os.environ.get('LOC_REGION_NAME', 'Oberrheingraben')
                LAT_MIN = float(os.environ.get('LOC_LAT_MIN', 47.30))
                LAT_MAX = float(os.environ.get('LOC_LAT_MAX', 50.28))
                LON_MIN = float(os.environ.get('LOC_LON_MIN', 7.45))
                LON_MAX = float(os.environ.get('LOC_LON_MAX', 9.60))
                
                # Filter Region
                mask_region = (process_df_run['WGS84_latitude'] >= LAT_MIN) & (process_df_run['WGS84_latitude'] <= LAT_MAX) & \\
                              (process_df_run['WGS84_longitude'] >= LON_MIN) & (process_df_run['WGS84_longitude'] <= LON_MAX)
                df_subset = process_df_run[mask_region]
                
                if not df_subset.empty:
                    # Counts pro Zelle
                    reg_counts = np.zeros((som_y, som_x))
                    for _, row in df_subset.iterrows():
                        reg_counts[int(row['som_y']), int(row['som_x'])] += 1
                        
                    f_reg, ax_reg = plt.subplots(figsize=(8,8))
                    ax_reg.set_aspect('equal')
                    
                    # Colormap: Rot -> Gelb -> Grün
                    from matplotlib.colors import LinearSegmentedColormap
                    # cdic = {'red':   [(0.0,  1.0, 1.0), (1.0,  0.0, 0.0)],
                    #         'green': [(0.0,  0.0, 0.0), (1.0,  1.0, 1.0)],
                    #         'blue':  [(0.0,  0.0, 0.0), (1.0,  0.0, 0.0)]}
                    # rd_gn = LinearSegmentedColormap('RdGn', cdic)
                    rd_gn = plt.cm.RdYlGn # Standard Red-Yellow-Green
                    
                    v_min, v_max = reg_counts.min(), reg_counts.max()
                    norm_reg = plt.Normalize(vmin=v_min if v_min < v_max else 0, vmax=v_max if v_max > 0 else 1)
                    
                    for y_idx in range(som_y):
                        for x_idx in range(som_x):
                            offset = 0.5 if y_idx % 2 != 0 else 0.0
                            center_x = x_idx + offset
                            center_y = y_idx * (np.sqrt(3) / 2)
                            
                            c_val = reg_counts[y_idx, x_idx]
                            fc = 'lightgrey' # Default für 0
                            if c_val > 0:
                                fc = rd_gn(norm_reg(c_val))
                            
                            hex_poly = mpatches.RegularPolygon((center_x, center_y), numVertices=6, radius=1/np.sqrt(3)*0.95,
                                                               orientation=np.radians(30), facecolor=fc, edgecolor='k', linewidth=0.5)
                            ax_reg.add_patch(hex_poly)
                            
                            # Labels [x,y] und Wert
                            ax_reg.text(center_x, center_y+0.15, f"[{x_idx+1},{y_idx+1}]", ha='center', va='center', fontsize=7, fontweight='bold', color='black')
                            ax_reg.text(center_x, center_y-0.15, f"{int(c_val)}", ha='center', va='center', fontsize=7, color='black')
                    
                    ax_reg.set_xlim(-0.5, som_x + 0.5)
                    ax_reg.set_ylim(-0.5, som_y * (np.sqrt(3)/2) + 0.5)
                    ax_reg.axis('off')
                    ax_reg.set_title(f"Anzahl Datenpunkte der ausgewählten Region: {REGION_NAME}", fontsize=12)
                    
                    # Colorbar
                    sm = plt.cm.ScalarMappable(cmap=rd_gn, norm=norm_reg)
                    plt.colorbar(sm, ax=ax_reg, label='Anzahl Datenpunkte')
                    
                    pdf.savefig(f_reg)
                    plt.close(f_reg)
"""

# --- Inject into Cell 14 (run_som_analysis) ---
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'def run_som_analysis' in source and 'create_component_planes_page(pdf, process_df_run' in source:
            lines = source.split('\n')
            new_lines = []
            for line in lines:
                new_lines.append(line)
                # Inject BEFORE component planes or after rock_type
                if 'pdf.savefig(f_rock)' in line and 'plt.close(f_rock)' in lines[lines.index(line)+1]:
                    # Wait for plt.close
                    pass 
                if 'plt.close(f_rock)' in line:
                    new_lines.append(regional_plot_code)
                    
            cell['source'] = [l + '\n' for l in "\n".join(new_lines).split('\n')]
            if len(cell['source']) > 0 and cell['source'][-1].endswith('\n'):
                cell['source'][-1] = cell['source'][-1][:-1]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated with regional density map page.")
