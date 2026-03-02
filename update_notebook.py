import json
import os

notebook_path = r"f:\Abschlussarbeit\Abschlussarbeit Bearbeitung\Jupyter Notebooks\3_Machine-Learning\3.2_Machine-Learning\MiniSom\MiniSom_Machine-Learning.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_code = """def identify_cluster_discrimination(df, cluster_x, cluster_y, feature_cols):
    from scipy.stats import mannwhitneyu
    import numpy as np
    import pandas as pd
    
    # ------------------------- Cluster vs Rest -------------------------
    df_cluster = df[(df['som_x'] == cluster_x) & (df['som_y'] == cluster_y)]
    df_rest = df[~((df['som_x'] == cluster_x) & (df['som_y'] == cluster_y))]
    
    results = []
    for col in feature_cols:
        vals_cluster = df_cluster[col].dropna()
        vals_rest = df_rest[col].dropna()
        vals_all = df[col].dropna()
        
        if len(vals_cluster) < 3 or len(vals_rest) < 3:
            continue
            
        mean_cluster = vals_cluster.mean()
        mean_rest = vals_rest.mean()
        mean_all = vals_all.mean()
        std_all = vals_all.std()
        
        if std_all == 0 or pd.isna(std_all):
            continue
            
        z_score = (mean_cluster - mean_all) / std_all
        
        # ------------------------- Mann-Whitney U Test -------------------------
        try:
            stat, p_val = mannwhitneyu(vals_cluster, vals_rest, alternative='two-sided')
        except Exception:
            p_val = 1.0
            
        factor = mean_cluster / mean_rest if mean_rest != 0 else np.nan
        
        results.append({
            'feature': col,
            'z_score': z_score,
            'p_value': p_val,
            'mean_cluster': mean_cluster,
            'mean_rest': mean_rest,
            'factor': factor
        })
        
    df_res = pd.DataFrame(results)
    if df_res.empty:
        return df_res
        
    # ------------------------- Sortierung nach Einzigartigkeit (Absoluter Z-Score und p-Wert) -------------------------
    df_res['abs_z'] = df_res['z_score'].abs()
    df_res = df_res.sort_values(by=['p_value', 'abs_z'], ascending=[True, False])
    return df_res


def create_discrimination_report(pdf_path, df_run, train_cols, som_x, som_y):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    
    # ------------------------- Original Features extrahieren -------------------------
    original_features = []
    for col in train_cols:
        clean = col.split('_log')[0].split('_gauss')[0].split('_boxcox')[0]
        if clean in df_run.columns:
            if clean not in original_features:
                original_features.append(clean)
        else:
            original_features.append(col)
            
    extras = ['temperature_in_c', 'depth_bgl_in_m', 'pH']
    for e in extras:
        if e in df_run.columns and e not in original_features:
            original_features.append(e)
            
    with PdfPages(pdf_path) as pdf:
        for iy in range(som_y):
            for ix in range(som_x):
                df_disc = identify_cluster_discrimination(df_run, ix, iy, original_features)
                
                if df_disc.empty:
                    continue
                    
                fig = plt.figure(figsize=(11.7, 16.5))
                
                # ------------------------- A4 Layout konfigurieren -------------------------
                gs = fig.add_gridspec(3, 1, height_ratios=[0.1, 0.4, 0.5], hspace=0.3)
                
                # ------------------------- Header -------------------------
                ax_header = fig.add_subplot(gs[0])
                ax_header.axis('off')
                
                mask = (df_run['som_x'] == ix) & (df_run['som_y'] == iy)
                n_cluster = mask.sum()
                ax_header.text(0.5, 0.7, f"Diskriminierungseigenschaften - Cluster [{ix+1},{iy+1}]", ha='center', fontsize=20, fontweight='bold')
                ax_header.text(0.5, 0.3, f"Anzahl Messungen (N): {n_cluster}", ha='center', fontsize=14)
                
                # ------------------------- Tabelle: Top Diskriminatoren -------------------------
                ax_table = fig.add_subplot(gs[1])
                ax_table.axis('off')
                
                top_n = min(10, len(df_disc))
                df_top = df_disc.head(top_n).copy()
                
                table_data = []
                table_data.append(["Feature", "Z-Score", "p-Wert", "Faktor (vs. Rest)", "Ø Cluster", "Ø Rest"])
                
                for _, row in df_top.iterrows():
                    feat = row['feature'].split('_in_')[0]
                    z = f"{row['z_score']:.2f}"
                    p = f"{row['p_value']:.4e}" if row['p_value'] < 0.001 else f"{row['p_value']:.4f}"
                    f = f"{row['factor']:.2f}x"
                    mc = f"{row['mean_cluster']:.2f}"
                    mr = f"{row['mean_rest']:.2f}"
                    table_data.append([feat, z, p, f, mc, mr])
                    
                table = ax_table.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1, 2.0)
                
                # ------------------------- Header der Tabelle hervorheben -------------------------
                for j in range(6):
                    cell = table[0, j]
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#dddddd')
                    
                # ------------------------- Visualisierung: Z-Scores Barplot -------------------------
                ax_plot = fig.add_subplot(gs[2])
                
                # ------------------------- Farben basierend auf Z-Score (+ = rot, - = blau) -------------------------
                colors = ['#d62728' if z > 0 else '#1f77b4' for z in df_top['z_score']]
                feat_names = [f.split('_in_')[0] for f in df_top['feature']]
                
                ax_plot.barh(feat_names[::-1], df_top['z_score'].values[::-1], color=colors[::-1])
                ax_plot.set_xlabel('Z-Score (Abweichung vom globalen Mittelwert)')
                ax_plot.set_title(f"Top {top_n} Unterscheidungsmerkmale (Z-Scores)", fontweight='bold')
                ax_plot.axvline(0, color='black', linewidth=1)
                ax_plot.grid(True, axis='x', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)"""

source_lines = [line + "\\n" for line in new_code.split('\n')]
# Remove the trailing \n from the very last line to perfectly match jupyter style
source_lines[-1] = source_lines[-1].replace("\\n", "")
# Wait, actually splitting '\n' drops the newline. Let's just do:
source_lines = []
lines = new_code.split('\n')
for i, line in enumerate(lines):
    if i < len(lines) - 1:
        source_lines.append(line + "\n")
    else:
        source_lines.append(line)

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "discrimination_report_logic",
    "metadata": {},
    "outputs": [],
    "source": source_lines
}

# Find the run_som_analysis cell
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        if any("def run_som_analysis(" in line for line in cell.get('source', [])):
            insert_idx = i
            break

if insert_idx != -1:
    nb['cells'].insert(insert_idx, new_cell)
    print("New cell inserted!")
else:
    print("Could not find run_som_analysis cell!")

run_som_cell = nb['cells'][insert_idx + 1] # It's shifted by 1
new_source = []
for line in run_som_cell['source']:
    if "print(f\"Report saved: {pdf_path}\")" in line:
        indent = "    "
        new_source.append(indent + "# ------------------------- Neue Diskriminierungs-PDF generieren -------------------------\n")
        new_source.append(indent + "disc_pdf_path = out_dir / f\"Report_{run_id_suffix}_{safe_combo}_Discrimination.pdf\"\n")
        new_source.append(indent + "create_discrimination_report(disc_pdf_path, df_run, train_cols, som_x, som_y)\n")
        new_source.append(indent + "print(f\"Diskriminierungs-Report saved: {disc_pdf_path}\")\n")
    new_source.append(line)

run_som_cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    
print("Notebook updated.")
