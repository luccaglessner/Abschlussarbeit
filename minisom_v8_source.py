import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
import seaborn as sns
import os
import itertools

from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import IncrementalPCA
from minisom import MiniSom

SESSION_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

pd.set_option('display.max_columns', None)
sns.set_theme(style="whitegrid")

# ----------------------- Hyperparameter -----------------------
SOM_ITERATIONS = 100000
SOM_SIGMA = 2.0
SOM_LEARNING_RATE = 0.9
SOM_RANDOM_SEED = 42

# ----------------------- Variablen -----------------------
SOM_X = 4
SOM_Y = 4
SOM_LOOP_LIMIT = 50

# ----------------------- Sonstige Konfiguration -----------------------
SOM_TOPOLOGY = 'hexagonal'
SOM_NEIGHBORHOOD = 'gaussian'
SOM_ACTIVATION = 'euclidean'
SOM_INPUT_LEN = None # --- Dynamisch ---

print("SOM Konfiguration geladen.")


# ------------------------- Konfiguration -------------------------
# --- MANUAL für manuelle Feature-Auswahl ---
# --- LOOP für automatische Feature-Kombination ---

EXECUTION_MODE = os.environ.get('SOM_MODE', 'MANUAL')


# ------------------------- Attribut-Auswahl (verpflichtend) -------------------------
# --- Die Hauptionen sollten immer dabei sein (FIXED_BASE_FEATURES) ---

FIXED_BASE_FEATURES = [
    "Na_in_mmol/L",
    "Mg_in_mmol/L",
    "Ca_in_mmol/L",
    "Cl_in_mmol/L",
    "SO4_in_mmol/L",
    "HCO3_in_mmol/L",
    # "total_dissolved_solids_in_mmol/L",
]


# ------------------------- Attribut-Auswahl (manuell) ------------------------- 
# --- Alle hier einkommentierten Features werden ZUSÄTZLICH zur Basis genutzt (OPTIONAL_FEATURES_POOL). ---

OPTIONAL_FEATURES_POOL = [
    # ------------------------- Physikalische Parameter -------------------------
    "temperature_in_c",
    "pH",
    "electrical_conductivity_25c_in_uS/cm",
    "redox_potential_in_mV",

    # ------------------------- Weitere Ionen / Spurenelemente -------------------------
    "K_in_mmol/L",
    "NO3_in_mmol/L",
    "Li_in_mmol/L",
    "Fe_in_mmol/L",
    "Mn_in_mmol/L",
    "HS_in_mmol/L",
    "O2_in_mmol/L",
    "Sr_in_umol/L",
    "NH4_in_umol/L",
    "F_in_umol/L",
    "H2SiO3_in_umol/L",
    
    # ------------------------- Metadaten (nur numerisch nutzen) -------------------------
    "depth_bgl_in_m",
    # "stratigraphic_period",
    # "rock_type"
]


print(f"Modus: {EXECUTION_MODE}")
print(f"Fixed Base ({len(FIXED_BASE_FEATURES)}): {FIXED_BASE_FEATURES}")
print(f"Optional Pool ({len(OPTIONAL_FEATURES_POOL)}): {OPTIONAL_FEATURES_POOL}")

# ------------------------- Basisverzeichnis (aktuelles Notebook-Verzeichnis) -------------------------
base_dir = Path.cwd()


# ------------------------- Suche nach dem neuesten Preprocessing-Ordner -------------------------
preprocessing_root = base_dir.parent.parent / "3.1_Preprocessing" / "Preprocessing"
timestamp_folders = [f for f in preprocessing_root.iterdir() if f.is_dir()]
if not timestamp_folders:
    raise FileNotFoundError(f"Keine Preprocessing-Ordner in {preprocessing_root} gefunden.")

latest_folder = max(timestamp_folders, key=lambda f: f.stat().st_mtime)
print(f"Verwendeter Preprocessing-Ordner: {latest_folder.name}")


# ------------------------- Zeitstempel finden (neuesten Ordner) -------------------------
try:
    prep_ts = datetime.strptime(latest_folder.name, "%Y-%m-%d_%H-%M-%S")
except ValueError:
    print("Warnung: Preprocessing-Ordnername entspricht nicht dem Schema. Nutze Dateisystem-Zeit.")
    prep_ts = datetime.fromtimestamp(latest_folder.stat().st_mtime)


# ------------------------- Lade Preprocessed Data -------------------------
input_path_prep = latest_folder / "Preprocessed_SOM_Ready.csv"
df_full = pd.read_csv(input_path_prep, low_memory=False)

print(f"Daten erfolgreich aus 3.1_Preprocessing geladen! Shape: {df_full.shape}")

# ------------------------- Features-Normierung aus 3.1 -------------------------

def get_training_features(user_selection, df_columns):
    training_features = []
    mapping_report = []
    
    for user_col in user_selection:
        found = False
        # ------------------------- Suche nach transformierter Version (_gauss) -------------------------
        candidates = [c for c in df_columns if c.startswith(user_col) and "_gauss" in c]
        if candidates:
            # ------------------------- Priorisiere _log_gauss -------------------------
            best_match = next((c for c in candidates if "_log_gauss" in c), candidates[0])
            training_features.append(best_match)
            mapping_report.append(f"{user_col} -> {best_match}")
            found = True
        else:
            # ------------------------- Exakter Match -------------------------
            if user_col in df_columns:
                training_features.append(user_col)
                mapping_report.append(f"{user_col} -> {user_col} (Raw)")
                found = True
        
        if not found:
             # ------------------------- Fallback -------------------------
             fuzzy = [c for c in df_columns if c.startswith(user_col) and "_gauss" in c]
             if fuzzy:
                 best_match = fuzzy[0]
                 training_features.append(best_match)
                 mapping_report.append(f"{user_col} -> {best_match} (Fuzzy)")
                 found = True
                 
        if not found:
             print(f"Feature '{user_col}' nicht gefunden! Wird ignoriert.")
             
    return training_features, mapping_report

# ------------------------- Plotting -------------------------
def plot_hex_map_to_axis(ax, data_matrix, title, cmap='viridis', label_suffix='', annotate=False, norm=None):
    sy, sx = data_matrix.shape
    ax.set_aspect('equal')
    patches = []
    colors = []
    for y in range(sy):
        for x in range(sx):
            val = data_matrix[y, x]
            if pd.isna(val): continue
            offset = 0.5 if y % 2 != 0 else 0.0
            center_x = x + offset
            center_y = y * (np.sqrt(3) / 2)
            radius = 1 / np.sqrt(3)
            hex_poly = mpatches.RegularPolygon((center_x, center_y), numVertices=6, radius=radius, orientation=np.radians(30), edgecolor='k', linewidth=0.5)
            patches.append(hex_poly)
            colors.append(val)
            if annotate:
                ax.text(center_x, center_y, f"{val:.1f}", ha='center', va='center', fontsize=7, fontweight='bold')
    if not patches: return
    collection = PatchCollection(patches, cmap=cmap, alpha=0.9, norm=norm)
    collection.set_array(np.array(colors))
    ax.add_collection(collection)
    
    # ------------------------- Präzise Grenzen zur Vermeidung von Clipping -------------------------
    radius = 1 / np.sqrt(3)
    stride_y = np.sqrt(3) / 2
    
    x_min = -radius - 0.05
    x_max = (sx - 1) + 0.5 + radius + 0.05

    y_min = -radius - 0.05
    y_max = (sy - 1) * stride_y + radius + 0.05
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis('off')
    ax.set_title(title, fontsize=12)
    return collection

def create_cluster_page(pdf_obj, df_cluster, cluster_id, train_cols, som_x, som_y, df_run):
    
    # ------------------------- Seitengröße und Layout ----------------------------

    # --- Zeilen: ---
    # --- 0. Header (12%) ---
    # --- 1. Physikalische Parameter (13%) ---
    # --- 2. Gesteinsschnitt (20%) ---
    # --- 3. SPACER (gegen Überlappung) (5%) ---
    # --- 4. Chemische zusammensetzung (Originalwerte) (45%) ---
    # --- 5. Stratigraphie (Text) (5%) ---
    
    fig = plt.figure(figsize=(11.7, 16.5)) # A4 Format
    
    # ------------------------- Gleiche Ratios -------------------------
    gs = fig.add_gridspec(6, 1, height_ratios=[0.18, 0.15, 0.20, 0.05, 0.39, 0.03], hspace=0.6)
    
    
    # ------------------------- Header Area  -------------------------
    gs_header = gs[0].subgridspec(2, 2, width_ratios=[0.55, 0.45], height_ratios=[0.4, 0.6])
    
    # ------------------------- Titel (oben links) -------------------------
    ax_title = fig.add_subplot(gs_header[0, 0])
    ax_title.axis('off')
    ax_title.text(0.0, 0.7, f"Cluster: [{cluster_id[1]},{cluster_id[0]}] (N={len(df_cluster)})", 
                  ha='left', va='center', fontsize=24, fontweight='bold')
                  

    #---------------------------- MiniMap (Oben rechts) -------------------------
    ax_map = fig.add_subplot(gs_header[1, 1])
    ax_map.set_aspect('equal')
    ax_map.axis('off')
    ax_map.set_title(f"Position (SOM {som_x}x{som_y})", fontsize=10)
    
    # ------------------------- MiniMap Logik -------------------------
    cy_idx = cluster_id[0] - 1
    cx_idx = cluster_id[1] - 1
    
    for y in range(som_y):
        for x in range(som_x):
            is_active = (x == cx_idx and y == cy_idx)
            
            offset = 0.5 if y % 2 != 0 else 0.0
            center_x = x + offset
            center_y = y * (np.sqrt(3) / 2)
            radius = 1 / np.sqrt(3)
            
            col = 'red' if is_active else '#EEEEEE'
            alpha = 1.0 if is_active else 0.3
            
            hex_poly = mpatches.RegularPolygon(
                (center_x, center_y),
                numVertices=6,
                radius=radius,
                orientation=np.radians(30), 
                facecolor=col,
                edgecolor='k',
                linewidth=0.5,
                alpha=alpha)
            ax_map.add_patch(hex_poly)
            

    # ------------------------- Fit -------------------------
    ax_map.set_xlim(-0.5, som_x + 0.5)
    ax_map.set_ylim(-0.5, som_y * (np.sqrt(3)/2) + 0.5)
    
    # ------------------------- 6 Haxgoene der Hauptionen darstellen -------------------------
    # ------------------------- Erweiterte Hexagon-Darstellung -------------------------
    
    # --- Initialisierung der Feature-Liste (Alle Viridis außer Temp) ---
    features_to_plot = []
    
    # ------------------------- 1. Haupt-Ionen -------------------------
    main_ions = ['Na_in_mmol/L', 'Mg_in_mmol/L', 'Ca_in_mmol/L', 'Cl_in_mmol/L', 'SO4_in_mmol/L', 'HCO3_in_mmol/L']
    labels_ions = ['Na', 'Mg', 'Ca', 'Cl', 'SO4', 'HCO3']
    
    for col, lbl in zip(main_ions, labels_ions):
        target = col
        for tc in train_cols:
            if tc.startswith(col):
                target = tc
                break
        features_to_plot.append({
            'col': target, 'label': lbl, 'cmap': plt.cm.viridis, 'log': False
        })
        
    # ------------------------- 2. TDS -------------------------
    if 'total_dissolved_solids_in_mmol/L' in df_run.columns:
        features_to_plot.append({
            'col': 'total_dissolved_solids_in_mmol/L', 'label': 'TDS', 'cmap': plt.cm.viridis, 'log': True
        })
        
    # ------------------------- 3. pH (Jetzt Viridis) -------------------------
    if 'pH' in df_run.columns:
        features_to_plot.append({
            'col': 'pH', 'label': 'pH', 'cmap': plt.cm.viridis, 'log': False
        })
        
    # ------------------------- 4. Sonstige (K, Fe, Mn, NO3, etc.) -------------------------
    exclude_keywords = ['som_', 'cluster', 'rock_type', 'stratigraphic', 'date', 'mesh', 'ID', 'year', 'month']
    exclude_cols = [f['col'] for f in features_to_plot] + main_ions + ['temperature_in_c', 'total_dissolved_solids_in_mmol/L', 'pH', 'depth_bgl_in_m']
    
    def is_relevant(c):
        if any(k in c for k in exclude_keywords): return False
        if c in exclude_cols: return False
        if any(c.startswith(f['col'].split('_')[0]) for f in features_to_plot if f['col']): return False
        return True

    for tc in train_cols:
        clean_name = tc.split('_log')[0].split('_gauss')[0].split('_boxcox')[0]
        if is_relevant(clean_name) and is_relevant(tc):
            lbl = clean_name.split('_in_')[0]
            features_to_plot.append({
                'col': tc, 'label': lbl, 'cmap': plt.cm.viridis, 'log': False
            })
            exclude_cols.append(tc)
            
    
    ax_ions = fig.add_subplot(gs_header[1, 0])
    ax_ions.axis('off')

    # ------------------------- Layout-Konfiguration für Hauptraster -------------------------
    hex_radius = 2.1
    spacing_x = 4.5
    spacing_y = 4.0
    cols_per_row = 6
    
    # ------------------------- PLOT MAIN GRID -------------------------
    max_x_grid = 0
    min_y_grid = 0
    
    for i, item in enumerate(features_to_plot):
        col_name = item['col']
        label = item['label']
        
        row_idx = i // cols_per_row
        col_idx = i % cols_per_row
        
        center_x = col_idx * spacing_x
        center_y = -row_idx * spacing_y
        
        # ------------------------- Grenzen verfolgen -------------------------
        max_x_grid = max(max_x_grid, center_x + hex_radius)
        min_y_grid = min(min_y_grid, center_y - hex_radius)
        
        val_mean = 0
        col_color = '#EEEEEE'
        text_col = 'black'
        
        if col_name in df_run.columns:
            agg = df_run.groupby(['som_x', 'som_y'])[col_name].mean()
            if not agg.empty:
                vmin = agg.min()
                vmax = agg.max()
                if len(df_cluster) > 0:
                    val_mean = df_cluster[col_name].mean()
                
                if item['log'] and vmin > 0 and val_mean > 0:
                     norm = LogNorm(vmin=vmin, vmax=vmax)
                else:
                     norm = plt.Normalize(vmin, vmax)
                
                cmap = item['cmap']
                col_color = cmap(norm(val_mean))
                
                # ------------------------- Logik für Viridis Textfarbe -------------------------
                n_val = norm(val_mean)
                text_col = 'white' if n_val < 0.6 else 'black'
                         
        hex_poly = mpatches.RegularPolygon((center_x, center_y), numVertices=6, radius=hex_radius,
                                           orientation=np.radians(30), 
                                           facecolor=col_color, edgecolor='k', linewidth=1)
        ax_ions.add_patch(hex_poly)
        ax_ions.text(center_x, center_y, label, ha='center', va='center', fontweight='bold', fontsize=8, color=text_col)


    # ------------------------- TEMPERATUR PLOTTEN (ISOLIERT) -------------------------
    if 'temperature_in_c' in df_run.columns:
        # --- Layout: Rechts vom Raster, größer, über ca. 2 Zeilen ---
        # --- Abstand ---
        gap = 7.0
        temp_center_x = max_x_grid + gap
        
        # --- Vertikal zentriert relativ zu genutzten Zeilen ---
        # --- Bei 1 Zeile (y=0), Mitte=0. ---
        # --- Bei 2 Zeilen (y=0, y=-2.5), Mitte=-1.25. ---
        
        # ------------------------- Rasterhöhe berechnen -------------------------
        rows_total = (len(features_to_plot) - 1) // cols_per_row + 1
        grid_height_y = (rows_total - 1) * spacing_y
        temp_center_y = -(grid_height_y / 2.0)
        
        temp_radius = hex_radius * 2.0 # 2x size
        
        col_name = 'temperature_in_c'
        val_mean = 0
        col_color = '#EEEEEE'
        text_col = 'black' # --- Standard für Temp ---
        
        agg = df_run.groupby(['som_x', 'som_y'])[col_name].mean()
        if not agg.empty:
            vmin = agg.min()
            vmax = agg.max()
            if len(df_cluster) > 0:
                val_mean = df_cluster[col_name].mean()
            
            norm = plt.Normalize(vmin, vmax)
            cmap = plt.cm.coolwarm
            col_color = cmap(norm(val_mean))
            
            # ------------------------- Coolwarm Textfarbe -------------------------
            n_val = norm(val_mean)
            text_col = 'white' if n_val < 0.3 or n_val > 0.7 else 'black'
            
        hex_poly = mpatches.RegularPolygon((temp_center_x, temp_center_y), numVertices=6, radius=temp_radius,
                                           orientation=np.radians(30), 
                                           facecolor=col_color, edgecolor='k', linewidth=1)
        ax_ions.add_patch(hex_poly)
        
        # ------------------------- Mehrzeiliges Label mit Abstand -------------------------
        ax_ions.text(temp_center_x, temp_center_y + 1.2, "Temp", ha='center', va='center', fontweight='bold', fontsize=12, color=text_col)
        ax_ions.text(temp_center_x, temp_center_y - 1.2, f"{val_mean:.1f}°C", ha='center', va='center', fontsize=11, color=text_col)
        
        # ------------------------- Breitenbegrenzung aktualisieren -------------------------
        max_x_grid = max(max_x_grid, temp_center_x + temp_radius)
        min_y_grid = min(min_y_grid, temp_center_y - temp_radius)
        
        
    # ------------------------- Grenzen anpassen -------------------------
    ax_ions.set_xlim(-3.5, max_x_grid + 1.0)
    ax_ions.set_ylim(min_y_grid - 1.0, 2.5)
    ax_ions.set_aspect('equal')
    
    
    # ------------------------- Physikalische Parameter (Reihe 1) -------------------------
    gs_phys = gs[1].subgridspec(1, 3, wspace=0.3)
    
    # ------------------------- 1. Temperatur -------------------------
    ax_temp = fig.add_subplot(gs_phys[0])
    if 'temperature_in_c' in df_cluster.columns and df_cluster['temperature_in_c'].notna().sum() > 1:
        sns.histplot(data=df_cluster, x='temperature_in_c', kde=True, ax=ax_temp, color='orange')
        ax_temp.set_title(f"Temp [°C] (Durchschnitt: {df_cluster['temperature_in_c'].mean():.1f})", fontsize=9)
        ax_temp.set_xlabel("")
        
    # ------------------------- 2. pH -------------------------
    ax_ph = fig.add_subplot(gs_phys[1])
    if 'pH' in df_cluster.columns and df_cluster['pH'].notna().sum() > 1:
        sns.histplot(data=df_cluster, x='pH', kde=True, ax=ax_ph, color='purple')
        ax_ph.set_title(f"pH [-] (Durchschnitt: {df_cluster['pH'].mean():.1f})", fontsize=9)
        ax_ph.set_xlabel("")
        
    # ------------------------- 3. Tiefe -------------------------
    ax_depth = fig.add_subplot(gs_phys[2])
    if 'depth_bgl_in_m' in df_cluster.columns and df_cluster['depth_bgl_in_m'].notna().sum() > 1:
        sns.histplot(data=df_cluster, x='depth_bgl_in_m', kde=True, ax=ax_depth, color='brown')
        ax_depth.set_title(f"Tiefe [m] (Durchschnitt: {df_cluster['depth_bgl_in_m'].mean():.1f})", fontsize=9)
        ax_depth.set_xlabel("")

    
    # ------------------------- Gesteinsarten (Reihe 2) -------------------------
    ax_bar = fig.add_subplot(gs[2])
    if 'rock_type' in df_cluster.columns:
        rock_counts = df_cluster['rock_type'].value_counts()
        new_labels = [str(l) for l in rock_counts.index]
        # ------------------------- Colorset 3 -------------------------
        unique_rocks_all = sorted(df_run['rock_type'].dropna().unique())
        pal = sns.color_palette('Set3', n_colors=12)
        rock_color_map = {rt: pal[i % 12] for i, rt in enumerate(unique_rocks_all)}
        
        # ------------------------- Labels in der Map -------------------------
        if len(new_labels) > 0:
            sns.barplot(x=new_labels, y=rock_counts.values, ax=ax_bar, hue=new_labels, palette=rock_color_map, legend=False)
            ax_bar.set_title("Gesteinsarten (Lokal)", fontsize=10)
            ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=45, ha='right')
            ax_bar.tick_params(axis='x', labelsize=8)

    # ------------------------- Chemische Verteilung (Reihe 3) -------------------------
    ax_chem_container = fig.add_subplot(gs[4])
    ax_chem_container.axis('off')
    ax_chem_container.set_title("Lokale Feature Verteilung (Originalwerte)", fontsize=12, pad=30)
    
    param_cols = train_cols[:12]
    n_plots = len(param_cols)
    
    if n_plots > 0:
        cols = 3
        rows = int(np.ceil(n_plots / cols))
        
        sub_gs = gs[4].subgridspec(rows, cols, hspace=3.0, wspace=0.6) 
        
        for i, col in enumerate(param_cols):
            row = i // cols
            col_idx = i % cols
            ax_sub = fig.add_subplot(sub_gs[row, col_idx])
            
            original_col_candidate = col.split('_log')[0].split('_gauss')[0].split('_boxcox')[0]
            
            plot_col = col 
            is_original = False
            if original_col_candidate in df_cluster.columns:
                plot_col = original_col_candidate
                is_original = True
            
            # ------------------------- Log Skalierung Logik -------------------------
            use_log = True
            if "pH" in plot_col or "temperature" in plot_col:
                use_log = False

            data_raw = df_cluster[plot_col].dropna()
            data_clean = data_raw
            if use_log:
                data_clean = data_raw[data_raw > 0]

            if len(data_clean) > 1:
                sns.histplot(data_clean, kde=True, ax=ax_sub, color='teal', element="step", log_scale=use_log)
                
                # ------------------------- Globaler Durchschnitt -------------------------
                global_mean = None
                if plot_col in df_run.columns:
                    global_mean = df_run[plot_col].mean()
                    # ------------------------- Nur plotten wenn gültig (für Log > 0)
                    if not use_log or (use_log and global_mean > 0):
                        ax_sub.axvline(global_mean, color='red', linestyle='--', linewidth=1.5, label='Global Ø')
                
                title_text = plot_col.split('_in_')[0]
                if is_original:
                    if "_in_" in plot_col:
                        unit = plot_col.split('_in_')[1]
                        title_text += f" [{unit}]"
                else:
                    title_text += " (trans.)"
                
                # ------------------------- Clipping an Rändern -------------------------
                q995 = data_clean.quantile(0.95)
                right_limit = None
                if q995 > 0:
                     right_limit = q995 * 1.1
                
                # ------------------------- Globaler Durchschnitt -------------------------
                if global_mean is not None:
                     if not use_log or (use_log and global_mean > 0):
                         if right_limit is None:
                             right_limit = max(data_clean.max(), global_mean) * 1.1
                         else:
                             if global_mean > right_limit:
                                 right_limit = global_mean * 1.1

                left_limit = 0
                if use_log:
                    left_limit = None

                ax_sub.set_xlim(left=left_limit, right=right_limit)
                    
                ax_sub.set_title(title_text, fontsize=8)
                ax_sub.set_xlabel("")
                ax_sub.set_ylabel("")
            
            
    # ------------------------- Stratigraphic Periods (Reihe 4) -------------------------
    ax_footer = fig.add_subplot(gs[5])
    ax_footer.axis('off')
    strat_text = "Keine Stratigraphie Daten"
    if 'stratigraphic_period' in df_cluster.columns:
        uniques = df_cluster['stratigraphic_period'].dropna().unique()
        if len(uniques) > 0:
            strat_text = "Stratigraphic Periods: " + ", ".join([str(s) for s in uniques])
        
    ax_footer.text(0.5, 0.5, strat_text, ha='center', va='center', fontsize=9, wrap=True)

            
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    pdf_obj.savefig(fig)
    plt.close(fig)

def create_component_planes_page(pdf_obj, df_run, train_cols, som_x, som_y):

    # ------------------------- Anzahl Plots -------------------------
    n_all = len(train_cols)
    
    # ------------------------- Layout Logik -------------------------
    # <= 6: 2x3 6 Diagramm pro Seite
    # 7-8: 2x4 8 Diagramme pro Seite
    # 9-12: 2x3 6 Diagramme je Seite, 2 Seiten
    # > 12: 2x4 8 Diagramme je Seite, 2 Seiten
    
    if n_all <= 6:
        cols = 2
        rows_grid = 3
        max_per_page = 6
    elif n_all <= 8:
        cols = 2
        rows_grid = 4
        max_per_page = 8
    elif n_all <= 12:
        cols = 2
        rows_grid = 3
        max_per_page = 6
    else:
        cols = 2
        rows_grid = 4
        max_per_page = 8

    # ------------------------- Anzahl Seiten berechnen -------------------------
    n_pages = (n_all + max_per_page - 1) // max_per_page

    for p in range(n_pages):
        start_idx = p * max_per_page
        end_idx = min(start_idx + max_per_page, n_all)
        current_batch = train_cols[start_idx:end_idx]
        
        # ------------------------- A4 Layout -------------------------
        fig = plt.figure(figsize=(11.7, 16.5)) 

        # ------------------------- Layout mit Header -------------------------
        # --- rows_grid ist fix für Konsistenz über Seiten hinweg ---
        gs = fig.add_gridspec(rows_grid + 1, cols, height_ratios=[0.05] + [1.0/rows_grid]*rows_grid)
        

        # ------------------------- Titel -------------------------
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        
        page_suffix = ""
        if n_pages > 1:
            page_suffix = f" (Seite {p+1}/{n_pages})"
            
        ax_title.text(0.5, 0.5, f"Globale Komponentenebene{page_suffix}\n(Durchschnittswert - Originale Einheiten)", 
                      ha='center', va='center', fontsize=16, fontweight='bold')
                      
        for i, feature in enumerate(current_batch):
            r = (i // cols) + 1
            c = i % cols
            ax = fig.add_subplot(gs[r, c])
            
            # ------------------------- Original-Spalte finden -------------------------
            original_col_candidate = feature.split('_log')[0].split('_gauss')[0].split('_boxcox')[0]
            
            plot_col = feature
            if original_col_candidate in df_run.columns:
                plot_col = original_col_candidate
            
            # ------------------------- Grid berechnen -------------------------
            grid = np.full((som_y, som_x), np.nan)
            agg = df_run.groupby(['som_x', 'som_y'])[plot_col].mean()
            for (gx, gy), val in agg.items():
                if gx < som_x and gy < som_y:
                    grid[gy, gx] = val
            

            # ------------------------- Name bereinigen für Titel -------------------------
            if "_in_" in plot_col:
                base, unit = plot_col.split("_in_")
                clean_name = f"{base} [{unit}]"
            else:
                clean_name = plot_col
            
            # ------------------------- Plotten -------------------------
            # --- Log-Skalierung Logik ---
            layer_norm = None
            use_log = True
            # ------------------------- pH und Temperatur von Log-Skalierung ausschließen -------------------------
            if "pH" in plot_col or "temperature" in plot_col:
                use_log = False
            
            if use_log:
                # ------------------------- <= 0 für Norm-Berechnung filtern -------------------------
                valid_vals = grid[np.isfinite(grid) & (grid > 0)]
                if len(valid_vals) > 0:
                    vmin = valid_vals.min()
                    vmax = valid_vals.max()
                    # ------------------------- LogNorm anwenden -------------------------
                    layer_norm = LogNorm(vmin=vmin, vmax=vmax)
                else:
                    pass
            
            collection = plot_hex_map_to_axis(ax, grid, clean_name, annotate=False, cmap='viridis', norm=layer_norm)
            
            
            # ---------------------------- Beschriftung: Colorbar -------------------------
            cbar = plt.colorbar(collection, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=6)
            
        plt.tight_layout()
        pdf_obj.savefig(fig)
        plt.close(fig)

def create_density_comparison_page(pdf_obj, df_run, train_cols, som_x, som_y):
    import matplotlib.ticker as ticker
    
    # ------------------------- Original Features ermitteln -------------------------
    original_features = []
    for col in train_cols:
        clean = col.split('_log')[0].split('_gauss')[0].split('_boxcox')[0]
        if clean in df_run.columns:
            if clean not in original_features:
                original_features.append(clean)
        else:
            original_features.append(col)
    
    # ------------------------- Zusatz Features (Temp & Tiefe) immer dazu -------------------------
    extras = ['temperature_in_c', 'depth_bgl_in_m']
    for e in extras:
        if e in df_run.columns:
            if e not in original_features:
                original_features.append(e)
            
    if not original_features: return
    
    # ------------------------- Layout Konfiguration & Seitenaufteilung -------------------------
    n_all = len(original_features)

    # --- Logik: ---
    # --- <=6: 2x3 Diagramm-Anordnung ---
    # --- 7-8: 2x4 Diagramm-Anordnung ---
    # --- >8:  pro Seite Bestimmung je Fall, aufgeteilt auf zwei Seiten ---
    
    if n_all <= 6:
        cols = 2
        rows_grid = 3
        max_per_page = 6
    elif n_all <= 8:
        cols = 2
        rows_grid = 4
        max_per_page = 8
    elif n_all <= 12:
        cols = 2
        rows_grid = 3
        max_per_page = 6
    else:
        cols = 2
        rows_grid = 4
        max_per_page = 8

    n_pages = (n_all + max_per_page - 1) // max_per_page

    # ------------------------- Farbpalette für Cluster -------------------------
    cluster_labels = []
    for y in range(som_y):
        for x in range(som_x):
            cluster_labels.append(f"[{x+1},{y+1}]")
            
    n_clusters = len(cluster_labels)
    palette = sns.color_palette("tab20", n_colors=n_clusters)
    color_map = {lbl: palette[i] for i, lbl in enumerate(cluster_labels)}


    for p in range(n_pages):
        start_idx = p * max_per_page
        end_idx = min(start_idx + max_per_page, n_all)
        current_batch = original_features[start_idx:end_idx]
        
        fig = plt.figure(figsize=(11.7, 16.5)) 
        gs = fig.add_gridspec(rows_grid + 1, cols, height_ratios=[0.05] + [1.0/rows_grid]*rows_grid, hspace=0.5, wspace=0.3)
        
        # ------------------------- Header Bereich -------------------------
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        
        page_suffix = ""
        if n_pages > 1:
            page_suffix = f" (Seite {p+1}/{n_pages})"
            
        ax_title.text(0.5, 0.5, f"Globale Feature Verteilung (KDE Trendlinien){page_suffix}\n(Logarithmisch skaliert, Ausnahmen: pH und Temperatur)", 
                      ha='center', va='center', fontsize=16, fontweight='bold')
                      
        
        # ------------------------- Plot-Schleife für aktuelle Seite -------------------------
        for i, feature in enumerate(current_batch):
            r = (i // cols) + 1
            c = i % cols
            ax = fig.add_subplot(gs[r, c])
            
            # ------------------------- Titel und Einheit bestimmen -------------------------
            unit = ""
            if "_in_" in feature:
                base_name, unit = feature.split('_in_')
                title = f"{base_name} [{unit}]"
            else:
                title = feature
                
            has_data = False
            
            # ------------------------- Log-Skalierung Logik -------------------------
            use_log = True
            
            # ------------------------- Ausnahme: pH ist bereits log -> Linear -------------------------
            if "pH" in feature or feature == "pH":
                 use_log = False

            # ------------------------- Ausnahme: Temperatur soll linear sein -------------------------
            if "temperature" in feature:
                 use_log = False
            
            # ------------------------- Daten plotten pro Cluster -------------------------
            for y in range(som_y):
                for x in range(som_x):
                    lbl = f"[{x+1},{y+1}]"
                    subset = df_run[(df_run['som_x'] == x) & (df_run['som_y'] == y)]
                    vals = subset[feature].dropna()
                    
                    if len(vals) > 2:
                        try:
                            if use_log:
                                # ------------------------- Filtern <= 0 für Log-Skala -------------------------
                                vals_log = vals[vals > 0]
                                if len(vals_log) > 2:
                                    sns.kdeplot(vals_log, ax=ax, color=color_map[lbl], linewidth=2.0, 
                                                label=lbl, warn_singular=False, cut=0, common_norm=False, log_scale=True)
                                    has_data = True
                            else:
                                # ------------------------- Linearer Plot (pH, Temp) -------------------------
                                sns.kdeplot(vals, ax=ax, color=color_map[lbl], linewidth=2.0, 
                                            label=lbl, warn_singular=False, cut=0, common_norm=False)
                                has_data = True
                        except Exception as e:
                            pass
            
            # ------------------------- Achsen und Beschriftungen -------------------------
            if has_data:
                ax.set_title(title, fontsize=10, fontweight='bold')
                # ------------------------- Y-Achsen Beschriftung mit Einheit -------------------------
                ylabel = "Dichte"
                if unit:
                    ylabel += f" [1/{unit}]"
                else:
                    ylabel += " [rel. Häufigkeit]"
                ax.set_ylabel(ylabel)
                
                if use_log:
                    ax.set_xscale('log')
                    # ------------------------- Spezifische Achsenbeschriftung für Tiefe -------------------------
                    if "depth" in feature:
                        ax.set_xlabel("m")
                        # ------------------------- Setze Skalar-Formatierung statt 10^x -------------------------
                        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
                        ax.ticklabel_format(style='plain', axis='x')
                    elif "K_in_mmol/L" in feature or feature == "K":
                        ax.set_xlabel("K in mmol/L")
                    elif unit:
                        ax.set_xlabel(f"{base_name} in {unit}")
                else:
                    # ------------------------- Labels für nicht-logarithmische Plots -------------------------
                    if "pH" in feature:
                        ax.set_xlabel("pH")
                    elif "temperature" in feature:
                        ax.set_xlabel("Temperatur (°C)")
                    elif "K_in_mmol/L" in feature or feature == "K":
                        ax.set_xlabel("K in mmol/L")
                    elif unit:
                        ax.set_xlabel(f"{base_name} in {unit}")
                    else:
                        ax.set_xlabel("Original Skala")
                
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
                
                # ------------------------- Legende anzeigen -------------------------
                ax.legend(fontsize=7.5, loc='upper left', ncol=2)
                
            else:
                ax.text(0.5,0.5, "Keine Daten (>0)", ha='center')
                
        plt.tight_layout()
        pdf_obj.savefig(fig)
        plt.close(fig)


def identify_cluster_profiling(df, cluster_x, cluster_y, feature_cols):
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


def create_profiling_report(pdf_path, df_run, train_cols, som_x, som_y):
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
            
    with PdfPages(pdf_path) as pdf:
        for ix in range(som_x):
            for iy in range(som_y):
                df_disc = identify_cluster_profiling(df_run, ix, iy, original_features)
                
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
                ax_header.text(0.5, 0.7, f"Cluster-Profilierung - Cluster [{ix+1},{iy+1}]", ha='center', fontsize=20, fontweight='bold')
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
                plt.close(fig)

def create_true_discrimination_report(pdf_path, df_run, train_cols, som_x, som_y):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import numpy as np
    
    # ------------------------- Original Features extrahieren -------------------------
    original_features = []
    for col in train_cols:
        clean = col.split('_log')[0].split('_gauss')[0].split('_boxcox')[0]
        if clean in df_run.columns:
            if clean not in original_features:
                original_features.append(clean)
        else:
            original_features.append(col)
            
    # ------------------------- Daten vorbereiten -------------------------
    df_clean = df_run.dropna(subset=original_features + ['som_x', 'som_y']).copy()
    if len(df_clean) < 10:
        return
        
    X = df_clean[original_features]
    y_global = df_clean.apply(lambda row: f"[{int(row['som_x'])+1},{int(row['som_y'])+1}]", axis=1)
    
    if len(y_global.unique()) <= 1:
        return
        
    with PdfPages(pdf_path) as pdf:
        # ------------------------- Globale Diskriminierung -------------------------
        rf_global = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf_global.fit(X, y_global)
        
        imp_global = pd.DataFrame({'feature': original_features, 'importance': rf_global.feature_importances_})
        imp_global['clean_name'] = imp_global['feature'].apply(lambda x: x.split('_in_')[0])
        imp_global = imp_global.sort_values(by='importance', ascending=True)
        
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[0.15, 0.85], hspace=0.3)
        
        ax_header = fig.add_subplot(gs[0])
        ax_header.axis('off')
        ax_header.text(0.5, 0.7, "Globale Cluster-Diskriminierung (Random Forest Feature Importance)", ha='center', fontsize=20, fontweight='bold')
        ax_header.text(0.5, 0.2, "Stärkste Trennung einzelner Parameter", ha='center', fontsize=14)
        
        ax_plot = fig.add_subplot(gs[1])
        colors_glob = plt.cm.viridis(imp_global['importance'] / imp_global['importance'].max())
        bars = ax_plot.barh(imp_global['clean_name'], imp_global['importance'], color=colors_glob)
        
        ax_plot.set_xlabel('Gini Feature Importance (Bedeutung für die globale Cluster-Trennung)')
        ax_plot.set_title("Einfluss der Parameter auf die gesamte Cluster-Zuordnung (Alle Cluster)", fontweight='bold')
        ax_plot.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        for bar in bars:
            width = bar.get_width()
            ax_plot.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', ha='left', va='center', fontsize=10)
        ax_plot.set_xlim(0, max(imp_global['importance']) * 1.15)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # ------------------------- Lokale Diskriminierung -------------------------
        for ix in range(som_x):
            for iy in range(som_y):
                cluster_label = f"[{ix+1},{iy+1}]"
                
                # ------------------------- Cluster in Datensatz ausfindig machen -------------------------
                if cluster_label not in y_global.values:
                    continue
                    
                # ------------------------- Cluster-Label setzen -------------------------
                y_local = (y_global == cluster_label).astype(int)
                
                # ------------------------- Nach Samples überprüfen -------------------------
                if y_local.nunique() <= 1:
                    continue
                
                # ------------------------- Random Forest auf aktuellem Cluster gegen alle anderen -------------------------
                rf_local = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
                rf_local.fit(X, y_local)
                
                imp_local = pd.DataFrame({'feature': original_features, 'importance': rf_local.feature_importances_})
                imp_local['clean_name'] = imp_local['feature'].apply(lambda x: x.split('_in_')[0])
                imp_local = imp_local.sort_values(by='importance', ascending=False) # ----------------------- Top 3 absteigend -----------------------
                
                top_3_features = imp_local.head(3)['feature'].tolist()
                
                # ----------------------- Aufsteigend sortieren -----------------------
                imp_local = imp_local.sort_values(by='importance', ascending=True)
                
                fig = plt.figure(figsize=(10, 10))
                gs = fig.add_gridspec(2, 1, height_ratios=[0.15, 0.85], hspace=0.3)
                
                ax_header = fig.add_subplot(gs[0])
                ax_header.axis('off')
                
                n_cluster = sum(y_local == 1)
                ax_header.text(0.5, 0.7, f"Spezifische Diskriminierung für Cluster {cluster_label} (N={n_cluster})", ha='center', fontsize=20, fontweight='bold', color='darkred')
                ax_header.text(0.5, 0.2, f"Charakterisierung der Parameter im Vergleich zu anderen Daten", ha='center', fontsize=14)
                
                ax_plot = fig.add_subplot(gs[1])

                # ------------------------- Orange-rote Darstellung -------------------------
                colors_loc = plt.cm.Oranges(imp_local['importance'] / (imp_local['importance'].max() + 1e-9))
                bars = ax_plot.barh(imp_local['clean_name'], imp_local['importance'], color=colors_loc)
                
                ax_plot.set_xlabel('Gini Feature Importance')
                ax_plot.set_title(f"Diskriminierende Parameter von Cluster {cluster_label}", fontweight='bold')
                ax_plot.grid(True, axis='x', linestyle='--', alpha=0.7)
                
                df_cluster_data = df_clean[y_global == cluster_label]
                
                for bar, feat_clean in zip(bars, imp_local['clean_name']):
                    width = bar.get_width()
                    # ----------------------- Originaler Feature Name -----------------------
                    feat_orig = imp_local[imp_local['clean_name'] == feat_clean]['feature'].values[0]
                    
                    text_str = f'{width:.3f}'
                    
                    if feat_orig in top_3_features and width > 0:
                         vals = df_cluster_data[feat_orig].dropna()
                         if len(vals) > 0:
                             if 'pH' not in feat_orig:
                                 import numpy as np
                                 # ----------------------- Log(1+x) verwenden, um 0 absolut sicher ohne NaN zu handhaben -----------------------
                                 vals_log = np.log1p(vals)
                                 mu_log = vals_log.mean()
                                 std_log = vals_log.std()
                                 
                                 # ----------------------- exm1 rechnet zurück (e^x - 1) -----------------------
                                 mu = np.expm1(mu_log)
                                 lower_bound = max(0, np.expm1(mu_log - std_log))
                                 upper_bound = np.expm1(mu_log + std_log)
                                 
                                 text_str += f'  (μ={mu:.2f}, σ_log={std_log:.2f}, Bereich: {lower_bound:.2f} < {feat_clean} < {upper_bound:.2f})'
                             else:
                                 mu = vals.mean()
                                 std_val = vals.std()
                                 lower_bound = max(0, mu - std_val)
                                 upper_bound = mu + std_val
                                 text_str += f'  (μ={mu:.2f}, σ={std_val:.2f}, Bereich: {lower_bound:.2f} < {feat_clean} < {upper_bound:.2f})'
                             
                    if width > 0:
                        # ----------------------- Breite der Achse -----------------------
                        max_val = imp_local['importance'].max()
                        xlim_max = (max_val * 1.5) if max_val > 0 else 1.0
                        # ----------------------- Textbreite -----------------------
                        estimated_text_width = len(text_str) * (xlim_max * 0.012)
                        
                        if width >= (max_val * 1.5) * 0.5:
                            # ----------------------- Helligkeit der Balkenfarbe berechnen -----------------------
                            rgba = bar.get_facecolor()
                            #  ----------------------- Leuchtdichte (Luminance) berechnen zur Kontrastprüfung -----------------------
                            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                            text_color = 'white' if luminance < 0.5 else 'black'
                            
                            ax_plot.text(width - 0.01, bar.get_y() + bar.get_height()/2, text_str, ha='right', va='center', fontsize=10, color=text_color)
                        else:
                            ax_plot.text(width + 0.01, bar.get_y() + bar.get_height()/2, text_str, ha='left', va='center', fontsize=10, color='black')
                
                max_val = max(imp_local['importance'])
                ax_plot.set_xlim(0, (max_val * 1.5) if max_val > 0 else 1.0) 
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


def run_som_analysis(selection_names, run_id_suffix="", som_x=None, som_y=None, custom_out_dir=None):
    import gc
    
    # ----------------------- Fallback auf Globale Parameter -----------------------
    if som_x is None: som_x = SOM_X
    if som_y is None: som_y = SOM_Y

    # ----------------------- Mapping -----------------------
    train_cols, report = get_training_features(selection_names, df_full.columns)
    print(f"\n--- Start Run: {run_id_suffix} (Size: {som_x}x{som_y}) ---")
    print("Mapping:", report)
    if not train_cols:
        print("Keine validen Features. Skip.")
        return None, None
        
    # ----------------------- Datenauswahl -----------------------
    analysis_cols = [c for c in selection_names if c in df_full.columns]
    mandatory = list(set(train_cols + analysis_cols + ['total_dissolved_solids_in_mmol/L', 'temperature_in_c', 'rock_type', 'pH', 'depth_bgl_in_m', 'stratigraphic_period', 'WGS84_latitude', 'WGS84_longitude']))
    mandatory = [c for c in mandatory if c in df_full.columns]
    
    df_run = df_full[mandatory].copy()
    
    # ----------------------- Temperaturfilter >= 10°C -----------------------
    if 'temperature_in_c' in df_run.columns:
         df_run['temperature_in_c'] = pd.to_numeric(df_run['temperature_in_c'], errors='coerce')
         cond = (df_run['temperature_in_c'] >= 10) | (df_run['temperature_in_c'].isna())
         df_run = df_run[cond]
    
    # ----------------------- Filter für komplette Daten -----------------------
    df_run.dropna(subset=train_cols, inplace=True)
    if len(df_run) < 50:
        print(f"Zu wenig Daten ({len(df_run)}). Skip.")
        return None, None
        
    # ----------------------- Skalierung -----------------------
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df_run[train_cols].values)

    # ----------------------- SOM trainieren -----------------------
    som = MiniSom(x=som_x, y=som_y, input_len=len(train_cols), sigma=SOM_SIGMA, learning_rate=SOM_LEARNING_RATE, 
                  topology=SOM_TOPOLOGY, neighborhood_function=SOM_NEIGHBORHOOD, activation_distance=SOM_ACTIVATION, random_seed=SOM_RANDOM_SEED)
    
    # ----------------------- Initialisierung des Garbage Collectors um Abstürze zu vermeiden -----------------------
    gc.collect()

    # ----------------------- Inkrementelles PCA erlaubt es, alle Daten zu nutzen, ohne den RAM zu sprengen -----------------------
    ipca = IncrementalPCA(n_components=2, batch_size=min(len(data_scaled), 2000))
    ipca.fit(data_scaled)
    
    # ----------------------- Manuelle Initialisierung der Gewichte basierend auf den ersten beiden Hauptkomponenten -----------------------
    comp = ipca.components_
    it = np.nditer(np.zeros((som_x, som_y)), flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        val_x = (idx[0] / (som_x - 1) * 2 - 1) if som_x > 1 else 0
        val_y = (idx[1] / (som_y - 1) * 2 - 1) if som_y > 1 else 0
        som._weights[idx[0], idx[1]] = ipca.mean_ + \
                                      val_x * np.sqrt(ipca.explained_variance_[0]) * comp[0] + \
                                      val_y * np.sqrt(ipca.explained_variance_[1]) * comp[1]
        it.iternext()
    
    # ----------------------- Training starten -----------------------
    som.train_random(data_scaled, SOM_ITERATIONS)
    
    # ----------------------- Fehlerberechnung -----------------------
    quantization_error = som.quantization_error(data_scaled)
    topographic_error = som.topographic_error(data_scaled)
    
    #  ---------------------- Analyse ----------------------
    winner_coords = [som.winner(x) for x in data_scaled]
    df_run['som_x'] = [c[0] for c in winner_coords]
    df_run['som_y'] = [c[1] for c in winner_coords]
    
    # ------------------------- Temperatur Karte -------------------------
    temp_map = np.full((som_y, som_x), np.nan)
    if 'temperature_in_c' in df_run.columns:
         agg = df_run.groupby(['som_x', 'som_y'])['temperature_in_c'].mean()
         for (gx, gy), val in agg.items(): temp_map[gy, gx] = val

    # ------------------------- Output Pfad -------------------------
    if custom_out_dir:
        out_dir = custom_out_dir
    else:
        out_dir = base_dir / "MiniSom_Results" / SESSION_TIMESTAMP
    out_dir.mkdir(parents=True, exist_ok=True)
    
    time_str = datetime.now().strftime("%H-%M-%S")
    clean_names = [n.split('_in_')[0] for n in selection_names]
    combo_str = "-".join(clean_names)[:50]
    safe_combo = combo_str.replace("/", "_").replace("\\", "_")
    # --------- deutscher kommentar ---------
    # df_run = Komplettsatz, df_run_oberrhein = Gefilterter Satz
    # ---------------------------------------
    # --------- deutscher kommentar ---------
    # Exportiere SOM Modell für spätere Nutzung (z.B. Lokalisierung)
    # ---------------------------------------
    import pickle
    model_data = {
        'som': som,
        'scaler': scaler,
        'train_cols': train_cols
    }
    model_path = out_dir / f"MODEL_{run_id_suffix}_{safe_combo}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"[EXPORT] Model saved: {model_path}")
    
    suffix = ""
    process_df_run = df_run
    pdf_path = out_dir / f"REPORT_{run_id_suffix}_{safe_combo}.pdf"
    with PdfPages(pdf_path) as pdf:
        # ------------------------------- Seite 1: Deckblatt -------------------------------
        plt.figure(figsize=(10,6))
        plt.text(0.5, 0.5, f"Durchlauf: {run_id_suffix}\n" + \
                   f"Features: {', '.join(clean_names)}\n" + \
                   f"Vollständige Zeilen: {len(process_df_run)}\n" + \
                   f"SOM Größe: {som_x}x{som_y}\n\n" + \
                   f"--- Parameter ---\n" + \
                   f"Iterationen: {SOM_ITERATIONS}\n" + \
                   f"Sigma: {SOM_SIGMA}\n" + \
                   f"Lernrate: {SOM_LEARNING_RATE}\n" + \
                   f"Nachbarschaft: {SOM_NEIGHBORHOOD}\n\n" + \
                   f"--- Ergebnisse ---\n" + \
                   f"Quantisierungsfehler: {quantization_error:.4f}\n" + \
                   f"Topographischer Fehler: {topographic_error:.4f}", ha='center', fontsize=12)
        plt.axis('off')
        pdf.savefig()
        plt.close()
    
        # ------------------------------- Seite 1b: TDS Karte -------------------------------
        if 'total_dissolved_solids_in_mmol/L' in process_df_run.columns:
            f_tds, ax_tds = plt.subplots(figsize=(8,8))
        
            tds_map = np.full((som_y, som_x), np.nan)
            agg_tds = process_df_run.groupby(['som_x', 'som_y'])['total_dissolved_solids_in_mmol/L'].mean()
            for (gx, gy), val in agg_tds.items():
                 if gx < som_x and gy < som_y: tds_map[gy, gx] = val
        
            valid_vals = process_df_run[process_df_run['total_dissolved_solids_in_mmol/L'] > 0]['total_dissolved_solids_in_mmol/L']
            norm = None
            if len(valid_vals) > 0:
                 norm = LogNorm(vmin=valid_vals.min(), vmax=valid_vals.max())
        
            col_tds = plot_hex_map_to_axis(ax_tds, tds_map, "Total Dissolved Solids (TDS) [mmol/L] (Logarithmisch)", 
                                           annotate=False, cmap='viridis', norm=norm)
        
            for y in range(som_y):
                for x in range(som_x):
                     val = tds_map[y, x]
                     if pd.isna(val): continue
                 
                     offset = 0.5 if y % 2 != 0 else 0.0
                     center_x = x + offset
                     center_y = y * (np.sqrt(3) / 2)
                 
                     txt_col = 'black'
                     if norm and val > 0:
                         if norm(val) < 0.5: txt_col = 'white'
                         
                     ax_tds.text(center_x, center_y+0.15, f"[{x+1},{y+1}]", ha='center', va='center', fontsize=6, fontweight='bold', color=txt_col)
                     ax_tds.text(center_x, center_y-0.15, f"{val:.1f}", ha='center', va='center', fontsize=6, color=txt_col)
        
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax_tds)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(col_tds, cax=cax, label="TDS [mmol/L]")
            f_tds.text(0.5, 0.02, "Lesehilfe: [Zeile, Spalte]. Farbskala: Lila (Niedrig) > Blau > Grün > Gelb (Hoch)", ha='center', fontsize=8)
            pdf.savefig(f_tds)
            plt.close(f_tds)
    
        # ------------------------------- Seite 2: Temperatur Karte -------------------------------
        f, ax = plt.subplots(figsize=(8,8))
        if 'temperature_in_c' in process_df_run.columns:
             col_t = plot_hex_map_to_axis(ax, temp_map, "Durchschnittstemperatur Übersicht", annotate=False, cmap='coolwarm')
             for y in range(som_y):
                for x in range(som_x):
                     offset = 0.5 if y % 2 != 0 else 0.0
                     center_x = x + offset
                     center_y = y * (np.sqrt(3) / 2)
                     idx_label = f"[{x+1},{y+1}]"
                     val = temp_map[y, x]
                     val_str = f"{val:.1f}" if not pd.isna(val) else "-"
                     ax.text(center_x, center_y+0.15, idx_label, ha='center', va='center', fontsize=6, fontweight='bold', color='black')
                     ax.text(center_x, center_y-0.15, val_str, ha='center', va='center', fontsize=6, color='blue')
             from mpl_toolkits.axes_grid1 import make_axes_locatable
             divider = make_axes_locatable(ax)
             cax = divider.append_axes("right", size="5%", pad=0.05)
             plt.colorbar(col_t, cax=cax, label="Durchschnittstemperatur [°C]")
             f.text(0.5, 0.05, "Lesehilfe: [Zeile, Spalte]. Blau = Temp.", ha='center')
        pdf.savefig(f)
        plt.close(f)

        # ------------------------------- Seite 3: Gesteinstyp -------------------------------
        if 'rock_type' in process_df_run.columns:
            f_rock, ax_rock = plt.subplots(figsize=(8,8))
            dom_rocks = process_df_run.groupby(['som_x', 'som_y'])['rock_type'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
            unique_rocks = sorted(process_df_run['rock_type'].dropna().unique())
            pal = sns.color_palette('Set3', n_colors=12)
            rock_color_map = {rt: pal[i % 12] for i, rt in enumerate(unique_rocks)}
            legend_handles = []
            seen_rocks = set()
            ax_rock.set_aspect('equal')
            for y in range(som_y):
                for x in range(som_x):
                    rock = dom_rocks.get((x, y), None)
                    offset = 0.5 if y % 2 != 0 else 0.0
                    center_x = x + offset
                    center_y = y * (np.sqrt(3) / 2)
                    radius = (1 / np.sqrt(3)) * 0.95
                    fc = 'lightgrey'
                    if rock and rock in rock_color_map:
                        fc = rock_color_map[rock]
                        if rock not in seen_rocks:
                             legend_handles.append(mpatches.Patch(color=fc, label=str(rock)))
                             seen_rocks.add(rock)
                    hex_poly = mpatches.RegularPolygon((center_x, center_y), numVertices=6, radius=radius,
                                                       orientation=np.radians(30), 
                                                       facecolor=fc, edgecolor='k', linewidth=0.5, alpha=0.9)
                    ax_rock.add_patch(hex_poly)
                    idx_label = f"[{x+1},{y+1}]"
                    rock_label = str(rock) if rock else "-"
                    # ----------------------- Textumbruch -----------------------
                    if len(rock_label) > 10 and ' ' in rock_label:
                        parts = rock_label.split(' ')
                        if len(parts) > 1:
                            mid = len(parts) // 2
                            rock_label = ' '.join(parts[:mid]) + '\n' + ' '.join(parts[mid:])
                    elif len(rock_label) > 12:
                        # ----------------------- Fallback für lange Einzelwörter -----------------------
                        pass
                    ax_rock.text(center_x, center_y+0.15, idx_label, ha='center', va='center', fontsize=7, fontweight='bold', color='black')
                    ax_rock.text(center_x, center_y-0.15, rock_label, ha='center', va='center', fontsize=6, color='black')
            ax_rock.set_xlim(-1.0, som_x + 0.5)
            ax_rock.set_ylim(-0.5, som_y * (np.sqrt(3)/2) + 0.5)
            ax_rock.axis('off')
            ax_rock.set_title("Dominante Gesteinsarten Übersicht", fontsize=14)
            ax_rock.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=8)
            pdf.savefig(f_rock)
            plt.close(f_rock)

            # ------------------------------- NEU: Regionaler Dichtevergleich (Seite 4) -------------------------------
            REGION_MODE = os.environ.get('SOM_REGION')
            if REGION_MODE and REGION_MODE != 'ALL':
                REGION_NAME = os.environ.get('LOC_REGION_NAME', 'Oberrheingraben')
                LAT_MIN = float(os.environ.get('LOC_LAT_MIN', 47.30))
                LAT_MAX = float(os.environ.get('LOC_LAT_MAX', 50.28))
                LON_MIN = float(os.environ.get('LOC_LON_MIN', 7.45))
                LON_MAX = float(os.environ.get('LOC_LON_MAX', 9.60))
                
                # Filter Region
                mask_region = (process_df_run['WGS84_latitude'] >= LAT_MIN) & (process_df_run['WGS84_latitude'] <= LAT_MAX) & \
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


        create_component_planes_page(pdf, process_df_run, train_cols, som_x, som_y)
    
        for ix in range(som_x):
            for iy in range(som_y):
                mask = (process_df_run['som_x'] == ix) & (process_df_run['som_y'] == iy)
                df_cluster = process_df_run[mask]
                create_cluster_page(pdf, df_cluster, (iy+1, ix+1), train_cols, som_x, som_y, process_df_run)
                plt.close('all')
                gc.collect()
         
        create_density_comparison_page(pdf, process_df_run, train_cols, som_x, som_y)
    
    # ------------------------- Neue Diskriminierungs-PDF generieren -------------------------
    disc_pdf_path = out_dir / f"PROFILING_{run_id_suffix}_{safe_combo}{suffix}.pdf"
    create_profiling_report(disc_pdf_path, process_df_run, train_cols, som_x, som_y)
    print(f"Profiling-Report saved: {disc_pdf_path}")
    # ------------------------- Echte Diskriminierungs-PDF (Random Forest) -------------------------
    true_disc_pdf_path = out_dir / f"DISCRIMINATION_{run_id_suffix}_{safe_combo}{suffix}.pdf"
    create_true_discrimination_report(true_disc_pdf_path, process_df_run, train_cols, som_x, som_y)
    print(f"True Discrimination-Report saved: {true_disc_pdf_path}")
    print(f"Report saved: {pdf_path}")
    del som, data_scaled, df_run, temp_map
    plt.close('all')
    gc.collect()
    
    return quantization_error, topographic_error





# ------------------------------- VORBEREITUNG DER KOMBINATIONEN -------------------------------
base = FIXED_BASE_FEATURES
pool = OPTIONAL_FEATURES_POOL
combos_all = [[]] # --- Basis ohne Extras ---
labels_all = ["Base_without_pH"]

max_r = 3
for r in range(1, max_r + 1):
    for subset in itertools.combinations(pool, r):
        combos_all.append(list(subset))
        addon_names = [n.split('_in_')[0] for n in subset]
        lbl_temp = "Plus_" + "-".join(addon_names)
        if lbl_temp == "Plus_pH": lbl_temp = "Base_with_pH"
        labels_all.append(lbl_temp)
        
combos_all.append(["pH", "K_in_mmol/L", "Fe_in_mmol/L", "Mn_in_mmol/L"])
labels_all.append("Plus_pH-K-Fe-Mn")

# ------------------------- Separate Filterung -------------------------
print("\nBerechne Datenverfügbarkeit für Sortierung...")
combo_stats = []
for idx, combo in enumerate(combos_all):
    lbl = labels_all[idx]
    if lbl == "Base_without_pH": count = 999999999
    elif lbl == "Base_with_pH": count = 999999998
    elif lbl == "Plus_pH-Fe": count = 999999997
    elif lbl == "Plus_pH-K-Fe": count = 999999996
    elif lbl == "Plus_pH-K-Fe-Mn": count = 999999995
    elif lbl == "Plus_temperature": count = 999999994
    else:
        current_selection = FIXED_BASE_FEATURES + list(combo)
        t_cols, _ = get_training_features(current_selection, df_full.columns)
        count = df_full[t_cols].dropna().shape[0] if t_cols else 0

    combo_stats.append({'combo': combo, 'label': lbl, 'count': count})

combo_stats.sort(key=lambda x: x['count'], reverse=True)
sorted_combos = [x['combo'] for x in combo_stats]
sorted_labels = [x['label'] for x in combo_stats]

# ------------------------------- Liste mit Separatoren -------------------------------
display_combos = sorted_combos.copy()
display_labels = sorted_labels.copy()

if len(display_labels) > 0 and display_labels[0] == "Base_without_pH": display_labels[0] = "Test-Run"
if len(display_combos) >= 1:
    display_combos.insert(1, ["SEPARATOR"])
    display_labels.insert(1, "Basis_Durchlaeufe")
if len(display_combos) >= 7:
    display_combos.insert(7, ["SEPARATOR"])
    display_labels.insert(7, "Zusatzinformationen")

# ------------------------------- MODUS: MANUAL -------------------------------
if EXECUTION_MODE == 'MANUAL':
    print("\n=== Modus: MANUAL (Einträge: bis Report_010) ===")
    out_dir = base_dir / "MiniSom_Results" / SESSION_TIMESTAMP
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(min(11, len(display_combos))):
        # --------- INJECTED: SINGLE RUN FILTER ---------
        target_idx = os.environ.get('SOM_TARGET_INDEX')
        if target_idx is not None and i != int(target_idx): continue
        # -----------------------------------------------
        combo = display_combos[i]
        run_label = display_labels[i]
        run_id = f"{i:03d}_{run_label}"
        
        if combo == ["SEPARATOR"]:
            print(f"\n{'='*20} {run_label.upper()} {'='*20}")
            sep_path = out_dir / f"Report_{run_id}_Trennlinie.txt"
            with open(sep_path, "w") as f: f.write(f"Trennlinie: {run_label}")
            continue
            
        print(f"\n--- Schritt {i+1}: {run_id} ---")
        current_selection = list(base) + list(combo)
        run_som_analysis(current_selection, run_id)
        plt.close('all')

# ------------------------------- MODUS: LOOP -------------------------------
elif EXECUTION_MODE == 'LOOP':
    print("\n=== Modus: LOOP ===")
    run_counter = 0
    out_dir_base = base_dir / "MiniSom_Results" / SESSION_TIMESTAMP
    out_dir_base.mkdir(parents=True, exist_ok=True)
    
    for i, combo in enumerate(display_combos):
        # --------- INJECTED: SINGLE RUN FILTER ---------
        target_idx = os.environ.get('SOM_TARGET_INDEX')
        if target_idx is not None and i != int(target_idx): continue
        # -----------------------------------------------
        if run_counter >= SOM_LOOP_LIMIT: break
        run_label = display_labels[i]
        run_id = f"{i:03d}_{run_label}"
        
        if combo == ["SEPARATOR"]:
            sep_path = out_dir_base / f"Report_{run_id}_Trennlinie.txt"
            with open(sep_path, "w") as f: f.write(f"Trennlinie: {run_label}")
            continue
            
        print(f"\n--- Schritt {run_counter+1}: {run_id} ---")
        current_selection = list(base) + list(combo)
        run_som_analysis(current_selection, run_id)
        run_counter += 1
        plt.close('all')

# ------------------------------- MODUS: SIZE_ITERATIONS -------------------------------
elif EXECUTION_MODE == 'SIZE_ITERATIONS':
    print("\n=== Modus: SIZE_ITERATIONS (2x2 bis 10x10) ===")
    all_errors = []
    
    for size in range(2, 11):
        s_x, s_y = size, size
        print(f"\n{'#'*30} STARTE GRÖSSE {s_x}x{s_y} {'#'*30}")
        
        size_dir = base_dir / "MiniSom_Results" / SESSION_TIMESTAMP / f"Size_{s_x}x{s_y}"
        size_dir.mkdir(parents=True, exist_ok=True)
        
        q_errs = []
        t_errs = []
        
        for i in range(min(11, len(display_combos))):
            # --------- INJECTED: SINGLE RUN FILTER ---------
            target_idx = os.environ.get('SOM_TARGET_INDEX')
            if target_idx is not None and i != int(target_idx): continue
            # -----------------------------------------------
            combo = display_combos[i]
            run_label = display_labels[i]
            run_id = f"{i:03d}_{run_label}"
            
            if combo == ["SEPARATOR"]:
                sep_path = size_dir / f"Report_{run_id}_Trennlinie.txt"
                with open(sep_path, "w") as f: f.write(f"Trennlinie: {run_label}")
                continue
                
            current_selection = list(base) + list(combo)
            qe, te = run_som_analysis(current_selection, run_id, som_x=s_x, som_y=s_y, custom_out_dir=size_dir)
            
            if qe is not None:
                q_errs.append(qe)
                t_errs.append(te)
            plt.close('all')
            
        if q_errs:
            all_errors.append({
                'size': s_x,
                'q_mean': np.mean(q_errs),
                't_mean': np.mean(t_errs)
            })

        # ------------------------------- Erstelle Fehler-Zusammenfassung PDF (Inkrementell) -------------------------------
        if all_errors:
            df_err = pd.DataFrame(all_errors)
            summary_pdf = base_dir / "MiniSom_Results" / SESSION_TIMESTAMP / "Size_Iteration_Errors.pdf"
            
            try:
                with PdfPages(summary_pdf) as pdf:
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    color1 = 'tab:blue'
                    ax1.set_xlabel('Grid Size (NxN)')
                    ax1.set_ylabel('Quantization Error', color=color1)
                    ax1.plot(df_err['size'], df_err['q_mean'], marker='o', color=color1, label='Quantization Error')
                    ax1.tick_params(axis='y', labelcolor=color1)
                    ax1.grid(True, linestyle='--', alpha=0.7)
                    
                    ax2 = ax1.twinx()
                    color2 = 'tab:red'
                    ax2.set_ylabel('Topographic Error', color=color2)
                    ax2.plot(df_err['size'], df_err['t_mean'], marker='s', color=color2, label='Topographic Error')
                    ax2.tick_params(axis='y', labelcolor=color2)
                    
                    plt.title(f'Durchschnittliche Fehler über SOM-Größen (Status: {s_x}x{s_y})')
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
                print(f"\\n[UPDATE] Fehler-Zusammenfassung aktualisiert: {summary_pdf}")
            except Exception as e:
                print(f"\\n[WARNUNG] Konnte PDF nicht speichern (evtl. geöffnet?): {e}")

print("\nDone. Alle Läufe abgeschlossen.")

