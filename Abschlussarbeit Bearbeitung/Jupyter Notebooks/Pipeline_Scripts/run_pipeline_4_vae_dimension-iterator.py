import os
import subprocess
import sys
import time
import pipeline_logger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
import shutil
import nbformat
from nbconvert import PythonExporter
import numpy as np
import threading
import math
from sklearn.metrics import mean_squared_error, r2_score

from pathlib import Path
from datetime import datetime

# ------------------------- Konfiguration -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# ------------------------- Parameter -------------------------
FIXED_BATCH_SIZE = 128
FIXED_EPOCHS = 50
FIXED_KLD_WEIGHT = 0.05

# ------------------------- Loops -------------------------
LATENT_DIMS = [2, 4, 8, 16, 32]
HIDDEN_DIMS = [32, 64, 128, 256, 512]

# ------------------------- Daten Vorschau Helfer -------------------------
def get_latest_preprocessing_file(base_dir):
    # ------------------------- Logik kopiert und angepasst aus VAE_Imputation.ipynb -------------------------
    
    # ------------------------- Helfer: Projektwurzel vom Skriptort finden -------------------------
    script_path = Path(__file__).parent
    
    # ------------------------- Suche nach "3_Machine-Learning" von den Eltern des Skripts aus -------------------------
    root = script_path
    found_prep = None
    for _ in range(4):
        if (root / "3_Machine-Learning").exists():
            found_prep = root / "3_Machine-Learning" / "3.1_Preprocessing" / "Preprocessing"
            break
        root = root.parent
        
    if not found_prep or not found_prep.exists():
        # ------------------------- Fallback fest codiert für bekannte Pfadstruktur -------------------------
        root = Path("f:/Abschlussarbeit/Abschlussarbeit Bearbeitung")
        found_prep = root / "3_Machine-Learning" / "3.1_Preprocessing" / "Preprocessing"
        
    if not found_prep.exists():
        print(f"WARNUNG: Preprocessing Ordner nicht gefunden unter {found_prep}")
        return None

    timestamp_folders = [f for f in found_prep.iterdir() if f.is_dir()]
    if not timestamp_folders:
        return None
        
    latest_folder = max(timestamp_folders, key=lambda f: f.stat().st_mtime)
    return latest_folder / "Preprocessed_SOM_Ready.csv"

def get_training_features(user_selection, df_columns):
    # ------------------------- Kopiert aus VAE_Imputation.ipynb -------------------------
    training_features = []
    
    for user_col in user_selection:
        found = False
        candidates = [c for c in df_columns if c.startswith(user_col) and "_gauss" in c]
        if candidates:
            # ------------------------- Priorisierung von _log_gauss -------------------------
            best_match = next((c for c in candidates if "_log_gauss" in c), candidates[0])
            training_features.append(best_match)
            found = True
        else:
            if user_col in df_columns:
                training_features.append(user_col)
                found = True
        
        if not found:
            # ------------------------- Fallback -------------------------
             fuzzy = [c for c in df_columns if c.startswith(user_col) and "_gauss" in c]
             if fuzzy:
                 best_match = fuzzy[0]
                 training_features.append(best_match)
                 found = True
    return training_features

def get_run_preview_info():
    csv_path = get_latest_preprocessing_file(None)
    if not csv_path or not csv_path.exists():
        return "Konnte Daten nicht laden (Pfadfehler)."
        
    print(f"  -> Lade Datenvorschau von: {csv_path.name}...")
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        return f"Fehler beim Laden: {e}"

    FIXED_BASE_FEATURES = [
        "Na_in_mmol/L", "Mg_in_mmol/L", "Ca_in_mmol/L", 
        "Cl_in_mmol/L", "SO4_in_mmol/L", "HCO3_in_mmol/L"
    ]
    
    
    # ------------------------- Definitionen entsprechend der Logik -------------------------
    runs = [
        {"name": "Base_without_pH", "add": []},
        {"name": "Base_with_pH",    "add": ["pH"]},
        {"name": "Plus_pH-Fe",      "add": ["pH", "Fe_in_mmol/L"]},
        {"name": "Plus_pH-K-Fe",    "add": ["pH", "K_in_mmol/L", "Fe_in_mmol/L"]},
        {"name": "Plus_pH-K-Fe-Mn", "add": ["pH", "K_in_mmol/L", "Fe_in_mmol/L", "Mn_in_mmol/L"]},
        {"name": "Plus_temperature", "add": ["temperature_in_c"]}
    ]
    
    results = []
    for i, r in enumerate(runs):
        selection = FIXED_BASE_FEATURES + r["add"]
        t_cols = get_training_features(selection, df.columns)
        count = df[t_cols].dropna().shape[0]
        results.append({
            "idx": i,
            "name": r["name"],
            "count": count,
            "features": len(t_cols)
        })
        
    return results

# ------------------------- Hilfsfunktionen -------------------------
# ------------------------- Pfade zu den Notebooks -------------------------
NOTEBOOK_4_1 = BASE_DIR / "4_Imputation" / "4.1_VAE_Imputation" / "VAE_Imputation.ipynb"
NOTEBOOK_4_2 = BASE_DIR / "4_Imputation" / "4.2_Inference" / "Inference.ipynb"
NOTEBOOK_4_3 = BASE_DIR / "4_Imputation" / "4.3_Evaluation" / "Evaluation.ipynb"

# ------------------------- Globaler Status -------------------------
results_history = []
history_lock = threading.Lock()

def run_notebook_with_params(notebook_path, params_dict, run_name):
    print(f"\n--- Starte: {run_name} ---")
    
    # ------------------------- Notebook zu Python-Code -------------------------
    print("  -> Lade Code aus Notebook...")
    convert_cmd = ["jupyter", "nbconvert", "--to", "python", "--stdout", str(notebook_path)]
    try:
        python_code = subprocess.check_output(convert_cmd, cwd=notebook_path.parent).decode('utf-8', errors='ignore')
    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Konvertieren von {notebook_path}: {e}")
        return False, None

    # ------------------------- Magics entfernen -------------------------
    filtered_code_lines = []
    for line in python_code.splitlines():
        if "get_ipython()" in line:
            filtered_code_lines.append(f"# {line}  # Filtered Magic")
        else:
            filtered_code_lines.append(line)
    
    final_code = "\n".join(filtered_code_lines)


    # ------------------------- Parameter Injektion -------------------------
    if params_dict:
        print("  -> Injiziere Parameter...")
        
        # ------------------------- 1. Parameter (Variablen) - Am Anfang von Main injizieren -------------------------
        injection_code_params = "\n# --- INJECTED PARAMETERS ---\n"
        for key, value in params_dict.items():
            if key == "MAX_ITERATIONS": 
                injection_code_params += f"{key} = {value}\n"
            elif key == "TARGET_RUN_INDEX":
                injection_code_params += f"{key} = {value}\n"
            elif isinstance(value, str):
                injection_code_params += f"{key} = '{value}'\n"
            else:
                injection_code_params += f"{key} = {value}\n"
        
        if "TARGET_RUN_INDEX" not in params_dict:
             injection_code_params += "TARGET_RUN_INDEX = 0\n"

        if 'if __name__ == "__main__":' in final_code:
            final_code = final_code.replace('if __name__ == "__main__":', f'{injection_code_params}\nif __name__ == "__main__":')
        else:
            final_code += injection_code_params

        # ------------------------- 2. Logik Injektion (Filtern der Liste) -------------------------
        if params_dict.get("MAX_ITERATIONS") == 1:
             indent = "    "
             target_run_idx = params_dict.get("TARGET_RUN_INDEX", 0)
             
             # ------------------------- Robuste Logik: Loop-Start suchen und Filter davor einfügen -------------------------
             # ------------------------- Berücksichtigung von Separatoren im VAE_Imputation Code -------------------------
             logic_code = f"\n{indent}# --- INJECTED LOGIC (FILTERING) ---\n"
             logic_code += f"{indent}target_idx_usr = {target_run_idx}\n"
             logic_code += f"{indent}# Mapping User Index -> Real Index (mit Separatoren)\n"
             logic_code += f"{indent}real_idx = target_idx_usr\n"
             logic_code += f"{indent}if real_idx > 0: real_idx += 1 # Skip First Separator after Test-Run\n"
             logic_code += f"{indent}if real_idx > 5: real_idx += 1 # Skip Second Separator after Basis Runs\n"
             logic_code += f"{indent}print(f'DEBUG: User Index {{target_idx_usr}} -> Real Index {{real_idx}} (wegen Separatoren)')\n"
             logic_code += f"{indent}if real_idx < len(sorted_combos):\n"
             logic_code += f"{indent}    sorted_combos = [sorted_combos[real_idx]]\n"
             logic_code += f"{indent}    sorted_labels = [sorted_labels[real_idx]]\n"
             logic_code += f"{indent}else:\n"
             logic_code += f"{indent}    print(f'WARNUNG: Real Index {{real_idx}} ausserhalb. Fallback auf 0.')\n"
             logic_code += f"{indent}    sorted_combos = [sorted_combos[0]]\n"
             logic_code += f"{indent}    sorted_labels = [sorted_labels[0]]\n"
             logic_code += f"{indent}# --------------------------------------\n"
             
             # ------------------------- Suche nach dem Loop-Start -------------------------
             loop_line = "for i, combo in enumerate(sorted_combos):"
             
             if loop_line in final_code:
                 final_code = final_code.replace(loop_line, logic_code + "\n" + indent + loop_line)
             else:
                 print("FEHLER: Loop-Start nicht gefunden. Injektion fehlgeschlagen!")

        # ------------------------- 4. Model Override Injektion (für Parallelität) -------------------------
        if params_dict.get("TARGET_MODEL_FOLDER"):
            # ------------------------- Spezielles Handling für Parallelität: Erzwinge die Nutzung eines bestimmten Modell-Ordners -------------------------
            target_folder = params_dict["TARGET_MODEL_FOLDER"]
            
            # ------------------------- Pattern zum Ersetzen der originalen Ordner-Auswahl (die normalerweise 'max' nach Zeitstempel nimmt) -------------------------
            match_pattern = "max(timestamp_folders, key=lambda f: f.stat().st_mtime)"
            
            if match_pattern in final_code:
                 # ------------------------- Code zum Filtern der Ordnerliste auf das Zielmodell injizieren -------------------------
                 filter_code = f"\n# --- INJECTED FOLDER FILTER ---\n"
                 filter_code += f"target_name = '{target_folder}'\n"
                 filter_code += f"timestamp_folders = [f for f in timestamp_folders if f.name == target_name]\n"
                 filter_code += f"if not timestamp_folders: print(f'WARNUNG: Target Folder {{target_name}} nicht gefunden in {{[f.name for f in timestamp_folders]}}')\n"
                 filter_code += f"# ------------------------------\n"
                 
                 # ------------------------- Originalen Aufruf markieren/behalten (wird auf gefilterte Liste angewendet) -------------------------
                 final_code = final_code.replace(match_pattern, f"max(timestamp_folders, key=lambda f: f.stat().st_mtime) # Original")
                 
                 # ------------------------- Code-Injektion an der richtigen Stelle (vor der Verwendung von timestamp_folders) -------------------------
                 lines = final_code.splitlines()
                 new_lines = []
                 for line in lines:
                     new_lines.append(line)
                     if "timestamp_folders = [" in line and ".iterdir()" in line:
                         # ------------------------- Indentierung übernehmen -------------------------
                         current_indent = line[:len(line) - len(line.lstrip())]
                         
                         indented_filter_code = ""
                         for f_line in filter_code.splitlines():
                             if f_line.strip():
                                 indented_filter_code += current_indent + f_line + "\n"
                             else:
                                 indented_filter_code += "\n"
                                 
                         new_lines.append(indented_filter_code)
                         
                 final_code = "\n".join(new_lines)
            else:
                 print("WARNUNG: Konnte timestamp_folders Logik nicht finden für Model Override.")

        # ------------------------- 3. Seeding Injektion (für Reproduzierbarkeit in Inference) -------------------------
        if params_dict.get("FIXED_SEED"):
            print("  -> Injiziere Random Seed...")
            seed_code = f"\nimport random\nimport numpy as np\nrandom.seed({params_dict['FIXED_SEED']})\nnp.random.seed({params_dict['FIXED_SEED']})\nprint('DEBUG: Festen Seed auf {params_dict['FIXED_SEED']} gesetzt')\n"
            # ------------------------- Am besten ganz am Anfang von Main oder nach den Imports injizieren -------------------------
            if 'if __name__ == "__main__":' in final_code:
                 final_code = final_code.replace('if __name__ == "__main__":', f'{seed_code}\nif __name__ == "__main__":')
            else:
                 final_code = seed_code + final_code
        
    # ------------------------- Temporäres Python-Script erstellen -------------------------
    start_ts = time.time()
    job_name = f"{notebook_path.stem}_Job_Temp.py"
    temp_script_path = notebook_path.parent / job_name
    
    with open(temp_script_path, "w", encoding="utf-8") as f:
        f.write(final_code)
    
    # ------------------------- Ausführen -------------------------
    print(f"  -> Starte Prozess ({job_name})...")
    
    try:
        # ------------------------- Sequentielle Ausführung mit run -------------------------
        result = subprocess.run(
            ["python", "-u", str(temp_script_path)],
            cwd=notebook_path.parent,
            capture_output=False, # Direkt auf stdout ausgeben
            check=False # Noch nicht prüfen
        )
        
        if result.returncode != 0:
            print(f"  -> FEHLER: Prozess beendet mit Code {result.returncode}")
            return False, None
            
        print("  -> Prozess erfolgreich.")
        return True, start_ts
        
    except Exception as e:
        print(f"  -> Exception: {e}")
        return False, None
    finally:
        if temp_script_path.exists():
            try:
                os.remove(temp_script_path)
            except: pass

def get_latest_folder(parent_dir, start_time_ts):
    if not parent_dir.exists(): return None
    candidates = [d for d in parent_dir.iterdir() if d.is_dir()]
    if not candidates: return None
    
    # ------------------------- Neuesten holen -------------------------
    latest = max(candidates, key=lambda d: d.stat().st_mtime)
    if latest.stat().st_mtime >= start_time_ts:
       # ------------------------- Wir nehmen das erste -------------------------
    return None

def update_incremental_report(eval_folder, run_label, latent, hidden):
    global results_history
    
    # ------------------------- Metadaten suchen -------------------------
    timestamp = eval_folder.name
    models_root = NOTEBOOK_4_1.parent / "Models"
    model_dir = models_root / timestamp
    
    if not model_dir.exists():
        print(f"Warnung: Model Ordner {model_dir} nicht gefunden.")
        return

    print(f"  -> Lese Metadaten aus {model_dir.name}...")
    
    meta_files = list(model_dir.glob("*_meta.json"))
    if not meta_files:
        print("  -> Keine Metadata gefunden.")
        return
        
    meta_path = meta_files[0]
    
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        # ------------------------- Grundlegende Daten aus Meta -------------------------
        entry = {
            "Run_Label": run_label,
            "Timestamp": timestamp,
            "Latent_Dim": meta.get("latent_dim", latent),
            "Hidden_Dim": meta.get("hidden_dim", hidden),
            "CV_Mean_Loss (Train)": meta.get("cv_mean_score", None),
            "Masked_CV_Loss (Train)": meta.get("cv_masked_score", None),
        }
        
        # ------------------------- Rohdaten für Split-Metriken (Training vs Test) -------------------------
        # ------------------------- Suche nach Imputation_Results CSVs im Inference Ordner -------------------------
        inf_results_root = NOTEBOOK_4_2.parent / "Inference_Results" / model_dir.name
        
        train_rmse_list = []
        test_rmse_list = []
        train_r2_list = []
        test_r2_list = []
        
        if inf_results_root.exists():
             csv_files = list(inf_results_root.glob("Imputation_Results_*.csv"))
             print(f"  -> Berechne Split-Metriken aus {len(csv_files)} Dateien...")
             
             for csv_file in csv_files:
                 if csv_file.stat().st_size == 0: continue
                 
                 try:
                     df = pd.read_csv(csv_file)
                     if "Split" not in df.columns: continue
                     
                     # ------------------------- Split Analysis -------------------------
                     for split_name in ["Train", "Test"]:
                         subset = df[df["Split"] == split_name]
                         if subset.empty: continue
                         
                         y_true = subset["Original"]
                         y_pred = subset["Imputed"]
                         
                         if len(y_true) > 0:
                             mse = mean_squared_error(y_true, y_pred)
                             rmse = math.sqrt(mse)
                             r2 = r2_score(y_true, y_pred)
                             
                             if split_name == "Train":
                                 train_rmse_list.append(rmse)
                                 train_r2_list.append(r2)
                             else:
                                 test_rmse_list.append(rmse)
                                 test_r2_list.append(r2)
                 except Exception as e:
                     print(f"    Fehler bei {csv_file.name}: {e}")

        # ------------------------- Durchschnittswerte berechnen -------------------------
        entry["Avg_RMSE_Train"] = np.mean(train_rmse_list) if train_rmse_list else None
        entry["Avg_RMSE_Test"] = np.mean(test_rmse_list) if test_rmse_list else None
        entry["Avg_R2_Train"] = np.mean(train_r2_list) if train_r2_list else None
        entry["Avg_R2_Test"] = np.mean(test_r2_list) if test_r2_list else None
        
        print(f"  -> Result: RMSE Train={entry['Avg_RMSE_Train']:.4f} / Test={entry['Avg_RMSE_Test']:.4f}")

        with history_lock:
             results_history.append(entry)
             # ------------------------- PDF Erstellen -------------------------
             generate_progress_pdf(eval_folder)
        
    except Exception as e:
        print(f"Fehler beim Erstellen des Reports: {e}")
        import traceback
        traceback.print_exc()

def generate_progress_pdf(output_folder):
    if not results_history: return
    
    df = pd.DataFrame(results_history)
    
    # ------------------------- Index hinzufügen -------------------------
    df["Index"] = range(1, len(df) + 1)
    
    pdf_path = output_folder / "Incremental_Progress.pdf"
    
    print(f"  -> Generiere PDF: {pdf_path}")
    
    # ------------------------- Plotting Setup -------------------------
    sns.set_theme(style="whitegrid")
    
    # ------------------------- Multipage PDF speichern -------------------------
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(pdf_path) as pdf:
        
        # ------------------------- Seite 1: Tabellarische Auflistung -------------------------
        fig1, ax1 = plt.subplots(figsize=(11.69, 8.27))
        ax1.axis('off')
        ax1.set_title("Incremental Run History", fontsize=18, weight='bold')
        
        # ------------------------- Spalten für Tabelle auswählen -------------------------
        display_cols = ["Index", "Latent_Dim", "Hidden_Dim",  
                        "Avg_RMSE_Train", "Avg_RMSE_Test", 
                        "Avg_R2_Train", "Avg_R2_Test"]
        
        # ------------------------- Formatieren für Anzeige -------------------------
        df_display = df.copy()
        for col in display_cols[3:]: # Ab RMSE alle formatieren
             if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
        
        # ------------------------- Tabelle zeichnen -------------------------
        table_data = [display_cols] + df_display[display_cols].values.tolist()
        table = ax1.table(cellText=table_data, colLabels=display_cols, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        pdf.savefig(fig1)
        plt.close(fig1)
        
        # ------------------------- Seite 2: Plots (RMSE & R2) -------------------------
        fig2, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        fig2.suptitle("Performance Metrics Progression (Train vs Test)", fontsize=16)
        
        # ------------------------- Plot Helper Funktion -------------------------
        def plot_metric(ax, metric_train, metric_test, title, higher_better=False):
            if metric_train in df.columns:
                sns.lineplot(data=df, x="Index", y=metric_train, marker="o", ax=ax, color="blue", label="Train")
            if metric_test in df.columns:
                sns.lineplot(data=df, x="Index", y=metric_test, marker="o", ax=ax, color="red", label="Test")
            
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
        
        # ------------------------- 1. RMSE Plot (Oben Links) -------------------------
        plot_metric(axes[0, 0], "Avg_RMSE_Train", "Avg_RMSE_Test", "Average RMSE (Lower is better)")
        
        # ------------------------- 2. R2 Plot (Oben Rechts) -------------------------
        plot_metric(axes[0, 1], "Avg_R2_Train", "Avg_R2_Test", "Average R² (Higher is better)", higher_better=True)
        
        # ------------------------- 3. CV Loss (Unten Links) -------------------------
        ax_cv = axes[1, 0]
        if "CV_Mean_Loss (Train)" in df.columns:
            sns.lineplot(data=df, x="Index", y="CV_Mean_Loss (Train)", marker="x", linestyle="--", ax=ax_cv, label="CV Mean (Train)", color="green")
        if "Masked_CV_Loss (Train)" in df.columns:
            sns.lineplot(data=df, x="Index", y="Masked_CV_Loss (Train)", marker="x", linestyle="--", ax=ax_cv, label="Masked CV (Train)", color="orange")
            
        ax_cv.set_title("Cross-Validation Loss (Training Data Only)")
        ax_cv.legend()
        ax_cv.grid(True)
        
        # ------------------------- 4. Leer (Unten Rechts) -------------------------
        axes[1, 1].axis("off")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig2)
        plt.close(fig2)
        
        # ------------------------- Seite 3: Erklärung -------------------------
        fig3, ax3 = plt.subplots(figsize=(11.69, 8.27))
        ax3.axis('off')
        
        explanation_text = (
            "Metric Explanation\n\n"
            "CV Mean Loss (Train):\n"
            "Measures how well the model learns the general data structure during training.\n"
            "It includes both reconstruction error and latent space regularity (KLD).\n"
            "Lower is better.\n\n"
            "Masked CV Loss (Train):\n"
            "A stress test during training where the model must guess 20% hidden values.\n"
            "Measures pure imputation capability on known data.\n"
            "Lower is better.\n\n"
            "RMSE (Test):\n"
            "The critical metric for generalization. It measures how accurately the model fills gaps\n"
            "in data it has never seen before (Test Set).\n"
            "If this increases while Training RMSE decreases, the model is overfitting.\n"
            "Lower is better.\n\n"
            "R² Score (Test):\n"
            "Indicates how much of the variance in the unseen test data is explained\n"
            "by the model's predictions.\n"
            "Higher is better (closer to 1.0)."
        )
        
        ax3.text(0.1, 0.8, explanation_text, fontsize=14, va='top', family='monospace', linespacing=1.8)
        ax3.set_title("Metrics Definition", fontsize=18, weight='bold')
        
        pdf.savefig(fig3)
        plt.close(fig3)

def main():
    print("==================================================")
    print("       START PIPELINE 4 (Dimension Iterator)      ")
    print("==================================================")
    
    # ------------------------- Daten Vorschau -------------------------
    print("\nLade Datenvorschau für Entscheidungshilfe...")
    preview_data = get_run_preview_info()
    
    if isinstance(preview_data, list):
        print("\nVerfügbare Runs und deren Datenmenge:")
        print(f"{'Index':<6} | {'Name':<20} | {'Samples':<10} | {'Features':<10}")
        print("-" * 55)
        for entry in preview_data:
            print(f"{entry['idx']:<6} | {entry['name']:<20} | {entry['count']:<10} | {entry['features']:<10}")
        print("-" * 55)
    else:
        print(f"Warnung: Konnte Vorschau nicht laden ({preview_data})")
        
    
    # ------------------------- Benutzer Auswahl -------------------------
    print("\nWelcher Run soll verwendet werden?")
    
    valid_choice = False
    target_idx = 0
    target_label_name = "Base_without_pH" # Default fallback
    
    while not valid_choice:
        try:
            choice_str = input(f"Bitte Index eingeben (0-{len(preview_data)-1 if isinstance(preview_data, list) else 5}) [Default 0]: ").strip()
            if choice_str == "":
                target_idx = 0
                if isinstance(preview_data, list): target_label_name = preview_data[0]['name']
                valid_choice = True
            else:
                target_idx = int(choice_str)
                if isinstance(preview_data, list):
                    if 0 <= target_idx < len(preview_data):
                        target_label_name = preview_data[target_idx]['name']
                        valid_choice = True
                    else:
                        print(f"Bitte Index zwischen 0 und {len(preview_data)-1} wählen.")
                else:
                    valid_choice = True # Blindes Vertrauen bei Fehler in Vorschau
        except ValueError:
            print("Ungültige Eingabe.")
            
    print(f"--> Verwende Run Index: {target_idx} ({target_label_name})")
    
    total_steps = len(LATENT_DIMS) * len(HIDDEN_DIMS)
    current_step = 0
    
    for l_dim in LATENT_DIMS:
        for h_dim in HIDDEN_DIMS:
            current_step += 1
            run_label = f"L{l_dim}_H{h_dim}"
            
            print(f"\n\n###############################################################")
            print(f"SCHRITT {current_step}/{total_steps}: Latent={l_dim}, Hidden={h_dim} (Run: {target_label_name})")
            print(f"###############################################################")
            
            # ------------------------- 1. Training (4.1) -------------------------
            params_4_1 = {
                "MAX_ITERATIONS": 1, 
                "TARGET_RUN_INDEX": target_idx,
                "LATENT_DIM": l_dim,
                "HIDDEN_DIM": h_dim,
                "BATCH_SIZE": FIXED_BATCH_SIZE,
                "EPOCHS": FIXED_EPOCHS,
                "KLD_WEIGHT": FIXED_KLD_WEIGHT,
                "VAE_MODE": "MANUAL" 
            }
            
            os.environ["VAE_MODE"] = "MANUAL"
            
            success, start_ts = run_notebook_with_params(NOTEBOOK_4_1, params_4_1, "4.1 VAE Training")
            if not success:
               print("Abbruch wegen Fehler in 4.1")
               continue
               
            # ------------------------- Output Ordner finden -------------------------
            models_root = NOTEBOOK_4_1.parent / "Models"
            model_dir = get_latest_folder(models_root, start_ts)
            
            if not model_dir:
                print("Fehler: Kein neuer Modell-Ordner gefunden!")
                continue
                
            print(f"  -> Modell gespeichert in: {model_dir.name}")
            
            print(f"  -> Modell gespeichert in: {model_dir.name}")
            
            # ------------------------- Hintergrund-Thread für Post-Processing -------------------------
            def run_post_processing(current_model_dir, current_run_label, current_l, current_h):
                try:
                    print(f"  [Background] Starte Post-Processing für {current_run_label}...")
                    
                    # ------------------------- 2. Inference (4.2) -------------------------
                    params_4_2 = {
                        "FIXED_SEED": 42
                    }
                    
                    # ------------------------- Setze Ziel-Modell-Ordner explizit für parallele Ausführung -------------------------
                    params_4_2["TARGET_MODEL_FOLDER"] = current_model_dir.name
                    
                    success_inf, start_ts_inf = run_notebook_with_params(NOTEBOOK_4_2, params_4_2, f"4.2 Inference ({current_run_label})")
                    
                    if not success_inf:
                        print(f"  [Background] Abbruch Inference für {current_run_label}")
                        return

                    # ------------------------- Inferenz Ergebnisse finden -------------------------
                    inf_results_root = NOTEBOOK_4_2.parent / "Inference_Results"
                    inf_dir = inf_results_root / current_model_dir.name
                    
                    if not inf_dir.exists():
                         print(f"  [Background] Fehler: Inferenz-Ordner {inf_dir} nicht gefunden.")
                         return

                    # ------------------------- 3. Evaluation (4.3) -------------------------
                    params_4_3 = {
                        "TARGET_MODEL_FOLDER": current_model_dir.name 
                    }
                    success_eval, start_ts_eval = run_notebook_with_params(NOTEBOOK_4_3, params_4_3, f"4.3 Evaluation ({current_run_label})")
                    
                    if not success_eval:
                        print(f"  [Background] Abbruch Evaluation für {current_run_label}")
                        return
                    
                    # ------------------------- Evaluation Ergebnisse finden -------------------------
                    eval_results_root = NOTEBOOK_4_3.parent / "Evaluation_Results"
                    eval_dir = eval_results_root / current_model_dir.name
                    
                    if not eval_dir.exists():
                        print(f"  [Background] Fehler: Evaluation-Ordner {eval_dir} nicht gefunden.")
                        return
                        
                    # ------------------------- 4. Inkrementelles Reporting -------------------------
                    update_incremental_report(eval_dir, current_run_label, current_l, current_h)
                    print(f"  [Background] Fertig für {current_run_label}")

                except Exception as e:
                    print(f"  [Background] Exception in Post-Processing für {current_run_label}: {e}")

            # ------------------------- Thread starten -------------------------
            t = threading.Thread(target=run_post_processing, args=(model_dir, run_label, l_dim, h_dim))
            t.start()
            
            print(f"  -> Post-Processing gestartet im Hintergrund. Mache weiter mit nächstem Step...")
            
            # ------------------------- Kurz warten, damit der Thread sicher anläuft -------------------------
            time.sleep(5)

if __name__ == "__main__":
    main()
