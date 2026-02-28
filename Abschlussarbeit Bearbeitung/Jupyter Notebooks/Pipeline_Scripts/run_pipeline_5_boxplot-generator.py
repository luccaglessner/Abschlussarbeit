import os
import subprocess
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.backends.backend_pdf import PdfPages

# ------------------------- Konfiguration -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_KNN = BASE_DIR / "5_kNN" / "5.1_kNearest-Neighbors.ipynb"

# ------------------------- Daten Vorschau Helfer -------------------------
def get_latest_preprocessing_file():
    # ------------------------- Sucht die neueste Preprocessing-Datei -------------------------
    script_path = Path(__file__).parent
    root = script_path
    found_prep = None
    for _ in range(4):
        if (root / "3_Machine-Learning").exists():
            found_prep = root / "3_Machine-Learning" / "3.1_Preprocessing" / "Preprocessing"
            break
        root = root.parent
        
    if not found_prep or not found_prep.exists():
        root = Path("f:/Abschlussarbeit/Abschlussarbeit Bearbeitung")
        found_prep = root / "3_Machine-Learning" / "3.1_Preprocessing" / "Preprocessing"
        
    if not found_prep.exists():
        return None

    timestamp_folders = [f for f in found_prep.iterdir() if f.is_dir()]
    if not timestamp_folders:
        return None
        
    latest_folder = max(timestamp_folders, key=lambda f: f.stat().st_mtime)
    return latest_folder / "Preprocessed_SOM_Ready.csv"

def get_training_features(user_selection, df_columns):
    # ------------------------- Findet die passenden Trainings-Features im Datensatz -------------------------
    training_features = []
    for user_col in user_selection:
        found = False
        candidates = [c for c in df_columns if c.startswith(user_col) and "_gauss" in c]
        if candidates:
            best_match = next((c for c in candidates if "_log_gauss" in c), candidates[0])
            training_features.append(best_match)
            found = True
        else:
            if user_col in df_columns:
                training_features.append(user_col)
                found = True
        
        if not found:
             fuzzy = [c for c in df_columns if c.startswith(user_col) and "_gauss" in c]
             if fuzzy:
                 best_match = fuzzy[0]
                 training_features.append(best_match)
                 found = True
    return training_features

def get_run_preview_info():
    # ------------------------- Lädt Informationen für die Run-Vorschau -------------------------
    csv_path = get_latest_preprocessing_file()
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

def run_notebook(name, path, params_dict=None):
    # ------------------------- Führt ein Notebook mittels Konvertierung zu Python aus -------------------------
    print(f"\n--- Starte: {name} ---")
    if not path.exists():
        print(f"FEHLER: Datei nicht gefunden: {path}")
        return False

    print("  -> Lade Code aus Notebook...")
    convert_cmd = ["jupyter", "nbconvert", "--to", "python", "--stdout", str(path)]
    try:
        python_code = subprocess.check_output(convert_cmd, cwd=path.parent).decode('utf-8', errors='ignore')
    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Konvertieren: {e}")
        return False

    # ------------------------- IPython Magics entfernen -------------------------
    filtered_code_lines = []
    for line in python_code.splitlines():
        if "get_ipython()" in line:
            filtered_code_lines.append(f"# {line}")
        else:
            filtered_code_lines.append(line)
    
    final_code = "\n".join(filtered_code_lines)

    # ------------------------- Parameter-Injektion -------------------------
    if params_dict:
        injection = "\n# --- INJECTED PARAMETERS ---\n"
        for k, v in params_dict.items():
             if isinstance(v, str):
                 injection += f"{k} = '{v}'\n"
             else:
                 injection += f"{k} = {v}\n"
        final_code = injection + final_code

    job_name = f"{path.stem}_Job_Temp.py"
    temp_script_path = path.parent / job_name
    
    try:
        with open(temp_script_path, "w", encoding="utf-8") as f:
            f.write(final_code)
        
        result = subprocess.run(["python", "-u", str(temp_script_path)], cwd=path.parent)
        return result.returncode == 0
    finally:
        # Cleanup
        if temp_script_path.exists():
            os.remove(temp_script_path)

def generate_boxplot_pdf(model_name, results_root):
    # ------------------------- Generiert einen PDF-Bericht mit Boxplots -------------------------
    output_pdf = results_root.parent / f"Boxplot_Summary_Level_Analysis_{model_name}.pdf"
    csv_files = list(results_root.glob("Imputation_Results_kNN_Level_*.csv"))
    if not csv_files:
        print("Keine CSVs gefunden!")
        return

    all_data = []
    for csv_file in csv_files:
        try:
            # ------------------------- Level aus Dateiname extrahieren -------------------------
            level = int(csv_file.stem.split("_Level_")[1])
            df = pd.read_csv(csv_file)
            if df.empty: continue
            
            grouped = df.groupby("Masked_Combination")
            for name, group in grouped:
                y_true, y_pred = group['Original'], group['Imputed']
                if len(y_true) < 2: continue
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                all_data.append({"Level": level, "Combination": name, "RMSE": rmse, "R2": r2, "Samples": len(group)})
        except Exception as e:
             print(f"Fehler bei {csv_file.name}: {e}")

    if not all_data:
        print("Keine Daten für PDF gefunden.")
        return

    df_plot = pd.DataFrame(all_data).sort_values("Level")
    sns.set_theme(style="whitegrid")
    
    # ------------------------- Beschriftungen mit Zählern (k und n) -------------------------
    level_stats = df_plot.groupby("Level").agg({"Combination": "count", "Samples": "max"})
    unique_levels = sorted(df_plot["Level"].unique())
    labels = []
    for lvl in unique_levels:
        k = level_stats.loc[lvl, "Combination"]
        n = int(level_stats.loc[lvl, "Samples"])
        labels.append(f"{lvl}\n(k={k})\n(n={n})")

    with PdfPages(output_pdf) as pdf:
        # ------------------------- Seite 1: RMSE Boxplot -------------------------
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        sns.boxplot(data=df_plot, x="Level", y="RMSE", ax=ax, color="skyblue")
        ax.set_title(f"kNN RMSE Verteilung (n_neighbors=5)\nModell: {model_name} (VAE Testset angepasst)", fontsize=16)
        ax.set_xticklabels(labels)
        pdf.savefig(fig); plt.close(fig)
        
        # ------------------------- Seite 2: R2 Boxplot -------------------------
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        sns.boxplot(data=df_plot, x="Level", y="R2", ax=ax, color="lightgreen")
        ax.set_ylim(-1.0, 1.1)
        ax.set_title(f"kNN R² Verteilung\nModell: {model_name} (VAE Testset angepasst)", fontsize=16)
        ax.set_xticklabels(labels)
        pdf.savefig(fig); plt.close(fig)
        
        # ------------------------- Seite 3: Tabellarische Zusammenfassung -------------------------
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        summary = df_plot.groupby("Level")[["RMSE", "R2"]].mean().reset_index().round(4)
        summary["k (Combos)"] = summary["Level"].map(level_stats["Combination"])
        summary["n (Samples)"] = summary["Level"].map(level_stats["Samples"])
        table_title = f"kNN Statistik-Zusammenfassung (VAE Testset angepasst)"
        ax.text(0.5, 0.95, table_title, horizontalalignment='center', fontsize=14, transform=ax.transAxes)
        table = ax.table(cellText=([summary.columns.tolist()] + summary.values.tolist()), loc='center', cellLoc='center')
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.5)
        pdf.savefig(fig); plt.close(fig)

    print(f"PDF erstellt: {output_pdf.name}")

def main():
    print("==================================================")
    print("      START kNN PIPELINE (BOXPLOT GENERATOR)      ")
    print("==================================================")
    
    # ------------------------- Auswahlmenü anzeigen -------------------------
    preview_data = get_run_preview_info()
    if isinstance(preview_data, list):
        print("\nVerfügbare Runs:")
        print(f"{'Index':<6} | {'Name':<20} | {'Samples':<10} | {'Features':<10}")
        print("-" * 55)
        for entry in preview_data:
            print(f"{entry['idx']:<6} | {entry['name']:<20} | {entry['count']:<10} | {entry['features']:<10}")
        print("-" * 55)
    else:
        print(f"Warnung: Vorschau fehlgeschlagen ({preview_data})")

    target_idx = 0
    valid_choice = False
    
    # ------------------------- CLI Argumente prüfen -------------------------
    if len(sys.argv) > 1:
        try:
            target_idx = int(sys.argv[1])
            print(f"--> Verwende Run Index aus CLI Argument: {target_idx}")
            valid_choice = True
        except ValueError:
            pass

    # ------------------------- Umgebungsvariable prüfen -------------------------
    if not valid_choice:
        env_idx = os.environ.get("KNN_TARGET_INDEX")
        if env_idx:
            try:
                target_idx = int(env_idx)
                print(f"--> Verwende Run Index aus Umgebungsvariable: {target_idx}")
                valid_choice = True
            except ValueError:
                print(f"Ungültiger Wert in KNN_TARGET_INDEX: {env_idx}")
            
    # ------------------------- Benutzereingabe abfragen -------------------------
    while not valid_choice:
        try:
            choice_str = input(f"Bitte Index eingeben (0-5) [Default 0]: ").strip()
            if choice_str == "":
                target_idx = 0; valid_choice = True
            else:
                target_idx = int(choice_str)
                if 0 <= target_idx <= 5: valid_choice = True
                else: print("Index ungültig.")
        except ValueError:
            print("Ungültige Eingabe.")
            
    print(f"--> Verwende Run Index: {target_idx}")
    
    # ------------------------- kNN Imputation ausführen -------------------------
    success = run_notebook("kNN Imputation", NOTEBOOK_KNN, {"TARGET_RUN_INDEX": target_idx})
    if success:
        results_dir = NOTEBOOK_KNN.parent / "Inference_Results" / f"kNN_Run_{target_idx:03d}"
        generate_boxplot_pdf(f"kNN_Run_{target_idx:03d}", results_dir)

if __name__ == "__main__":
    main()
