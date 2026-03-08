import os
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.backends.backend_pdf import PdfPages

# ------------------------- Konfiguration -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_VAE = BASE_DIR / "4_Imputation" / "4.2_Inference" / "Inference.ipynb"
NOTEBOOK_KNN = BASE_DIR / "5_kNN" / "5.1_kNearest-Neighbors.ipynb"

# ------------------------- Daten-Vorschau Helfer -------------------------
def get_latest_preprocessing_file():
    # ------------------------- Sucht die neueste Preprocessing-Datei -------------------------
    prep_root = BASE_DIR / "3_Machine-Learning" / "3.1_Preprocessing" / "Preprocessing"
    if not prep_root.exists(): return None
    timestamp_folders = [f for f in prep_root.iterdir() if f.is_dir()]
    if not timestamp_folders: return None
    latest_folder = max(timestamp_folders, key=lambda f: f.stat().st_mtime)
    return latest_folder / "Preprocessed_SOM_Ready.csv"

def get_run_preview_info():
    # ------------------------- Lädt Informationen für die Run-Vorschau -------------------------
    csv_path = get_latest_preprocessing_file()
    if not csv_path: return []
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except: return []

# ------------------------- Globaler Run-Index (Synchronisiert) -------------------------
RUNS = [
    {"name": "Base_without_pH", "add": []},
    {"name": "Base_with_pH",    "add": ["pH"]},
    {"name": "Plus_pH-Fe",      "add": ["pH", "Fe_in_mmol/L"]},
    {"name": "Plus_pH-K-Fe",    "add": ["pH", "K_in_mmol/L", "Fe_in_mmol/L"]},
    {"name": "Plus_pH-K-Fe-Mn", "add": ["pH", "K_in_mmol/L", "Fe_in_mmol/L", "Mn_in_mmol/L"]},
    {"name": "Plus_temperature", "add": ["temperature_in_c"]}
]

def get_run_preview_info():
    # ------------------------- Lädt Informationen für die Run-Vorschau -------------------------
    csv_path = get_latest_preprocessing_file()
    if not csv_path: return []
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except: return []

    FIXED_BASE_FEATURES = ["Na_in_mmol/L", "Mg_in_mmol/L", "Ca_in_mmol/L", "Cl_in_mmol/L", "SO4_in_mmol/L", "HCO3_in_mmol/L"]
    
    # ------------------------- Hilfsfunktion für Features in der Vorschau -------------------------
    def get_feats(user_list, cols):
        res = []
        for u in user_list:
            cand = [c for c in cols if c.startswith(u) and "_gauss" in c]
            if cand: res.append(next((c for c in cand if "_log_gauss" in c), cand[0]))
            elif u in cols: res.append(u)
        return res

    results = []
    for i, r in enumerate(RUNS):
        tf = get_feats(FIXED_BASE_FEATURES + r["add"], df.columns)
        results.append({"idx": i, "name": r["name"], "count": df[tf].dropna().shape[0], "total": len(df), "features": len(tf)})
    return results

def run_notebook(path, params):
    # ------------------------- Führt ein Notebook mit injizierten Parametern aus -------------------------
    print(f"  -> Notebook: {path.name} mit Params: {params}")
    convert_cmd = ["jupyter", "nbconvert", "--to", "python", "--stdout", str(path)]
    code = subprocess.check_output(convert_cmd, cwd=path.parent).decode('utf-8', errors='ignore')
    
    # ------------------------- Parameter-Injektion -------------------------
    injection = "\n# --- INJECTED ---\nimport itertools\nimport random\n"
    for k, v in params.items():
        if isinstance(v, str): injection += f"{k} = '{v}'\n"
        else: injection += f"{k} = {v}\n"
    
    job_script = path.parent / f"TEMP_Joint_{path.stem}.py"
    with open(job_script, "w", encoding="utf-8") as f:
        f.write(injection + "\nimport sys\n" + code.replace("get_ipython()", "# get_ipython()"))
    
    try:
        res = subprocess.run(["python", "-u", str(job_script)], cwd=path.parent)
        return res.returncode == 0
    finally:
        # ------------------------- Cleanup -------------------------
        if job_script.exists(): os.remove(job_script)

def generate_comparison_pdf(idx, include_incomplete, v_results_path, k_results_path, vae_idx=None):
    # ------------------------- Erstellt einen Vergleichs-Bericht als PDF -------------------------
    if vae_idx is None: vae_idx = idx
    output_pdf = BASE_DIR / "4-5_Comparison" / f"Comparison_VAE_vs_kNN_Run_{idx:03d}_{'Incomplete' if include_incomplete else 'Complete'}.pdf"
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    def load_data(root, pattern, model_tag):
        # ------------------------- Lädt die Ergebnisse für den Vergleich -------------------------
        files = list(root.glob(pattern))
        data = []
        for f in files:
            try:
                lvl = int(str(f.stem).split("_Level_")[1])
                df = pd.read_csv(f)
                grouped = df.groupby("Masked_Combination")
                for name, gp in grouped:
                    y_t, y_p = gp["Original"], gp["Imputed"]
                    if len(y_t)<2: continue
                    data.append({"Level": lvl, "Model": model_tag, "RMSE": np.sqrt(mean_squared_error(y_t, y_p)), "R2": r2_score(y_t, y_p), "Samples": len(gp)})
            except: pass
        return pd.DataFrame(data)

    v_pattern = f"Imputation_Results_*_{vae_idx:03d}*_Level_*.csv"
    df_v = load_data(v_results_path, v_pattern, "VAE")
    df_k = load_data(k_results_path, f"Imputation_Results_kNN_Level_*.csv", "kNN")
    
    if df_v.empty or df_k.empty:
        print("Fehler: Konnte nicht genügend Daten für den Vergleich laden."); return

    df_all = pd.concat([df_v, df_k]).sort_values("Level")
    sns.set_theme(style="whitegrid")
    palette = {"VAE": "#e74c3c", "kNN": "#3498db"} # ------------------------- Rot vs Blau

    with PdfPages(output_pdf) as pdf:
        # ------------------------- Erstellt Boxplots für RMSE und R2 -------------------------
        for metric in ["RMSE", "R2"]:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            import math
            # ------------------------- Achsen-Beschriftungen vorbereiten -------------------------
            unique_lvls = sorted(df_all["Level"].unique())
            max_feat_count = max(unique_lvls) + 1
            labels = [f"{lvl}\n(n={math.comb(max_feat_count, lvl)})" for lvl in unique_lvls]
            
            sns.boxplot(data=df_all, x="Level", y=metric, hue="Model", ax=ax, palette=palette)
            ax.set_xticklabels(labels)
            title = f"{metric} Vergleich: VAE vs kNN\n(Run {idx:03d}, Modus: {'Incomplete' if include_incomplete else 'Complete'})"
            ax.set_title(title, fontsize=16)
            if metric == "R2": ax.set_ylim(-0.25, 1.1)
            pdf.savefig(fig); plt.close(fig)
            
        # ------------------------- Seite mit tabellarischer Zusammenfassung -------------------------
        fig, ax = plt.subplots(figsize=(11.69, 8.27)); ax.axis('off')
        summary = df_all.groupby(["Level", "Model"])[["RMSE", "R2"]].mean().unstack().round(4)
        ax.table(cellText=([summary.columns.tolist()]+summary.values.tolist()), loc='center', cellLoc='center').scale(1, 1.5)
        pdf.savefig(fig); plt.close(fig)
    print(f"Report erstellt: {output_pdf.name}")

def main():
    print("==================================================")
    print("    JOINT PIPELINE: VAE vs kNN COMPARISON         ")
    print("==================================================")
    
    # ------------------------- Auswahl-Logik (CLI / Env / Interaktiv) -------------------------
    idx = None
    inc = None
    
    # ------------------------- CLI Prüfung -------------------------
    if len(sys.argv) > 1:
        try: idx = int(sys.argv[1]); print(f"-> Index {idx} aus CLI-Argument")
        except: pass
    if len(sys.argv) > 2:
        inc = sys.argv[2].lower() in ['y', 'yes', 'inc', '1', 'true']
        print(f"-> Incomplete {inc} aus CLI-Argument")
    
    # ------------------------- Env Prüfung -------------------------
    if idx is None and os.environ.get("PIPE_INDEX"):
        idx = int(os.environ.get("PIPE_INDEX"))
    if inc is None and os.environ.get("PIPE_INC"):
        inc = os.environ.get("PIPE_INC").lower() == 'true'

    if idx is None:
        previews = get_run_preview_info()
        for p in previews: print(f"[{p['idx']}] {p['name']:<20} | Complete: {p['count']:<6} | Total: {p['total']}")
        idx = int(input("\nIndex wählen (0-5) [0]: ") or 0)
    
    if inc is None:
        inc = input("Inklusive unvollständige Samples? (y/n) [n]: ").lower() == 'y'
    
    print(f"\n---> Starte Vergleich für Index {idx} ({'Incomplete' if inc else 'Complete'})\n")
    
    # ------------------------- kNN Ausführung -------------------------
    success_k = run_notebook(NOTEBOOK_KNN, {"TARGET_RUN_INDEX": idx, "INCLUDE_INCOMPLETE": inc})
    
    # ------------------------- VAE Ausführung -------------------------
    run_name = RUNS[idx]["name"]
    models_root = BASE_DIR / "4_Imputation" / "4.1_VAE_Imputation" / "Models"
    
    potential_models = list(models_root.glob(f"**/Model_*_{run_name}_vae.pth"))
            
    if not potential_models:
        print(f"FAILED: Could not find any VAE model with name '{run_name}'.")
        return

    latest_model_file = max(potential_models, key=lambda f: f.stat().st_mtime)
    latest_v_folder = latest_model_file.parent
    vae_idx = int(latest_model_file.name.split("_")[1])
    
    print(f"  -> Using VAE model: Index {vae_idx} ({run_name}) from folder {latest_v_folder.name}")
    
    success_v = run_notebook(NOTEBOOK_VAE, {"TARGET_RUN_INDEX": vae_idx, "INCLUDE_INCOMPLETE": inc, "FORCE_MODEL_FOLDER": latest_v_folder.name})
    
    # ------------------------- Bericht-Generierung -------------------------
    if success_k and success_v:
        generate_comparison_pdf(idx, inc, NOTEBOOK_VAE.parent / "Inference_Results" / latest_v_folder.name, NOTEBOOK_KNN.parent / "Inference_Results" / f"kNN_Run_{idx:03d}", vae_idx=vae_idx)
    
    input("\nPipline beendet. Drücke Enter zum Schließen...")

if __name__ == "__main__": main()
