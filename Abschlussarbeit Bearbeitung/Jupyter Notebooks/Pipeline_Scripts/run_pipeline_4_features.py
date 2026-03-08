import os
import subprocess
import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
import textwrap

def replace_with_indent(code, target, injection):
    if target not in code: return code
    lines = code.splitlines()
    for i, line in enumerate(lines):
        if target in line:
            indent = line[:line.find(target)]
            dedented = textwrap.dedent(injection).strip()
            indented = textwrap.indent(dedented, indent)
            lines[i] = line + "\n" + indented
            break
    return "\n".join(lines)

# ------------------------- Konfiguration -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# ------------------------- Pfade zu den Notebooks -------------------------
NOTEBOOK_4_1 = BASE_DIR / "4_Imputation" / "4.1_VAE_Imputation" / "VAE_Imputation.ipynb"
NOTEBOOK_4_2 = BASE_DIR / "4_Imputation" / "4.2_Inference" / "Inference.ipynb"
NOTEBOOK_4_3 = BASE_DIR / "4_Imputation" / "4.3_Evaluation" / "Evaluation.ipynb"

# ------------------------- Daten Vorschau Helfer -------------------------
def get_latest_preprocessing_file():
    prep_root = BASE_DIR / "3_Machine-Learning" / "3.1_Preprocessing" / "Preprocessing"
    if not prep_root.exists(): return None
    timestamp_folders = [f for f in prep_root.iterdir() if f.is_dir()]
    if not timestamp_folders: return None
    latest_folder = max(timestamp_folders, key=lambda f: f.stat().st_mtime)
    return latest_folder / "Preprocessed_SOM_Ready.csv"

def get_training_features(user_selection, df_columns):
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

def run_notebook_as_job(name, path, params_dict=None):
    # ------------------------- Führt ein Notebook mittels Konvertierung zu Python aus -------------------------
    print(f"\n--- Starte: {name} ---")
    
    if not path.exists():
        print(f"FEHLER: Datei nicht gefunden: {path}")
        return False
    # ------------------------- Notebook zu Python-Code konvertieren -------------------------
    print(f"  -> Lade Code aus {path.name}...")
    try:
        convert_cmd = ["jupyter", "nbconvert", "--to", "python", "--stdout", str(path)]
        python_code = subprocess.check_output(convert_cmd, cwd=path.parent).decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Fehler beim Konvertieren von {path.name}: {e}")
        return False
    
    # ------------------------- DYNAMISCHE PATCHES FÜR CLIPPING (4.1) -------------------------
    if "VAE_Imputation" in str(path):
        python_code = python_code.replace("QUANTILE_CLIPPING = True", "QUANTILE_CLIPPING = (os.environ.get('VAE_CLIPPING') == '1')")
        python_code = python_code.replace("QUANTILE_CLIPPING = False", "QUANTILE_CLIPPING = (os.environ.get('VAE_CLIPPING') == '1')")
        
        # ------------------------- Meta-Tracking Initialisierung -------------------------
        python_code = replace_with_indent(python_code, "# ------------------------- Quantile Clipping -------------------------", 'clipping_meta = {"active": False}')
        
        # ------------------------- Erfassen der Grenzen (lower_q und upper_q) -------------------------
        bounds_capture = """
            if QUANTILE_CLIPPING:
                lower_q_scaled = scaler.transform(lower_q.reshape(1, -1))[0]
                upper_q_scaled = scaler.transform(upper_q.reshape(1, -1))[0]
                clipping_meta = {"active": True, "threshold": QUANTILE_THRESHOLD, "lower": lower_q_scaled.tolist(), "upper": upper_q_scaled.tolist()}
        """
        python_code = replace_with_indent(python_code, "X_train_scaled = scaler.fit_transform(X_train_raw)", bounds_capture)
        
        # ------------------------- In Metadaten-Dictionary injizieren -------------------------
        python_code = python_code.replace('"training_loss_history": epoch_loss_history,', '"training_loss_history": epoch_loss_history, "quantile_clipping": clipping_meta,')
        
    elif "Evaluation" in str(path):
        # ------------------------- Patch für Evaluation Clipping -------------------------
        clipping_bounds_logic = """
            # --- INJEKTION: GRENZWERTERFASSUNG ---
            is_clipped = False
            low_val = -1e9; high_val = 1e9 
            if "quantile_clipping" in meta and meta["quantile_clipping"].get("active"):
                is_clipped = True
                q_lower = meta["quantile_clipping"].get("lower")
                q_upper = meta["quantile_clipping"].get("upper")
                try:
                    f_idx = list(meta["features_mapped"]).index(feature)
                    low_val = q_lower[f_idx]
                    high_val = q_upper[f_idx]
                except: is_clipped = False
        """
        python_code = replace_with_indent(python_code, "feat_subset = subset[subset['Feature'] == feature]", clipping_bounds_logic)
        
        # ------------------------- Clipping für Metriken hinzufügen -------------------------
        python_code = python_code.replace(
            "y_true = feat_subset['Original']", 
            "y_true = np.clip(feat_subset['Original'], low_val, high_val) if is_clipped else feat_subset['Original']"
        ).replace(
            "y_pred = feat_subset['Imputed']", 
            "y_pred = np.clip(feat_subset['Imputed'], low_val, high_val) if is_clipped else feat_subset['Imputed']"
        )
        
        # ------------------------- Plot-Anpassungen -------------------------
        plot_clipping = """
            if is_clipped:
                subset_plot = subset_plot.copy()
                subset_plot['Original'] = np.clip(subset_plot['Original'], low_val, high_val)
                subset_plot['Imputed'] = np.clip(subset_plot['Imputed'], low_val, high_val)
        """
        python_code = replace_with_indent(python_code, "# ------------------------- Scatter-Plots -------------------------", plot_clipping)

    # ------------------------- Bereinigung Code -------------------------
    filtered_code_lines = []
    for line in python_code.splitlines():
        if "get_ipython()" in line:
            filtered_code_lines.append(f"# {line}  # Bereinigt")
        else:
            filtered_code_lines.append(line)
    
    final_code = "\n".join(filtered_code_lines)

    # ------------------------- Parameter-Injektion -------------------------
    if params_dict:
        injection_code_params = "\n# --- INJECTED PARAMETERS ---\n"
        for key, value in params_dict.items():
            if isinstance(value, str):
                injection_code_params += f"{key} = '{value}'\n"
            else:
                injection_code_params += f"{key} = {value}\n"
        
    # ------------------------- Injektion vor dem Main-Block (falls vorhanden), ansonsten am Ende anhängen -------------------------
    if 'if __name__ == "__main__":' in final_code:
        final_code = final_code.replace('if __name__ == "__main__":', f'{injection_code_params}\nif __name__ == "__main__":')
    else:
        final_code += injection_code_params

        # ------------------------- Logik für Training-Run Selektion (nur für 4.1) -------------------------
        if "TARGET_RUN_INDEX" in params_dict and "VAE_Imputation" in str(path):
             indent = "    "
             target_run_idx = params_dict["TARGET_RUN_INDEX"]
             
         logic_code = f"\n{indent}# ------------------------- INJIZIERTE LOGIK (FILTERUNG) -------------------------\n"
         logic_code += f"{indent}target_idx_usr = {target_run_idx}\n"
         logic_code += f"{indent}real_idx = target_idx_usr\n"
         logic_code += f"{indent}if real_idx > 0: real_idx += 1 \n"
         logic_code += f"{indent}if real_idx > 5: real_idx += 1 \n"
         logic_code += f"{indent}if real_idx < len(sorted_combos):\n"
         logic_code += f"{indent}    sorted_combos = [sorted_combos[real_idx]]\n"
         logic_code += f"{indent}    sorted_labels = [sorted_labels[real_idx]]\n"
         logic_code += f"{indent}else:\n"
         logic_code += f"{indent}    sorted_combos = [sorted_combos[0]]\n"
         logic_code += f"{indent}    sorted_labels = [sorted_labels[0]]\n"
             
             loop_line = "for i, combo in enumerate(sorted_combos):"
             if loop_line in final_code:
                 final_code = final_code.replace(loop_line, logic_code + "\n" + indent + loop_line)

    temp_script_path = path.parent / f"{path.stem}_Job_Temp.py"
    
    try:
        with open(temp_script_path, "w", encoding="utf-8") as f:
            f.write(final_code)
        
        #  ------------------------- Sequentielle Ausführung mit Live Output ------------------------- 
        p = subprocess.Popen(["python", "-u", str(temp_script_path)], cwd=path.parent, text=True, encoding='utf-8')
        p.wait()
        
        return p.returncode == 0
    except Exception as e:
        print(f"  -> Exception: {e}")
        return False
    finally:
        if temp_script_path.exists():
            try: os.remove(temp_script_path)
            except: pass

def main():
    print("==================================================")
    print("       PIPELINE 4 (FEATURE SELECTION)             ")
    print("==================================================")
    
    # ------------------------- Auswahlmenü -------------------------
    preview_data = get_run_preview_info()
    if isinstance(preview_data, list):
        print("\nVerfügbare Feature-Kombinationen:")
        print(f"{'Index':<6} | {'Name':<20} | {'Samples':<10} | {'Features':<10}")
        print("-" * 55)
        for entry in preview_data:
            print(f"{entry['idx']:<6} | {entry['name']:<20} | {entry['count']:<10} | {entry['features']:<10}")
        print("-" * 55)
    else:
        print(f"Warnung: {preview_data}")

    target_idx = 0
    valid = False
    while not valid:
        try:
            choice = input(f"Index wählen (0-5) [0]: ").strip()
            if choice == "": target_idx = 0; valid = True
            else:
                target_idx = int(choice)
                if 0 <= target_idx <= 5: valid = True
                else: print("Index ungültig.")
        except ValueError: print("Ungültige Eingabe.")
            
    print(f"--> Selektierter Index: {target_idx}")
    os.environ["VAE_MODE"] = "MANUAL"
    
    # ------------------------- Clipping Abfrage -------------------------
    print("\nSoll Quantile Clipping (2%) angewendet werden?")
    print("y) JA (Erzeugt 'geclippte' Darstellung in Plots)")
    print("n) NEIN (Original-Ausreißer bleiben sichtbar)")
    clip_choice = input("Auswahl (y/n) [n]: ").strip().lower()
    
    vae_clipping = "0"
    if clip_choice == "y":
        vae_clipping = "1"
    
    os.environ["VAE_CLIPPING"] = vae_clipping
    print(f"Clipping gesetzt auf: {'AKTIVIERT' if vae_clipping == '1' else 'DEAKTIVIERT'}")
    
    # ------------------------- 1. Training (4.1) -------------------------
    start_ts = time.time()
    success_4_1 = run_notebook_as_job("4.1 VAE Training", NOTEBOOK_4_1, {"TARGET_RUN_INDEX": target_idx, "MAX_ITERATIONS": 1})
    
    if not success_4_1:
        print("Abbruch wegen Fehler in 4.1")
        return

    # ------------------------- 2. Modell finden -------------------------
    models_root = NOTEBOOK_4_1.parent / "Models"
    time.sleep(2)
    candidates = [d for d in models_root.iterdir() if d.is_dir()]
    if not candidates: print("Kein Modell-Ordner gefunden!"); return
    latest_model = max(candidates, key=lambda d: d.stat().st_mtime)
    print(f"Aktives Modell-Verzeichnis: {latest_model.name}")

    # ------------------------- 3. Inferenz (4.2) -------------------------

    print(f"\n[2/3] Starte 4.2 Inferenz...")
    inference_script = NOTEBOOK_4_2.parent / "Inference_Job.py"
    try:
        # ------------------------- Wir übergeben FORCE_MODEL_FOLDER damit er genau diesen Ordner nimmt -------------------------
        p_4_2 = subprocess.Popen(["python", "-u", str(inference_script)], cwd=NOTEBOOK_4_2.parent, text=True, encoding='utf-8', env={**os.environ, "FORCE_MODEL_FOLDER": latest_model.name})
        p_4_2.wait()
        success_4_2 = p_4_2.returncode == 0
    except Exception as e:
        print(f"Fehler in 4.2: {e}")
        success_4_2 = False

    if not success_4_2:
        print("Abbruch wegen Fehler in 4.2")
        return

    # ------------------------- 4. Auswertung (4.3) -------------------------
    success_4_3 = run_notebook_as_job("4.3 VAE Evaluation", NOTEBOOK_4_3)
    
    if success_4_3:
        print("\n==================================================")
        print("       PIPELINE SUCCESSFULLY COMPLETED            ")
        print("==================================================")
    else:
        print("\nPipeline beendet mit Fehlern in der Auswertung.")

    input("\nDrücke [ENTER] zum Beenden...")

if __name__ == "__main__":
    main()
