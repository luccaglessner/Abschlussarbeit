import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# ------------------------- Konfiguration -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = BASE_DIR / "3_Machine-Learning" / "3.2_Machine-Learning" / "MiniSom" / "MiniSom_Machine-Learning.ipynb"

# ------------------------- Feature Kombinationen laut Tabelle 2 -------------------------
FIXED_BASE_FEATURES = [
    "Na_in_mmol/L", "Mg_in_mmol/L", "Ca_in_mmol/L", 
    "Cl_in_mmol/L", "SO4_in_mmol/L", "HCO3_in_mmol/L"
]

SOM_COMBINATIONS = [
    {"name": "1_Base",           "add": []},
    {"name": "2_Base_pH",        "add": ["pH"]},
    {"name": "3_Base_pH_Fe",     "add": ["pH", "Fe_in_mmol/L"]},
    {"name": "4_Base_pH_Fe_K",   "add": ["pH", "Fe_in_mmol/L", "K_in_mmol/L"]},
    {"name": "5_Base_pH_Fe_K_Mn","add": ["pH", "Fe_in_mmol/L", "K_in_mmol/L", "Mn_in_mmol/L"]}
]

GRID_SIZES = range(2, 7) # ------------------------- 2x2 bis 6x6

import re

def run_som_notebook(som_x, som_y, selection_names, run_id, timestamp):
    """Führt das MiniSom Notebook mit injizierten Parametern aus."""
    print(f"\n[RUN] {run_id} | Grid: {som_x}x{som_y} | Features: {len(selection_names)}")
    
    # ------------------------- Notebook zu Python konvertieren -------------------------
    convert_cmd = ["jupyter", "nbconvert", "--to", "python", "--stdout", str(NOTEBOOK_PATH)]
    try:
        python_code = subprocess.check_output(convert_cmd, cwd=NOTEBOOK_PATH.parent).decode('utf-8', errors='ignore')
    except subprocess.CalledProcessError as e:
        print(f"Fehler bei Konvertierung: {e}")
        return False

    #  ------------------------- IPython Magics filtern -------------------------
    filtered_code = "\n".join([line if "get_ipython()" not in line else f"# {line}" for line in python_code.splitlines()])

    # ------------------------- Definition überschreiben -------------------------
    overrides = {
        r"^\s*SOM_X\s*=\s*.*": f"SOM_X = {som_x}",
        r"^\s*SOM_Y\s*=\s*.*": f"SOM_Y = {som_y}",
        r"^\s*EXECUTION_MODE\s*=\s*.*": "EXECUTION_MODE = 'MANUAL'",
        r"^\s*display_combos\s*=\s*.*": f"display_combos = [{selection_names}]",
        r"^\s*display_labels\s*=\s*.*": f"display_labels = ['{run_id}']",
        r"^\s*SESSION_TIMESTAMP\s*=\s*.*": f"SESSION_TIMESTAMP = '{timestamp}'",
        r"^\s*base\s*=\s*.*": "base = []",
        r"^\s*pool\s*=\s*.*": "pool = []"
    }

    final_code = filtered_code
    for pattern, replacement in overrides.items():
        final_code = re.sub(pattern, replacement, final_code, flags=re.MULTILINE)

    # ------------------------- Nur einma in MANUAL -------------------------

    final_code = re.sub(r"for\s+i\s+in\s+range\(min\(11,\s*len\(display_combos\)\)\):", "for i in range(1):", final_code)
    final_code = re.sub(r"for\s+i,\s+combo\s+in\s+enumerate\(display_combos\):", "for i, combo in enumerate(display_combos[:1]):", final_code)

    temp_script = NOTEBOOK_PATH.parent / f"temp_run_{run_id}.py"
    try:
        with open(temp_script, "w", encoding="utf-8") as f:
            f.write(final_code)
        
        subprocess.run(["python", "-u", str(temp_script)], cwd=NOTEBOOK_PATH.parent, check=True)
        return True
    except Exception as e:
        print(f"Fehler bei Ausführung {run_id}: {e}")
        return False
    finally:
        if temp_script.exists():
            try:
                temp_script.unlink()
            except:
                pass

def main():
    print("=========================================================")
    print("   SOM GRID-SEARCH PIPELINE (2x2 - 6x6)")
    print("   Basierend auf Tabelle 2 Kombinationen")
    print("=========================================================")
    
    if not NOTEBOOK_PATH.exists():
        print(f"FEHLER: Notebook nicht gefunden unter {NOTEBOOK_PATH}")
        return

    start_time = time.time()
    # ------------------------- Gemeinsamer Zeitstempel für alle GridSearch-Läufe -------------------------
    global_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_GridSearch"
    
    for combo in SOM_COMBINATIONS:
        selection = FIXED_BASE_FEATURES + combo["add"]
        combo_name = combo["name"]
        
        for size in GRID_SIZES:
            run_id = f"{combo_name}_{size}x{size}"
            success = run_som_notebook(size, size, selection, run_id, global_ts)
            if not success:
                print(f"Abbruch bei {run_id}")
    
    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f"\n=========================================================")
    print(f"PIPELINE ABGESCHLOSSEN in {duration:.2f} Minuten.")
    print(f"Ergebnisse befinden sich im Ordner MiniSom_Results/{global_ts}")
    print("=========================================================")

if __name__ == "__main__":
    main()
