import os
import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime

# ----------------------------------------------- Basisverzeichnis -----------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()

def run_notebook(notebook_path):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starte: {notebook_path.name}")
    try:
        start_time = time.time()
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            str(notebook_path)
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  -> Fertig in {time.time() - start_time:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  -> FEHLER: {e.stderr}")
        return False

def main():
    print("=== Pipeline 2 & 3: Analytik & Standard Machine Learning ===")
    
    # 2.1 - 2.3 Explorative Analysen
    notebooks = [
        BASE_DIR / "2_Analysis/2.1_Explorative-Datenanalyse/Data_Exploration.ipynb",
        BASE_DIR / "2_Analysis/2.2_Rock-Type_Analysis/Rock-Type_Analyzer.ipynb",
        BASE_DIR / "2_Analysis/2.3_Temperature_Analysis/Temperature_Analysis.ipynb"
    ]
    
    for nb in notebooks:
        if nb.exists(): run_notebook(nb)
        else: print(f"Warnung: {nb} nicht gefunden.")

    # Poster-Assets generieren
    asset_script = BASE_DIR / "generate_poster_assets.py"
    if asset_script.exists():
        subprocess.run([sys.executable, str(asset_script)], check=True)

    # 2.4 Datenqualität (Unterordner)
    dq_root = BASE_DIR / "2_Analysis/2.4_Data-Quality_Ionic-Balance-Error"
    if dq_root.exists():
        for sub in dq_root.iterdir():
            if sub.is_dir():
                nb_a = sub / "1.1_Data-Quality_Raw_Data.ipynb"
                nb_b = sub / "1.2_Data-Quality_Adjusted_Data.ipynb"
                if nb_a.exists(): run_notebook(nb_a)
                if nb_b.exists(): run_notebook(nb_b)
        
        # 2.4 Filterung
        filter_nb = dq_root / "2.4_Filter_Database.ipynb"
        if filter_nb.exists(): run_notebook(filter_nb)

    # 3. Machine Learning
    prep_nb = BASE_DIR / "3_Machine-Learning/3.1_Preprocessing/Preprocessing.ipynb"
    if prep_nb.exists(): run_notebook(prep_nb)
    
    som_nb = BASE_DIR / "3_Machine-Learning/3.2_Machine-Learning/MiniSom/MiniSom_Machine-Learning.ipynb"
    if som_nb.exists(): run_notebook(som_nb)

    print("\n--- Pipeline 2 & 3 abgeschlossen ---")

if __name__ == "__main__":
    main()
