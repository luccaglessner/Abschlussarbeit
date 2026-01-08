import os
import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime

# ----------------------------------------------- Basisverzeichnis -----------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()

import pipeline_logger

def run_notebook(notebook_path):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starte: {notebook_path.name}")
    try:
        start_time = time.time()
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            str(notebook_path)
        ]
        # Stderr für Logging erfassen
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  -> Fertig in {time.time() - start_time:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  -> FEHLER: {e.stderr}")
        pipeline_logger.log_error(f"Fehler bei Ausführung von {notebook_path.name}", exception=e, stderr=e.stderr)
        return False
    except Exception as e:
        print(f"  -> UNERWARTETER FEHLER: {e}")
        pipeline_logger.log_error(f"Unerwarteter Fehler bei {notebook_path.name}", exception=e)
        return False

def main():
    print("=== Pipeline 5: Synthetische Daten & Machine Learning ===")
    
    # 5.1 Synthetic Data Generation (VAE)
    nb_51 = BASE_DIR / "5_Synthetic-Data/5.1_Synthetic-Data/Variational-Autoencoder/VAE_Synthetic_Data_Generation.ipynb"
    if nb_51.exists():
        if not run_notebook(nb_51):
            print("Kritischer Fehler in 5.1. Abbruch.")
            sys.exit(1)
    else:
        print(f"Fehler: {nb_51} nicht gefunden.")

    # 5.2 Preprocessing (Synthetic)
    nb_52 = BASE_DIR / "5_Synthetic-Data/5.2_Preprocessing/Preprocessing.ipynb"
    if nb_52.exists():
        if not run_notebook(nb_52):
            print("Kritischer Fehler in 5.2. Abbruch.")
            sys.exit(1)
    else:
        print(f"Fehler: {nb_52} nicht gefunden.")

    # 5.3 Machine Learning (Synthetic SOM)
    nb_53 = BASE_DIR / "5_Synthetic-Data/5.3_Machine-Learning/MiniSom/MiniSom_Machine-Learning.ipynb"
    if nb_53.exists(): run_notebook(nb_53)
    else: print(f"Fehler: {nb_53} nicht gefunden.")

    print("\n--- Pipeline 5 abgeschlossen ---")

if __name__ == "__main__":
    main()
