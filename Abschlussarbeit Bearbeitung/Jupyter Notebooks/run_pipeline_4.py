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
    print("=== Pipeline 4: Imputation & Machine Learning ===")
    
    # 4.1 Imputation
    nb_41 = BASE_DIR / "4_Imputation/4.1_Imputation/VAE_Imputation.ipynb"
    if nb_41.exists():
        if not run_notebook(nb_41):
            print("Kritischer Fehler in 4.1. Abbruch.")
            sys.exit(1)
    else:
        print(f"Fehler: {nb_41} nicht gefunden.")

    # 4.2 Vorverarbeitung
    nb_42 = BASE_DIR / "4_Imputation/4.2_Preprocessing/Preprocessing.ipynb"
    if nb_42.exists():
        if not run_notebook(nb_42):
            print("Kritischer Fehler in 4.2. Abbruch.")
            sys.exit(1)
    else:
        print(f"Fehler: {nb_42} nicht gefunden.")

    # 4.3 Maschinelles Lernen (Haupt-Ionen)
    nb_43 = BASE_DIR / "4_Imputation/4.3_Machine-Learning-Main-Ions/MiniSom_Machine-Learning.ipynb"
    if nb_43.exists(): run_notebook(nb_43)
    else: print(f"Fehler: {nb_43} nicht gefunden.")

    # 4.4 Maschinelles Lernen (Alle Features)
    nb_44 = BASE_DIR / "4_Imputation/4.4_Machine-Learning-Everything/MiniSom_Machine-Learning-Everything.ipynb"
    if nb_44.exists(): run_notebook(nb_44)
    else: print(f"Fehler: {nb_44} nicht gefunden.")

    print("\n--- Pipeline 4 abgeschlossen ---")

if __name__ == "__main__":
    main()
