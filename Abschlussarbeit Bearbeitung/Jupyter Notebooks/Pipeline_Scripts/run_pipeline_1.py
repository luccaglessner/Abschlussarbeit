import os
import subprocess
import sys
import time

from pathlib import Path
from datetime import datetime


# ----------------------------------------------- Basisverzeichnis definieren -----------------------------------------------
BASE_DIR = Path(__file__).parent.parent.resolve()

def run_notebook(notebook_path):
    # ------------------------- Führt ein Notebook aus und speichert es inplace -------------------------
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starte: {notebook_path.name}")
    try:
        start_time = time.time()
        # ------------------------- Jupyter nbconvert via subprocess ausführen -------------------------
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            str(notebook_path)
        ]
        
        # ------------------------- CWD auf Notebook-Ordner setzen, damit relative Pfade funktionieren -------------------------
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=notebook_path.parent)
        duration = time.time() - start_time
        print(f"  -> Fertig in {duration:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  -> FEHLER bei {notebook_path.name}!")
        print(f"  -> Return Code: {e.returncode}")
        print(f"  -> Stderr:\n{e.stderr}")
        return False
    except Exception as e:
        print(f"  -> UNERWARTETER FEHLER: {e}")
        return False

def main():
    print("========================================================")
    print("   Starte Ausführung: Data Acquisition Wrapper")
    print("========================================================")
    print(f"Basisverzeichnis: {BASE_DIR}\n")

    # --------------------------- Schritt 1: Akquise ---------------------------
    step_1 = BASE_DIR / "1_Acquisition/1.1_Data-Acquisition-Wrapper" / "Data-Acquisition-Wrapper.ipynb"
    
    if not step_1.exists():
        print(f"WARNUNG: Datei nicht gefunden: {step_1}")
    else:
        print(f"\n--- Schritt 1: {step_1.parent.name} ---")
        success = run_notebook(step_1)
        if not success:
            print("Abbruch der Pipeline aufgrund von Fehlern.")
            input("Taste [ENTER] drücken zum Beenden...")
            sys.exit(1)

    print("\n========================================================")
    print("   Data Acquisition erfolgreich abgeschlossen!")
    print("========================================================")
    if os.environ.get("PIPELINE_BATCH_MODE") != "1":
        input("Taste [ENTER] drücken zum Schließen...")

if __name__ == "__main__":
    main()
