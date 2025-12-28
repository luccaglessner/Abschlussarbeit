import os
import subprocess
import sys
from pathlib import Path
import time

# Basisverzeichnis (Ordner in dem dieses Skript liegt)
BASE_DIR = Path(__file__).parent.resolve()

def run_notebook(notebook_path):
    """Führt ein Notebook aus und speichert es inplace."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starte: {notebook_path.name}")
    try:
        start_time = time.time()
        # Verwende jupyter nbconvert via subprocess
        # --execute: Führt Zellen aus
        # --inplace: Speichert Ergebnisse in der gleichen Datei
        # --to notebook: Zielformat
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            str(notebook_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
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

from datetime import datetime

def main():
    print("========================================================")
    print("   Starte Ausführung der Jupyter Notebook Pipeline")
    print("========================================================")
    print(f"Basisverzeichnis: {BASE_DIR}\n")

    # Liste der auszuführenden Schritte (1-4 sind einzelne Dateien)
    steps = [
        # 1. Akquise
        BASE_DIR / "1.1_Data-Acquisition-Wrapper" / "Data-Acquisition-Wrapper.ipynb",
        
        # 2. Analytik (Exploration)
        BASE_DIR / "2.1_Explorative-Datenanalyse" / "Data_Exploration.ipynb",
        
        # 3. Analytik (Rock Type)
        BASE_DIR / "2.2_Rock-Type_Analysis" / "Rock-Type_Analyzer.ipynb",
        
        # 4. Temperatur Analyse
        BASE_DIR / "2.3_Temperature_Analysis" / "Temperature_Analysis.ipynb"
    ]

    # Schritt 1-4 ausführen
    for i, nb_path in enumerate(steps, 1):
        if not nb_path.exists():
            print(f"WARNUNG: Datei nicht gefunden: {nb_path}")
            continue
            
        print(f"\n--- Schritt {i}: {nb_path.parent.name} ---")
        success = run_notebook(nb_path)
        if not success:
            print("Abbruch der Pipeline aufgrund von Fehlern.")
            input("Taste drücken zum Beenden...")
            sys.exit(1)

    # Schritt 5: Data Quality (Iterieren über Unterordner)
    print(f"\n--- Schritt 5: 2.4_Data-Quality_Ionic-Balance-Error ---")
    dq_root = BASE_DIR / "2.4_Data-Quality_Ionic-Balance-Error"
    
    if not dq_root.exists():
        print(f"WARNUNG: Ordner nicht gefunden: {dq_root}")
    else:
        # Alle Unterordner finden
        subfolders = [f for f in dq_root.iterdir() if f.is_dir()]
        print(f"Gefundene Varianten: {[f.name for f in subfolders]}")
        
        for sub in subfolders:
            print(f"\n  > Verarbeite Variante: {sub.name}")
            
            # Ablauf a und b
            nb_a = sub / "1.1_Data-Quality_Raw_Data.ipynb"
            nb_b = sub / "1.2_Data-Quality_Adjusted_Data.ipynb"
            
            # Ausführen wenn vorhanden
            if nb_a.exists():
                if not run_notebook(nb_a): 
                    print("Warnung: Fehler in 1.1, fahre dennoch fort...")
            
            if nb_b.exists():
                if not run_notebook(nb_b):
                    print("Warnung: Fehler in 1.2, fahre dennoch fort...")

    # Schritt 6: Filter Database (Erzeugt neue DB Version)
    print(f"\n--- Schritt 6: 2.4_Filter_Database (IBE Filter) ---")
    filter_nb = dq_root / "2.4_Filter_Database.ipynb"
    if filter_nb.exists():
        if not run_notebook(filter_nb):
            print("FEHLER beim Filtern der Datenbank! Pipeline wird gestoppt.")
            input("Taste drücken zum Beenden...")
            sys.exit(1)
    else:
        print(f"Fehler: Filter-Notebook nicht gefunden: {filter_nb}")
        sys.exit(1)

    # Schritt 7: Preprocessing
    print(f"\n--- Schritt 7: 3.1_Preprocessing ---")
    prep_nb = BASE_DIR / "3.1_Preprocessing" / "Preprocessing.ipynb"
    if prep_nb.exists():
        if not run_notebook(prep_nb):
            print("FEHLER im Preprocessing! Pipeline wird gestoppt.")
            input("Taste drücken zum Beenden...")
            sys.exit(1)
    else:
        print(f"Fehler: Preprocessing-Notebook nicht gefunden: {prep_nb}")
        sys.exit(1)

    print("\n========================================================")
    print("   Pipeline vollständig ausgeführt!")
    print("========================================================")
    input("Taste drücken zum Schließen...")

if __name__ == "__main__":
    main()
