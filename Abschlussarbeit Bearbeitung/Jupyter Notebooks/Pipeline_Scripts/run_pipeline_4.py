
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pipeline_logger

# ------------------------- Konfiguration -------------------------
BASE_DIR = Path(r"C:\Users\lucca\OneDrive\SPEICHER\Hochschule\7. Semester\Abschlussarbeit Bearbeitung\Jupyter Notebooks")

# ----------------------------- Pfade zu den Notebooks -----------------------------
NOTEBOOK_4_1 = BASE_DIR / "4_Imputation" / "4.1_VAE_Imputation" / "VAE_Imputation.ipynb"
NOTEBOOK_4_2 = BASE_DIR / "4_Imputation" / "4.2_Inference" / "Inference.ipynb"
NOTEBOOK_4_3 = BASE_DIR / "4_Imputation" / "4.3_Evaluation" / "Evaluation.ipynb"

NOTEBOOKS_TO_RUN = [
    ("4.1 VAE Training", NOTEBOOK_4_1),
    ("4.2 VAE Inference", NOTEBOOK_4_2),
    ("4.3 VAE Evaluation", NOTEBOOK_4_3)
]

def run_notebook(name, path):
    # ----------------------------- Führt ein Jupyter Notebook mittels nbconvert aus -----------------------------
    print(f"\n--- Starte: {name} ---")
    print(f"Pfad: {path}")
    
    if not path.exists():
        print(f"FEHLER: Datei nicht gefunden: {path}")
        return False

    # ----------------------------- Arbeitsverzeichnis ist der Ordner, in dem das jeweilige Notebook liegt -----------------------------
    cwd = path.parent
    
    cmd = [
        "jupyter", "nbconvert", 
        "--to", "notebook", 
        "--execute", 
        "--inplace", 
        str(path)
    ]
    
    try:
        start_time = datetime.now()
        # ----------------------------- Output-Log festhalten -----------------------------
        result = subprocess.run(cmd, check=True, cwd=cwd, capture_output=True, text=True)
        duration = datetime.now() - start_time
        print(f"Erfolgreich abgeschlossen ({duration})")
        pipeline_logger.log_success(f"Pipeline 4 Schritt: {name}", f"Dauer: {duration}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Fehler bei der Ausführung von {name}!")

        # ----------------------------- Ausschnitt des Fehlers -----------------------------
        if e.stderr:
            print(f"--- Notebook Stderr ---\n{e.stderr[:1000]}...\n-----------------------")
        
        pipeline_logger.log_error(f"Pipeline 4 Step: {name}", exception=e, stderr=e.stderr)
        return False

def main():
    print("==================================================")
    print("       START PIPELINE 4 (Parallelsierung)      ")
    print("==================================================")
    
    # ----------------------------- Checken, ob Notebook existiert -----------------------------
    for name, path in NOTEBOOKS_TO_RUN:
        if not path.exists():
            print(f"Notebook nicht gefunden: {path}")
            return

    procs = []

    # ----------------------------- Start Training -----------------------------
    print(f"\n[1/3] Starte 4.1 für Training:")
    cmd_4_1 = ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(NOTEBOOK_4_1)]
    p_4_1 = subprocess.Popen(cmd_4_1, cwd=NOTEBOOK_4_1.parent)
    procs.append(("4.1 Training", p_4_1))
    
    # ----------------------------- Ordner erstellen, kurz warten -----------------------------
    print("5s auf Erstellen des Ordners warten...")
    import time
    time.sleep(5)

    # ----------------------------- Inferenzskript starten -----------------------------
    print(f"\n[2/3] Starte 4.2 für Inferenzskript:")
    cmd_4_2 = ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(NOTEBOOK_4_2)]
    p_4_2 = subprocess.Popen(cmd_4_2, cwd=NOTEBOOK_4_2.parent)
    procs.append(("4.2 Inference", p_4_2))
    
    time.sleep(5)

    # ----------------------------- Auswertung starten -----------------------------
    print(f"\n[3/3] Starte 4.3 für Auswertung:")
    cmd_4_3 = ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(NOTEBOOK_4_3)]
    p_4_3 = subprocess.Popen(cmd_4_3, cwd=NOTEBOOK_4_3.parent)
    procs.append(("4.3 Evaluation", p_4_3))

    print("\nAlle Prozesse laufen")
    
    # ------------------------- Fortschrittsanzeige -------------------------
    import glob
    
    # ----------------------------- Aktives Machine Learning Modell finden -----------------------------
    # hier: letztes Modell 
    start_ts = time.time()
    models_root = NOTEBOOK_4_1.parent / "Models"
    active_model_dir = None
    
    # ----------------------------- Hilfsfunktion -----------------------------
    def get_latest_model_dir():
        if not models_root.exists(): return None
        subdirs = [d for d in models_root.iterdir() if d.is_dir()]
        if not subdirs: return None
        # ----------------------------- Höchster Timestamp -----------------------------
        latest = max(subdirs, key=lambda d: d.stat().st_mtime)
        # ----------------------------- Schauen ob es neuste Modifikation is -----------------------------
        if latest.stat().st_mtime > start_ts - 60:
            return latest
        return None

    # ----------------------------- Warten bis Ordner generiert sind -----------------------------
    while active_model_dir is None:
        active_model_dir = get_latest_model_dir()
        if active_model_dir:
            print(f"Aktiven Ordner gefunden: {active_model_dir.name}")
            break
        
        # ----------------------------- Prüfen ob Training beendet ist -----------------------------
        if p_4_1.poll() is not None:
            print("4.1 Training beendet")
            break
        time.sleep(2)
        
    # ----------------------------- Prüfung Loop -----------------------------
    known_models = set()
    known_results = set()
    
    # ----------------------------- Ergebnisordner für 4.2 -----------------------------
    results_root = NOTEBOOK_4_2.parent / "Imputation_Results" / (active_model_dir.name if active_model_dir else "UNKNOWN")

    while True:
        # ----------------------------- Prüfen ob Prozesse beendet sind -----------------------------
        p41_status = p_4_1.poll()
        p42_status = p_4_2.poll()
        p43_status = p_4_3.poll()
        
        if p41_status is not None and p42_status is not None and p43_status is not None:
            break
            
        # ----------------------------- Prüfen ob Model Ordner generiert sind -----------------------------
        if active_model_dir and active_model_dir.exists():
            current_models = set(f.name for f in active_model_dir.glob("*_meta.json"))
            new_models = current_models - known_models
            for m in sorted(new_models):
                print(f"[4.1 Training] --> Erstellt {m.replace('_meta.json', '')}")
                known_models.add(m)
                
        # ----------------------------- Prüfen ob Ergebnis Ordner generiert sind -----------------------------
        if results_root.exists():
            current_results = set(f.name for f in results_root.glob("Imputation_Results_*.csv"))
            new_results = current_results - known_results
            for r in sorted(new_results):
                print(f"[4.2 Inference] --> {r} erstellt")
                known_results.add(r)
                
        # ----------------------------- Prüfen ob Evaluation Summary generiert ist -----------------------------
        if results_root.exists() and (results_root / "Evaluation_Summary.csv").exists():
            print("[4.3 Evaluation] --> Evaluation Summary erstellt")
            
        time.sleep(2)

    # ------------------------- Abschlussbericht -------------------------
    print("\nProzesse beendet.")
    exit_codes = [p_4_1.poll(), p_4_2.poll(), p_4_3.poll()]

    if all(c == 0 for c in exit_codes):
        print("\n==================================================")
        print("       PIPELINE 4 SUCCESSFULLY COMPLETED          ")
        print("==================================================")
    else:
        print("\n==================================================")
        print("       PIPELINE 4 FAILED (See Logs above)         ")
        print("==================================================")
        print(f"Exit Codes: 4.1={exit_codes[0]}, 4.2={exit_codes[1]}, 4.3={exit_codes[2]}")
    
    input("\nDrücke [ENTER] um das Skript zu beenden...")

if __name__ == "__main__":
    main()
