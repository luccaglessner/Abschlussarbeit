import os
import subprocess
import sys
import time
import pipeline_logger

from pathlib import Path
from datetime import datetime


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

    # ----------------------------- Aktives Machine Learning Modell finden -----------------------------
    start_ts = time.time()
    models_root = NOTEBOOK_4_1.parent / "Models"
    
    # ----------------------------- Hilfsfunktion -----------------------------
    def get_latest_model_dir():
        if not models_root.exists(): return None
        subdirs = [d for d in models_root.iterdir() if d.is_dir()]
        if not subdirs: return None
        # ----------------------------- Höchster Timestamp -----------------------------
        latest = max(subdirs, key=lambda d: d.stat().st_mtime)
        # ----------------------------- Schauen ob es neuste Modifikation ist -----------------------------
        if latest.stat().st_mtime > start_ts - 60:
            return latest
        return None

    # ----------------------------- Start Training -----------------------------
    print(f"\n[1/3] Starte 4.1 für Training (Direct Stream):")
    
    # ------------------------- Notebook zu Python Code -------------------------
    print("  -> Lade Code aus Notebook...")
    convert_cmd = ["jupyter", "nbconvert", "--to", "python", "--stdout", str(NOTEBOOK_4_1)]
    # ------------------------- Python Code aus Notebook -------------------------
    python_code = subprocess.check_output(convert_cmd, cwd=NOTEBOOK_4_1.parent).decode('utf-8', errors='ignore')
    
    # ------------------------- Magics entfernen -------------------------
    filtered_code_lines = []
    for line in python_code.splitlines():
        if "get_ipython()" in line:
            filtered_code_lines.append(f"# {line}  # Filtered Magic")
        else:
            filtered_code_lines.append(line)
    
    final_code = "\n".join(filtered_code_lines)
    
    print(f"  -> Starte Python-Prozess (Live Output)...")
    
    # ------------------------- Temporäres Python-Script erstellen -------------------------
    temp_script_path = NOTEBOOK_4_1.parent / "VAE_Imputation_Job.py"
    
    # ----------------------------- Cleanup Handler Vorbereiten -----------------------------
    import atexit
    import signal
    
    def cleanup():
        if temp_script_path.exists():
            try:
                os.remove(temp_script_path)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Temporäre Datei gelöscht: {temp_script_path.name}")
            except Exception as e:
                print(f"Warnung: Konnte Datei nicht löschen: {e}")

    # ------------------------- Cleanup Handler registrieren -------------------------
    atexit.register(cleanup)
    
    # ------------------------- Signal Handler --------------------------------
    def signal_handler(sig, frame):
        print("\nAbbruch signalisiert! Beende Prozesse...")
        cleanup()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        with open(temp_script_path, "w", encoding="utf-8") as f:
            f.write(final_code)
        
        # ------------------------- Python Prozess starten -------------------------
        cmd_4_1 = ["python", "-u", str(temp_script_path)]
        
        # ------------------------- UTF-8 Kodierung -------------------------
        p_4_1 = subprocess.Popen(cmd_4_1, cwd=NOTEBOOK_4_1.parent, text=True, encoding='utf-8')
      
        
        procs.append(("4.1 Training", p_4_1))
        
        # ----------------------------- Warten auf Modell-Ordner -----------------------------
        print("Warte auf Erstellen des Modell-Ordners durch 4.1...")
        active_model_dir = None
        
        while active_model_dir is None:
            # ------------------------- Vergangenes Problem: Viele Abstürze bei 4.1 - hier Prüfung -------------------------
            if p_4_1.poll() is not None:
                print("4.1 Training wurde unerwartet beendet")
                break
                
            active_model_dir = get_latest_model_dir()
            if active_model_dir:
                print(f"Aktiven Ordner gefunden: {active_model_dir.name}. Generierung kann anfangs bis zu 15 Minuten dauern.")
                break
            time.sleep(2)

        p_4_2 = None
        p_4_3 = None

        if active_model_dir:
            # ----------------------------- Inferenzskript starten -----------------------------
            print(f"\n[2/3] Starte 4.2 für Inferenzskript:")
            cmd_4_2 = ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "--ExecutePreprocessor.timeout=-1", str(NOTEBOOK_4_2)]
            p_4_2 = subprocess.Popen(cmd_4_2, cwd=NOTEBOOK_4_2.parent)
            procs.append(("4.2 Inference", p_4_2))
            
            results_root = NOTEBOOK_4_2.parent / "Inference_Results" / active_model_dir.name
            print(f"Warte auf Ergebnis-Ordner: {results_root.name}...")
            print(f"Pfad erwartet: {results_root}")
            
            results_created = False
            while not results_created:
                 if p_4_2.poll() is not None:
                     print("4.2 Inference wurde unerwartet beendet!")
                     break
                 
                 if results_root.exists():
                     print("Ergebnis-Ordner wurde erstellt.")
                     results_created = True
                     break
                 time.sleep(2)
                 
            if results_created:
                # ----------------------------- Auswertung starten -----------------------------
                print(f"\n[3/3] Starte 4.3 für Auswertung:")
                cmd_4_3 = ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "--ExecutePreprocessor.timeout=-1", str(NOTEBOOK_4_3)]
                p_4_3 = subprocess.Popen(cmd_4_3, cwd=NOTEBOOK_4_3.parent)
                procs.append(("4.3 Evaluation", p_4_3))

        if not p_4_1 and not p_4_2 and not p_4_3: # ------------------------- Fehlerüberprüfung, da Probleme mit Abstürzen -------------------------
             print("Keine Prozesse gestartet.")
             return

        print("\nAlle Prozesse laufen")
        
        # ------------------------- Fortschrittsanzeige -------------------------
        known_models = set()
        known_results = set()
        known_pdfs = set()
        
        # ----------------------------- Ergebnisordner für 4.2 definieren falls noch nicht geschehen -----------------------------
        results_root = None
        eval_results_dir = None
        if active_model_dir:
             results_root = NOTEBOOK_4_2.parent / "Inference_Results" / active_model_dir.name
             eval_results_dir = NOTEBOOK_4_3.parent / "Evaluation_Results" / active_model_dir.name

        while True:
            # ----------------------------- Prüfen ob Prozesse beendet sind -----------------------------
            p41_status = p_4_1.poll() if p_4_1 else None
            p42_status = p_4_2.poll() if p_4_2 else None
            p43_status = p_4_3.poll() if p_4_3 else None
            
            # ------------------------- Schauen ob alle Prozesse beendet wurden -------------------------
            all_done = True
            if p_4_1 and p41_status is None: all_done = False
            if p_4_2 and p42_status is None: all_done = False
            if p_4_3 and p43_status is None: all_done = False
            
            if all_done:
                break
                
            # ----------------------------- Prüfen ob Model Ordner generiert sind -----------------------------
            if active_model_dir and active_model_dir.exists():
                current_models = set(f.name for f in active_model_dir.glob("*_meta.json"))
                new_models = current_models - known_models
                for m in sorted(new_models):
                    print(f"[4.1 Training] --> Erstellt {m.replace('_meta.json', '')}")
                    known_models.add(m)
                    
            # ----------------------------- Prüfen ob Ergebnis Ordner generiert sind -----------------------------
            if results_root and results_root.exists():
                current_results = set(f.name for f in results_root.glob("Imputation_Results_*.csv"))
                new_results = current_results - known_results
                for r in sorted(new_results):
                    print(f"[4.2 Inference] --> {r} erstellt")
                    known_results.add(r)
            
            # ----------------------------- Prüfen ob Evaluation PDFs generiert sind -----------------------------
            if eval_results_dir and eval_results_dir.exists():
                current_pdfs = set(f.name for f in eval_results_dir.glob("Analysis_Run_*.pdf"))
                new_pdfs = current_pdfs - known_pdfs
                for p in sorted(new_pdfs):
                     print(f"[4.3 Evaluation] --> {p} erstellt")
                     known_pdfs.add(p)
                    
            # ----------------------------- Prüfen ob Evaluation Summary generiert ist -----------------------------
            if eval_results_dir and eval_results_dir.exists() and (eval_results_dir / "Summary_Evaluation.csv").exists():
                print("[4.3 Evaluation] --> Evaluation Summary erstellt")
                
            time.sleep(2)

        # ------------------------- Abschlussbericht -------------------------
        print("\nProzesse beendet.")
        
        code_4_1 = p_4_1.poll() if p_4_1 else -1
        code_4_2 = p_4_2.poll() if p_4_2 else -1
        code_4_3 = p_4_3.poll() if p_4_3 else -1
        
        exit_codes = [c for c in [code_4_1, code_4_2, code_4_3] if c != -1]
        

        success = (len(exit_codes) > 0) and all(c == 0 for c in exit_codes)

        if success:
            print("\n==================================================")
            print("       PIPELINE 4 SUCCESSFULLY COMPLETED          ")
            print("==================================================")
        else:
            print("\n==================================================")
            print("       PIPELINE 4 FAILED (See Logs above)         ")
            print("==================================================")
            print(f"Exit Codes: 4.1={code_4_1}, 4.2={code_4_2}, 4.3={code_4_3}")
            print("Note: -1 bedeutet, dass der Prozess nicht gestartet wurde.")
        
    except Exception as e:
        print(f"\nFATAL ERROR in Pipeline: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ------------------------- Temporäres Script löschen -------------------------
        if 'temp_script_path' in locals() and temp_script_path.exists():
             try:
                os.remove(temp_script_path)
             except: pass
    
    input("\nDrücke [ENTER] um das Skript zu beenden...")

if __name__ == "__main__":
    main()
