import os
import subprocess
import sys
from pathlib import Path
import time
import signal
import atexit
from datetime import datetime

# ----------------------------------------------- Basisverzeichnis -----------------------------------------------
# Basisverzeichnis ist Elternordner
BASE_DIR = Path(__file__).parent.parent.resolve()
LOCK_FILE = Path(__file__).parent / "Pipeline_Locks" / "pipeline_2_3.lock"

import pipeline_logger

def check_and_kill_existing_process():

    # ----------------------------- Check for lock file -----------------------------
    if LOCK_FILE.exists():
        try:
            with open(LOCK_FILE, 'r') as f:
                old_pid = int(f.read().strip())
            
            # ----------------------------- Check if process exists -----------------------------
            try:
                os.kill(old_pid, 0) 
                print(f"WARNUNG: Laufende Pipeline-Instanz (2 & 3) gefunden (PID: {old_pid}). Beende Prozess...")
                
                os.kill(old_pid, signal.SIGTERM)
                time.sleep(1)
                
                try:
                    os.kill(old_pid, 0)
                    print(f"Instanz (PID: {old_pid}) reagiert nicht. Erzwinge Beendigung (SIGKILL)...")
                    os.kill(old_pid, signal.SIGTERM) 
                except OSError:
                    pass 
                
                print(f"Vorherige Instanz (PID: {old_pid}) erfolgreich beendet.")
                
            except OSError:
                print(f"Stale Lock-File gefunden (Prozess {old_pid} existiert nicht mehr). Bereinige...")
        
        except ValueError:
             print("Beschädigtes Lock-File gefunden. Bereinige...")
        except Exception as e:
            print(f"Fehler beim Prüfen der vorherigen Instanz: {e}")
        
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()

def register_current_process():
    pid = os.getpid()
    with open(LOCK_FILE, 'w') as f:
        f.write(str(pid))

def cleanup_lock():
    if LOCK_FILE.exists():
        try:
            with open(LOCK_FILE, 'r') as f:
                pid = int(f.read().strip())
            if pid == os.getpid():
                LOCK_FILE.unlink()
        except:
            pass 

def run_notebook(notebook_path):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starte: {notebook_path.name}")
    try:
        start_time = time.time()
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            str(notebook_path)
        ]

        # ----------------------------- Stderr für Logging erfassen -----------------------------
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
    # ----------------------------- Singleton Logic -----------------------------
    check_and_kill_existing_process()
    register_current_process()
    atexit.register(cleanup_lock)
    # ---------------------------------------------------------------------------

    print("=== Pipeline 2 & 3: Analytik & Standard Machine Learning ===")
    print(f"Process ID: {os.getpid()}")
    
    # ----------------------------- Interaktive Abfrage: Modus für SOM -----------------------------
    print("\n[KONFIGURATION] Bitte wähle den Ausführungsmodus für Machine Learning (Schritt 3.2):")
    print("  1) MANUAL (Standard: Konfiguration im Notebook)")
    print("  2) AUTO / LOOP (Automatische Kombinationstestung)")
    choice = input("Deine Wahl (1/2): ").strip()
    
    if choice == '2':
        os.environ['SOM_MODE'] = 'LOOP'
        print(">> Modus gesetzt: LOOP (Auto)\n")
    else:
        os.environ['SOM_MODE'] = 'MANUAL'
        print(">> Modus gesetzt: MANUAL\n")

    
    # ----------------------------- 2.1 - 2.3 Explorative Analysen -----------------------------
    notebooks = [
        BASE_DIR / "2_Analysis/2.1_Explorative-Datenanalyse/Data_Exploration.ipynb",
        BASE_DIR / "2_Analysis/2.2_Rock-Type_Analysis/Rock-Type_Analyzer.ipynb",
        BASE_DIR / "2_Analysis/2.3_Temperature_Analysis/Temperature_Analysis.ipynb"
    ]
    
    for nb in notebooks:
        if nb.exists(): run_notebook(nb)
        else: print(f"Warnung: {nb} nicht gefunden.")

    # ----------------------------- Poster-Assets generieren -----------------------------
    asset_script = BASE_DIR / "generate_poster_assets.py"
    if asset_script.exists():
        subprocess.run([sys.executable, str(asset_script)], check=True)

    # ----------------------------- 2.4 Datenqualität (Unterordner) -----------------------------
    dq_root = BASE_DIR / "2_Analysis/2.4_Data-Quality_Ionic-Balance-Error"
    if dq_root.exists():
        for sub in dq_root.iterdir():
            if sub.is_dir():
                nb_a = sub / "1.1_Data-Quality_Raw_Data.ipynb"
                nb_b = sub / "1.2_Data-Quality_Adjusted_Data.ipynb"
                if nb_a.exists(): run_notebook(nb_a)
                if nb_b.exists(): run_notebook(nb_b)
        
        # ----------------------------- 2.4 Filterungung -----------------------------
        filter_nb = dq_root / "2.4_Filter_Database.ipynb"
        if filter_nb.exists(): run_notebook(filter_nb)

    # ----------------------------- 2.5 Full Datasets Analysis (Vollständigkeits-Check) -----------------------------
    full_analysis_nb = BASE_DIR / "2_Analysis/2.5_Full-Datasets-Analysis/Full_Dataset_Analysis.ipynb"
    if full_analysis_nb.exists(): run_notebook(full_analysis_nb)

    # ----------------------------- 3. Machine Learning -----------------------------
    prep_nb = BASE_DIR / "3_Machine-Learning/3.1_Preprocessing/Preprocessing.ipynb"
    if prep_nb.exists(): run_notebook(prep_nb)
    
    som_nb = BASE_DIR / "3_Machine-Learning/3.2_Machine-Learning/MiniSom/MiniSom_Machine-Learning.ipynb"
    if som_nb.exists(): run_notebook(som_nb)

    print("\n--- Pipeline 2 & 3 abgeschlossen ---")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Die Pipeline wurde unerwartet beendet:\n{e}")
    finally:
        print("\n" + "="*60)
        pass
