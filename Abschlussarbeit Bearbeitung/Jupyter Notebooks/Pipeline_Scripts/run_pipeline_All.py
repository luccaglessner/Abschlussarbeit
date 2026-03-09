import subprocess
import sys
import time
import os
from datetime import datetime
from pathlib import Path

# ----------------------------------------------- Basisverzeichnis -----------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()

def run_script(script_name):
    script_path = BASE_DIR / script_name
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starte Sub-Pipeline: {script_name}")
    print(f"{'='*60}\n")
    
    if not script_path.exists():
        print(f"FEHLER: Skript nicht gefunden: {script_name}")
        return False
        
    try:
        start_time = time.time()
        # ----------------------------- sys.executable um gleichen Interpreter sicherzustellen -----------------------------
        cmd = [sys.executable, "-u", str(script_path)]
        
        # ----------------------------- Unterprozess ausführen und Ausgabe -----------------------------
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=BASE_DIR,
            env=os.environ.copy()
        )
        
        # ----------------------------- Echtzeit-Ausgabe -----------------------------
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode != 0:
            print(f"\nSub-Pipeline {script_name} fehlgeschlagen (Exit Code: {process.returncode}) !!!")
            return False
            
        print(f"\nSub-Pipeline {script_name} erfolgreich in {time.time() - start_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"\nFehler beim Starten von {script_name}: {e}")
        return False

def main():
    print(f"Komplette Pipeline starten: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Process ID: {os.getpid()}")
    print("-" * 60)

    # ----------------------------- Auto Mode Check -----------------------------
    auto_mode = "--auto" in sys.argv

    # ----------------------------- Global Batch Mode -----------------------------
    os.environ['PIPELINE_BATCH_MODE'] = '1'

    # ----------------------------- Daten-Download prüfen -----------------------------
    download_script = "../0_Preperation/0.2_Imports-and-Dependencies/download_data.py"
    print("\n[DATA CHECK] Überprüfe / Lade Quelldaten (Google Drive)...")
    if not run_script(download_script):
        print("WARNUNG: Daten-Download konnte nicht ausgeführt oder abgeschlossen werden.")

    # ----------------------------- Abfrage zu Timestamp Bereinigung -----------------------------
    print("\n[KONFIGURATION] Möchtest du die Timestamp-Ordner bereinigen?")
    if auto_mode:
        clean_choice = 'j'
        print("Deine Wahl (j/n): j (--auto Modus aktiv)")
    else:
        clean_choice = input("Deine Wahl (j/n): ").strip().lower()
    
    if clean_choice in ['j', 'ja', 'y', 'yes']:
        os.environ['TIMESTAMP_CLEANUP_CONFIRMED'] = '1'
        run_cleanup = True
        print(">> Bereinigung aktiviert.")
    else:
        run_cleanup = False
        print(">> Bereinigung übersprungen.")

    # ----------------------------- Abfrage zu LOOP Modus -----------------------------
    print("\n[KONFIGURATION] Soll Pipeline 3 (Machine Learning) im LOOP-Modus ausgeführt werden?")
    if auto_mode:
        loop_choice = 'j'
        print("Deine Wahl (j/n für LOOP/AUTO, sonst MANUAL): j (--auto Modus aktiv)")
    else:
        loop_choice = input("Deine Wahl (j/n für LOOP/AUTO, sonst MANUAL): ").strip().lower()
    
    if loop_choice in ['j', 'ja', 'y', 'yes', 'loop']:
        os.environ['SOM_MODE'] = 'LOOP'
        os.environ['SOM_LOOP_LIMIT'] = '50'
        print(">> Modus gesetzt: LOOP (Auto) - Limit: 50 Durchläufe")
    else:
        os.environ['SOM_MODE'] = 'MANUAL'
        print(">> Modus gesetzt: MANUAL")

    start_total = time.time()
    
    # ----------------------------- Ausführung -----------------------------

    # ----------------------------- Cleanup -----------------------------
    if run_cleanup:
        if not run_script("clear_timestamp_folders.py"):
            print("Abbruch wegen Fehler im Cleanup.")
            sys.exit(1)

    # ----------------------------- Pipelines -----------------------------
    pipelines = [
        "run_pipeline_1.py",
        "run_pipeline_2_3.py",
        "run_pipeline_4.py"
    ]
    
    for pipe in pipelines:
        success = run_script(pipe)
        if not success:
            print(f"\n{'-'*60}")
            print(f"CRITICAL: Pipeline gestoppt aufgrund von Fehler in {pipe}")
            print(f"{'-'*60}")
            sys.exit(1)
            
    print(f"\n{'='*60}")
    print(f"GESAMT-PIPELINE ERFOLGREICH ABGESCHLOSSEN")
    print(f"Gesamtdauer: {time.time() - start_total:.2f}s")
    print(f"{'='*60}")
    
    # ----------------------------- Am Ende warten -----------------------------
    if not auto_mode:
        input("Drücke [ENTER] um zu beenden...")

if __name__ == "__main__":
    main()
