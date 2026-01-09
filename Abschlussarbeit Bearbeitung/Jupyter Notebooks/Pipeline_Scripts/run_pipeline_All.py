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
        # Use sys.executable to ensure we use the same python interpreter
        cmd = [sys.executable, "-u", str(script_path)]
        
        # Run subprocess and stream output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=BASE_DIR
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode != 0:
            print(f"\n!!! Sub-Pipeline {script_name} fehlgeschlagen (Exit Code: {process.returncode}) !!!")
            return False
            
        print(f"\n-> Sub-Pipeline {script_name} erfolgreich in {time.time() - start_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"\nUNERWARTETER FEHLER beim Starten von {script_name}: {e}")
        return False

def main():
    print(f"Starting FULL PIPELINE Execution at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Process ID: {os.getpid()}")
    start_total = time.time()
    
    pipelines = [
        "run_pipeline_1.py",
        "run_pipeline_2_3.py",
        "run_pipeline_4.py",
        "run_pipeline_5.py"
    ]
    
    for pipe in pipelines:
        success = run_script(pipe)
        if not success:
            print(f"\n{'-'*60}")
            print(f"CRITICAL: Pipeline stopped due to error in {pipe}")
            print(f"{'-'*60}")
            sys.exit(1)
            
    print(f"\n{'='*60}")
    print(f"GESAMT-PIPELINE ERFOLGREICH ABGESCHLOSSEN")
    print(f"Gesamtdauer: {time.time() - start_total:.2f}s")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
