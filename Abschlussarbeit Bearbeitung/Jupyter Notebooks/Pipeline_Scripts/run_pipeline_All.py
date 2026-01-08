import subprocess
import sys
import time
import os
import signal
import atexit
from datetime import datetime
from pathlib import Path

# ----------------------------------------------- Basisverzeichnis -----------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
LOCK_FILE = BASE_DIR / "Pipeline_Locks" / "pipeline_all.lock"

def check_and_kill_existing_process():
    """
    Checks for an existing lock file. If found, reads the PID and terminates the process.
    """
    if LOCK_FILE.exists():
        try:
            with open(LOCK_FILE, 'r') as f:
                old_pid = int(f.read().strip())
            
            # Check if process exists
            try:
                # Provides a way to check if process exists without killing it first (0 signal)
                os.kill(old_pid, 0) 
                print(f"WARNUNG: Laufende Pipeline-Instanz gefunden (PID: {old_pid}). Beende Prozess...")
                
                # Terminate properly
                os.kill(old_pid, signal.SIGTERM)
                
                # Give it a moment to die
                time.sleep(1)
                
                # Check again, force kill if needed
                try:
                    os.kill(old_pid, 0)
                    print(f"Instanz (PID: {old_pid}) reagiert nicht. Erzwinge Beendigung (SIGKILL)...")
                    os.kill(old_pid, signal.SIGTERM) # Windows uses SIGTERM as kill mostly, but we trigger it again
                except OSError:
                    pass # Process is gone
                
                print(f"Vorherige Instanz (PID: {old_pid}) erfolgreich beendet.")
                
            except OSError:
                print(f"Stale Lock-File gefunden (Prozess {old_pid} existiert nicht mehr). Bereinige...")
        
        except ValueError:
             print("Beschädigtes Lock-File gefunden. Bereinige...")
        except Exception as e:
            print(f"Fehler beim Prüfen der vorherigen Instanz: {e}")
        
        # Cleanup lock file
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()

def register_current_process():
    """
    Writes the current PID to the lock file.
    """
    pid = os.getpid()
    with open(LOCK_FILE, 'w') as f:
        f.write(str(pid))

def cleanup_lock():
    """
    Removes the lock file on exit.
    """
    if LOCK_FILE.exists():
        # Only remove if it contains OUR pid (safety check)
        try:
            with open(LOCK_FILE, 'r') as f:
                pid = int(f.read().strip())
            if pid == os.getpid():
                LOCK_FILE.unlink()
        except:
            pass # Best effort

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
    # --- Singleton Logic ---
    check_and_kill_existing_process()
    register_current_process()
    atexit.register(cleanup_lock)
    # -----------------------

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
