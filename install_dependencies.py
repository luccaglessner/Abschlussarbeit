import os
import sys
import subprocess
from pathlib import Path

# ----------------------------------------------- Basisverzeichnis und requirements.txt Pfad -----------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"

def main():
    print(f"\n{'='*60}")
    print("   Überprüfe und installiere Pip-Abhängigkeiten...")
    print(f"{'='*60}\n")
    
    # ----------------------------------------------- Existenzprüfung -----------------------------------------------
    if not REQUIREMENTS_PATH.exists():
        print(f"WARNUNG: requirements.txt nicht gefunden unter: {REQUIREMENTS_PATH}")
        print("Installation kann nicht automatisch durchgeführt werden.")
        return False
        
    print(f"Verwende Datei: {REQUIREMENTS_PATH}")
    
    try:
        # ----------------------------------------------- Pip Installation ausführen -----------------------------------------------
        # Wir nutzen sys.executable um sicherzugehen, dass im aktiven Interpreter installiert wird
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)])
        print("\nAbhängigkeiten erfolgreich überprüft und ggf. aktualisiert.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nFEHLER bei der Installation der Abhängigkeiten: {e}")
        return False
    except Exception as e:
        print(f"\nUNERWARTETER FEHLER: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
         sys.exit(1)

