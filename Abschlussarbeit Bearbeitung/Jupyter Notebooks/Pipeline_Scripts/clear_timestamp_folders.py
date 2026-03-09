
import os
import shutil
import re
from pathlib import Path
import traceback

# ----------------------------- Konfiguration zu übergeordnetem Ordner -----------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
TARGET_PREFIXES = ('1', '2', '3', '4', '5')
TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

def is_timestamp_folder(path_name):
    return bool(TIMESTAMP_PATTERN.match(path_name))

def on_rm_error(func, path, exc_info):
    # ----------------------------- Read-Only Dateien behandeln -----------------------------
    import stat
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        print(f"-> Fehler beim Freigeben / Löschen von {path}: {e}")

def clean_timestamp_folders():
    if not ROOT_DIR.exists():
        print(f"Error: Root directory {ROOT_DIR} does not exist.")
        return

    # ----------------------------- Scan -----------------------------
    tasks = [] # --------------------------- Liste von (prefix, parent_path, newest_dir, dirs_to_delete)

    # ----------------------------- Durchgehen aller übergeordneten Ordner ----------------------------------------------------------
    for item in ROOT_DIR.iterdir():
        if item.is_dir() and item.name.startswith(TARGET_PREFIXES):
            prefix = item.name[0]
            
            # ----------------------------- Durchsuche alle Ordner -----------------------------
            for parent, dirs, files in os.walk(item):
                timestamp_dirs = [d for d in dirs if is_timestamp_folder(d)]
                
                if timestamp_dirs:
                    timestamp_dirs.sort()
                    newest_dir = timestamp_dirs[-1]
                    dirs_to_delete = timestamp_dirs[:-1]
                    
                    if dirs_to_delete:
                        path_parent = Path(parent)
                        tasks.append({
                            'prefix': prefix,
                            'parent': path_parent,
                            'newest': newest_dir,
                            'to_delete': dirs_to_delete
                        })
                    
                    # ----------------------------- Optimierung -----------------------------
                    for d in timestamp_dirs:
                        if d in dirs:
                            dirs.remove(d)

    if not tasks:
        print("Keine zu bereinigenden Ordner gefunden.")
        return

    # ----------------------------- Output und Bestätigung -----------------------------
    print("\nFolgende Pfade werden bereinigt:\n")
    for task in tasks:
        # ----------------------------- Ausgabe der betroffenen Ordner ----------------------------------------------------------
        rel_path = task['parent'].relative_to(ROOT_DIR)
        count = len(task['to_delete'])
        print(f"{task['prefix']} > Pfad: {rel_path} -> ({count} Ordner werden gelöscht)")


    print("\n")
    print("\n")
    
    if os.environ.get('TIMESTAMP_CLEANUP_CONFIRMED') == '1':
        confirm = 'ja'
        print("Automatische Bestätigung durch PIPELINE-Steuerung.")
    else:
        confirm = input("Bestätigte das Löschen der neusten Zeitstempel Ordner aus folgenden Pfaden (ja/nein): ")
    
    if confirm.lower() == 'ja':
        print("\nStarte Bereinigung...\n")
        # ----------------------------- Ausführung der Löschung -----------------------------
        for task in tasks:
            parent = task['parent']
            print(f"Bearbeite: {parent.relative_to(ROOT_DIR)}")
            print(f"  Behalten: {task['newest']}")
            
            for d_name in task['to_delete']:
                dir_to_remove = parent / d_name
                print(f"  Lösche: {d_name}")
                try:
                    shutil.rmtree(dir_to_remove, onerror=on_rm_error)
                    print(f"    -> Gelöscht.")
                except Exception as e:
                    print(f"    -> Fehler: {e}")
            print("-" * 30)
            
        print("\nLöschen abgeschlossen.")
    else:
        print("Abbruch.")

if __name__ == "__main__":
    try:
        clean_timestamp_folders()
    except Exception:
        traceback.print_exc()
    
    input("\nDrücke [ENTER] um zu beenden...")
