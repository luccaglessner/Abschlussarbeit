import os
import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime

# ----------------------------------------------- Basisverzeichnis definieren -----------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()

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

def main():
    print("========================================================")
    print("   Starte Ausführung: Analysis and Learning Pipeline")
    print("========================================================")
    print(f"Basisverzeichnis: {BASE_DIR}\n")

    notebooks_to_run = [
        # 2. Analytik (Exploration)
        BASE_DIR / "2_Analysis/2.1_Explorative-Datenanalyse" / "Data_Exploration.ipynb",
        
        # 3. Analytik (Rock Type)
        BASE_DIR / "2_Analysis/2.2_Rock-Type_Analysis" / "Rock-Type_Analyzer.ipynb",
        
        # 4. Temperatur Analyse
        BASE_DIR / "2_Analysis/2.3_Temperature_Analysis" / "Temperature_Analysis.ipynb"
    ]

    # ----------------------------------------- Schritte 2-4 ausführen -----------------------------------------
    for i, nb_path in enumerate(notebooks_to_run, 2):
        if not nb_path.exists():
            print(f"WARNUNG: Datei nicht gefunden: {nb_path}")
            continue
            
        print(f"\n--- Schritt {i}: {nb_path.parent.name} ---")
        success = run_notebook(nb_path)
        if not success:
            print("Abbruch der Pipeline aufgrund von Fehlern.")
            input("Taste drücken zum Beenden...")
            sys.exit(1)

    # -------------------------------- Poster Assets Generieren (Raw Data) --------------------------------
    print(f"\n--- ZWISCHENSCHRITT: Poster Assets Generieren (Raw Data) ---")
    asset_script = BASE_DIR / "generate_poster_assets.py"
    if asset_script.exists():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starte: {asset_script.name}")
        try:
            cmd = [sys.executable, str(asset_script)]
            subprocess.run(cmd, check=True)
            print("  -> Assets erfolgreich erstellt.")
        except subprocess.CalledProcessError as e:
            print(f"  -> FEHLER beim Erstellen der Assets: {e}")
    else:
        print(f"Warnung: {asset_script.name} nicht gefunden.")

    # ----------------------------- Schritt 5: Data Quality (Iterieren über Unterordner) -----------------------------
    print(f"\n--- Schritt 5: 2_Analysis/2.4_Data-Quality_Ionic-Balance-Error ---")
    dq_root = BASE_DIR / "2_Analysis/2.4_Data-Quality_Ionic-Balance-Error"
    
    if not dq_root.exists():
        print(f"WARNUNG: Ordner nicht gefunden: {dq_root}")
    else:
        # ------------------------------------- Alle Unterordner finden -------------------------------------
        subfolders = [f for f in dq_root.iterdir() if f.is_dir()]
        print(f"Gefundene Varianten: {[f.name for f in subfolders]}")
        
        for sub in subfolders:
            print(f"\n  > Verarbeite Variante: {sub.name}")
            
            nb_a = sub / "1.1_Data-Quality_Raw_Data.ipynb"
            nb_b = sub / "1.2_Data-Quality_Adjusted_Data.ipynb"
            
            # -------------------------------- Notebooks ausführen wenn vorhanden --------------------------------
            if nb_a.exists():
                if not run_notebook(nb_a): 
                    print("Warnung: Fehler in 1.1, fahre dennoch fort...")
            
            if nb_b.exists():
                if not run_notebook(nb_b):
                    print("Warnung: Fehler in 1.2, fahre dennoch fort...")

    # ----------------------------- Schritt 6: Filter Database (Erzeugt neue DB Version) -----------------------------
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

    # ----------------------------------------- Schritt 7: Preprocessing -----------------------------------------
    print(f"\n--- Schritt 7: 3_Machine-Learning/3.1_Preprocessing ---")
    prep_nb = BASE_DIR / "3_Machine-Learning/3.1_Preprocessing" / "Preprocessing.ipynb"
    if prep_nb.exists():
        if not run_notebook(prep_nb):
            print("FEHLER im Preprocessing! Pipeline wird gestoppt.")
            input("Taste drücken zum Beenden...")
            sys.exit(1)
    else:
        print(f"Fehler: Preprocessing-Notebook nicht gefunden: {prep_nb}")
        sys.exit(1)

    # ----------------------------------------- Schritt 8: MiniSom -----------------------------------------
    print(f"\n--- Schritt 8: 3_Machine-Learning/3.2_Machine-Learning (MiniSom) ---")
    minisom_nb = BASE_DIR / "3_Machine-Learning/3.2_Machine-Learning" / "MiniSom" / "MiniSom_Machine-Learning.ipynb"
    if minisom_nb.exists():
        if not run_notebook(minisom_nb):
            print("WARNUNG: Fehler in MiniSom! (Pipeline läuft weiter)")
    else:
        print(f"Fehler: MiniSom-Notebook nicht gefunden: {minisom_nb}")

    # ----------------------------------------- Schritt 9: 5.1 Synthetic Data Generation -----------------------------------------
    print(f"\n--- Schritt 9: 5_Synthetic-Data/5.1_Synthetic-Data (VAE Generation) ---")
    vae_nb = BASE_DIR / "5_Synthetic-Data/5.1_Synthetic-Data" / "Variational-Autoencoder" / "VAE_Synthetic_Data_Generation.ipynb"
    if vae_nb.exists():
        if not run_notebook(vae_nb): 
            print("WARNUNG: Fehler bei VAE Generation! Pipeline wird gestoppt.")
            input("Taste drücken zum Beenden...")
            sys.exit(1)
    else:
        print(f"Fehler: VAE-Notebook nicht gefunden: {vae_nb}")

    # ----------------------------------------- Schritt 10: 5.2 Preprocessing (Synthetic) -----------------------------------------
    print(f"\n--- Schritt 10: 5_Synthetic-Data/5.2_Preprocessing (Synthetic) ---")
    prep_syn_nb = BASE_DIR / "5_Synthetic-Data/5.2_Preprocessing" / "Preprocessing.ipynb"
    if prep_syn_nb.exists():
        if not run_notebook(prep_syn_nb):
            print("FEHLER im Synthetic Preprocessing! Pipeline wird gestoppt.")
            input("Taste drücken zum Beenden...")
            sys.exit(1)
    else:
        print(f"Fehler: Preprocessing-Notebook (5.2) nicht gefunden: {prep_syn_nb}")

    # ----------------------------------------- Schritt 11: 5.3 Machine-Learning (MiniSom Synthetic) -----------------------------------------
    print(f"\n--- Schritt 11: 5_Synthetic-Data/5.3_Machine-Learning (MiniSom Synthetic) ---")
    minisom_syn_nb = BASE_DIR / "5_Synthetic-Data/5.3_Machine-Learning" / "MiniSom" / "MiniSom_Machine-Learning.ipynb"
    if minisom_syn_nb.exists():
        if not run_notebook(minisom_syn_nb):
            print("WARNUNG: Fehler in MiniSom (Synthetic)! (Pipeline läuft weiter)")
    else:
        print(f"Fehler: MiniSom-Notebook (5.3) nicht gefunden: {minisom_syn_nb}")

    print("\n========================================================")
    print("   Analysis and Learning Pipeline vollständig ausgeführt!")
    print("========================================================")
    input("Taste drücken zum Schließen...")

if __name__ == "__main__":
    main()
