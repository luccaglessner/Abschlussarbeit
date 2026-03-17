import os
import subprocess
import sys
import time

from pathlib import Path
from datetime import datetime


# ----------------------------------------------- Basisverzeichnis -----------------------------------------------
BASE_DIR = Path(__file__).parent.parent.resolve()

import pipeline_logger

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
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=notebook_path.parent)
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
    print("=== Pipeline 2 & 3: Analytik & Standard Machine Learning ===")
    print(f"Process ID: {os.getpid()}")
    # ----------------------------- Interaktive Abfrage: Modus für SOM -----------------------------
    if "SOM_MODE" in os.environ:
        print(f"\n[KONFIGURATION] SOM_MODE bereits gesetzt auf: {os.environ['SOM_MODE']}")
    else:
        print("\n[KONFIGURATION] Bitte wähle den Ausführungsmodus für Machine Learning (Schritt 3.2):")
        print("  1) MANUAL (Kombinationen: bis Report_010)")
        print("  2) AUTO / LOOP (Automatische Kombinationstestung)")
        print("  3) SIZE_ITERATIONS (2x2 bis 10x10)")
        print("  4) SINGLE_FEATURE_SELECTION (Index wählen)")
        choice = input("Deine Wahl (1/2/3/4): ").strip()
        
        if choice == '1':
            os.environ['SOM_MODE'] = 'MANUAL'
            os.environ['SOM_REGION'] = input("\nSoll zusätzlich eine regionale Lokalisierung (z.B. Oberrheingraben) durchgeführt werden? (j/n): ").strip().lower()
            if os.environ['SOM_REGION'] in ['j', 'ja', 'y']:
                os.environ['SOM_REGION'] = 'OBERRHEINGRABEN'
                custom_coords = input("Standard-Koordinaten (Oberrheingraben) nutzen oder manuell eingeben? (s/m): ").strip().lower()
                if custom_coords == 'm':
                    os.environ['LOC_REGION_NAME'] = input("Name der Region: ").strip()
                    os.environ['LOC_LAT_MIN'] = input("Latitude Min: ").strip()
                    os.environ['LOC_LAT_MAX'] = input("Latitude Max: ").strip()
                    os.environ['LOC_LON_MIN'] = input("Longitude Min: ").strip()
                    os.environ['LOC_LON_MAX'] = input("Longitude Max: ").strip()
            print(">> Modus gesetzt: MANUAL\n")
        elif choice == '2':
            os.environ['SOM_MODE'] = 'LOOP'
            print(">> Modus gesetzt: LOOP (Auto)\n")
        elif choice == '3':
            os.environ['SOM_MODE'] = 'SIZE_ITERATIONS'
            print(">> Modus gesetzt: SIZE_ITERATIONS\n")
        elif choice == '4':
            os.environ['SOM_MODE'] = 'MANUAL'
            # ----------------------------- Feature Preview & Selection -----------------------------
            print("\n[INFO] Lade Datenvorschau für Feature-Auswahl...")
            try:
                import pandas as pd
                # Pfad zum aktuellsten Preprocessing finden (analog zu Location_Analysis)
                prep_root = BASE_DIR / "3_Machine-Learning/3.1_Preprocessing/Preprocessing"
                if prep_root.exists():
                    timestamp_folders = [f for f in prep_root.iterdir() if f.is_dir()]
                    if timestamp_folders:
                        latest_folder = max(timestamp_folders, key=lambda f: f.stat().st_mtime)
                        csv_path = latest_folder / "Preprocessed_SOM_Ready.csv"
                        if csv_path.exists():
                            df = pd.read_csv(csv_path, low_memory=False)
                            
                            # Standard-Kombinationen (Hardcoded für Vorschau, passend zum Notebook)
                            runs = [
                                {"name": "Test-Run",        "add": []},
                                {"name": "Base_with_pH",    "add": ["pH"]},
                                {"name": "Plus_pH-Fe",      "add": ["pH", "Fe_in_mmol/L"]},
                                {"name": "Plus_pH-K-Fe",    "add": ["pH", "K_in_mmol/L", "Fe_in_mmol/L"]},
                                {"name": "Plus_pH-K-Fe-Mn", "add": ["pH", "K_in_mmol/L", "Fe_in_mmol/L", "Mn_in_mmol/L"]},
                                {"name": "Plus_temperature", "add": ["temperature_in_c"]}
                            ]
                            
                            print("\nVerfügbare Feature-Kombinationen:")
                            print(f"{'Index':<6} | {'Name':<20} | {'Samples':<10}")
                            print("-" * 45)
                            for i, r in enumerate(runs):
                                # Einfache Schätzung der Samples (Variablen im Notebook nachbauen)
                                # Da die t_cols komplex gemappt werden, machen wir hier einen schnellen Count
                                base_cols = ["Na_in_mmol/L", "Mg_in_mmol/L", "Ca_in_mmol/L", "Cl_in_mmol/L", "SO4_in_mmol/L", "HCO3_in_mmol/L"]
                                check_cols = base_cols + r["add"]
                                # Suche echte Spaltennamen (gauss etc.)
                                real_cols = []
                                for c in check_cols:
                                    matches = [col for col in df.columns if col.startswith(c)]
                                    if matches: real_cols.append(matches[0])
                                
                                count = df[real_cols].dropna().shape[0] if real_cols else 0
                                print(f"{i:<6} | {r['name']:<20} | {count:<10}")
                            print("-" * 45)
                            
                            target = input("\nBitte Index wählen (0-5) [0]: ").strip()
                            if target == "": target = "0"
                            
                            # Realer Index im Notebook: 0=Test-Run, 1=Separator, 2...
                            # Pipeline 4 Logik:
                            real_idx = int(target)
                            if real_idx > 0: real_idx += 1 # Überspringe ersten SEPARATOR
                            # Im MiniSom Notebook sind die ersten 11 Einträge (inkl. Separatoren) für Manual relevant.
                            
                            os.environ['SOM_TARGET_INDEX'] = str(real_idx)
                            print(f">> Ziel-Index gesetzt: {target} (Intern: {real_idx})\n")
                        else:
                            print("Fehler: Preprocessing CSV nicht gefunden.")
                    else:
                        print("Fehler: Kein Preprocessing-Timestamp gefunden.")
                else:
                    print("Fehler: Preprocessing-Ordner nicht gefunden.")
            except ImportError:
                print("Fehler: pandas wird für die Vorschau benötigt.")
            except Exception as e:
                print(f"Fehler bei Feature-Vorschau: {e}")
        else:
            os.environ['SOM_MODE'] = 'MANUAL'
            print(">> Modus gesetzt: MANUAL\n")

    # --------- deutscher kommentar ---------
    # Abfrage für Oberrheingraben-Filter
    # ---------------------------------------
    if "SOM_REGION" not in os.environ:
        print("[KONFIGURATION] Soll zusätzlich ein Bericht für den Oberrheingraben erstellt werden?")
        choice_region = input("Deine Wahl (j/n): ").strip().lower()
        if choice_region in ['j', 'ja', 'y', 'yes']:
            os.environ['SOM_REGION'] = 'OBERRHEINGRABEN'
            print(">> Region gesetzt: OBERRHEINGRABEN\n")
        else:
            os.environ['SOM_REGION'] = 'ALL'
            print(">> Region gesetzt: ALL (Nur Hauptbericht)\n")

    
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
        # ---------------------------------------------------------------------------------
        # Die Unterordner (Global_scheme_ions, Main-Ions-Six, etc.) dienen nur der Dokumentation
        # und generieren PDFs. Sie verändern die Daten nicht und werden ab hier übersprungen.
        # ---------------------------------------------------------------------------------
        
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

    # ----------------------------- 3.3 Lokalisierung (Neu) -----------------------------
    if os.environ.get('SOM_REGION') == 'OBERRHEINGRABEN':
        loc_nb = BASE_DIR / "3_Machine-Learning/3.3_Lokalisieren/Location_Analysis.ipynb"
        if loc_nb.exists():
            print("\n[INFO] Starte regionale Lokalisierung (Oberrheingraben)...")
            run_notebook(loc_nb)
        else:
            print(f"Warnung: {loc_nb} nicht gefunden.")

    print("\n--- Pipeline 2 & 3 abgeschlossen ---")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Die Pipeline wurde unerwartet beendet:\n{e}")
    finally:
        print("\n" + "="*60)
        pass
