import subprocess
import sys
import os

# --------------------------------------------------------------------------------
# Liste der zu installierenden Pakete.
# Diese wird basierend auf Fehlermeldungen erweitert.
# --------------------------------------------------------------------------------

REQUIRED_PACKAGES = [
    # ------------------------- Datenanalyse & Mathe -------------------------
    "pandas",
    "numpy",
    
    # ------------------------- Visualisierung -------------------------
    "matplotlib",
    "seaborn",
    
    # ------------------------- Machine Learning -------------------------
    "scikit-learn",
    "torch",
    "minisom", # From project context (though not in grep, it's a key part of the project)
    
    # ------------------------- Jupyter / Reporting -------------------------
    "jupyter",
    "nbconvert",
    "reportlab",
    "openpyxl",
    "pyproj",
    "xlsxwriter",
]

def install(package):
    print(f"Installiere: {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"-> {package} erfolgreich installiert.")
    except subprocess.CalledProcessError:
        print(f"-> FEHLER bei der Installation von {package}.")

def main():
    print("=== Starte Installation fehlender Abhängigkeiten ===\n")
    
    if not REQUIRED_PACKAGES:
        print("Keine Pakete in der Liste 'REQUIRED_PACKAGES' definiert.")
        print("Bitte füge fehlende Pakete im Skript hinzu.")
        return

    for package in REQUIRED_PACKAGES:
        install(package)
        
    print("\n=== Vorgang abgeschlossen ===")
    
    if os.environ.get("PIPELINE_BATCH_MODE") != "1":
        input("Drücke [ENTER] um zu beenden...")

if __name__ == "__main__":
    main()
