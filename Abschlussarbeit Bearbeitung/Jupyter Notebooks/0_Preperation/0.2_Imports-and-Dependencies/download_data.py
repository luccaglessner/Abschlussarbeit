import os
import urllib.request
from pathlib import Path

# ----------------------------------------------- Basisverzeichnisse und Zielordner -----------------------------------------------
# Dieses Skript lädt nur die benötigten Datenbanken von Google Drive herunter,
# ohne dass externe Abhängigkeiten wie gdown erforderlich sind.

DEST_DIR = Path(__file__).resolve().parent.parent.parent / "1_Acquisition" / "1.1_Data-Acquisition-Wrapper" / "Gesammelte_Datenbanken"

# ----------------------------------------------- Google Drive File IDs -----------------------------------------------
# Mapping der benötigten Dateien zu ihren öffentlichen Google Drive IDs.
FILES_TO_DOWNLOAD = {
    "Argonne_Geothermal_Geochemical_Database_v2_00.xlsx": "1zjjnyLG7aZ_8CmKX9gQdCR4WCgvQRKC0",
    "Bayerisches_Landesamt_fur_Umwelt.xlsx": "1opKWdjEqenLXnmkufBYIsJqYGwc4f7On",
    "Geotis_DB_Aquifer_03052024.xlsx": "1g5L7qWwtnXUSR2ZjRuR1T8pmOBxTd-gd",
    "Hessisches_Landesamt_fur_Naturschutz_Umwelt_und_Geologie.xlsx": "1BO8brafhpHvB7FlnzkkfZFQw2N-HCe4p",
    "HYDROCHEMISTRY_OF_VARIOUS_AQUIFERS_GERMANY.xlsx": "178-RSm1C3deIhyUJmLf7CLV_7iK0oAsC",
    "Hydrochemistry_Deep_Groundwaters.xlsx": "1hJzBKoh_SyOR34uoDt2LUI9ADHFyFVDk",
    "Landesamt_fur_Bergbau_Energie_und_Geologie_Niedersachsen.xlsx": "1qvfqPmPGXCBSc1l8Od5qGoJ1Yh30B5hd",
    "Landesamt_fur_Bergbau_Geologie_und_Rohstoffe_Brandenburg.xlsx": "1CKxXWCw7I87MPChz4njqJjHd6QJboyFO",
    "Landesamt_fur_Geologie_und_Bergwesen_Sachsen-Anhalt.xlsx": "1O6DXjiEVjas2ETD6qGaC4KVuCoU0xnf6",
    "Landesamt_fur_Umwelt_Rheinland_Pfalz.xlsx": "1e970lXynjIq8_VwHI3KVUbMNssQWDzzf",
    "Mucke.xlsx": "1eXaCNjPX4i6-1WgBJd_V3VOMlqNPLl5o",
    "REFLECT_Horizon_2020-Data_Export_2025-11-05_20-59-36.xlsx": "1r6xxr0yV8kX2hHlpOsiSvD-EjD_cLNP-",
    "Staatliches_Geologisches_Institut_Bremen.xlsx": "1Wvftwt7jX1_VN9eX5OM32gWAAgsaxTIA",
}

def download_file(url, destination_path):
    print(f"Lade herunter: {destination_path.name}...")
    try:
        # User-Agent wird manchmal benötigt, um 403 Forbidden bei Google Drive Links zu vermeiden
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(destination_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"✅ Erfolg: {destination_path.name}")
        return True
    except Exception as e:
        print(f"❌ Fehler beim Download von {destination_path.name}: {e}")
        return False

def main():
    print(f"\n{'='*60}")
    print("   Automatische Überprüfung der Quelldaten (Google Drive)")
    print(f"{'='*60}\n")
    
    # ----------------------------------------------- Zielverzeichnis erstellen -----------------------------------------------
    if not DEST_DIR.exists():
        DEST_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Verzeichnis erstellt: {DEST_DIR}")

    all_success = True
    
    # ----------------------------------------------- Download-Schleife -----------------------------------------------
    for filename, file_id in FILES_TO_DOWNLOAD.items():
        file_path = DEST_DIR / filename
        
        # ----------------------------------------------- Nur herunterladen, wenn Datei nicht vorhanden -----------------------------------------------
        if file_path.exists():
            print(f"✅ Datei bereits vorhanden: {filename} (Download übersprungen)")
        else:
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            success = download_file(download_url, file_path)
            if not success:
                all_success = False

    print(f"\n{'='*60}")
    if all_success:
        print("Alle benötigten Datenbanken sind lokal verfügbar.")
    else:
        print("⚠️ Warnung: Einige Dateien konnten nicht heruntergeladen werden.")
        print("Prüfe die Fehlermeldungen oben oder lade die Dateien manuell herunter.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

