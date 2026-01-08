import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# ----------------------------------------------- Konfiguration -----------------------------------------------
# Log to parent directory (Jupyter Notebooks root)
LOG_DIR = Path(__file__).parent.parent / "Crash_Reports"

def setup_logging():
    """Stellt sicher, dass das Protokollverzeichnis existiert."""
    if not LOG_DIR.exists():
        LOG_DIR.mkdir()

def log_error(context, exception=None, stderr=None):
    """
    Dokumentiert einen kritischen Fehler in einer separaten Textdatei.
    
    Parameter:
    - context (str): Beschreibung des Kontextes (z.B. Dateiname, Schritt).
    - exception (Exception, optional): Das abgefangene Python-Exception-Objekt.
    - stderr (str, optional): Standard Error Output eines Subprozesses.
    """
    setup_logging()
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = LOG_DIR / f"Crash_Report_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("                            FEHLERPROTOKOLL (CRASH REPORT)\n")
        f.write("================================================================================\n\n")
        f.write(f"Zeitstempel:       {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
        f.write(f"Kontext:           {context}\n")
        f.write(f"Ausgeführtes Skript: {sys.argv[0]}\n\n")
        
        f.write("--------------------------------------------------------------------------------\n")
        f.write("FEHLERBESCHREIBUNG\n")
        f.write("--------------------------------------------------------------------------------\n")
        
        if exception:
            f.write(f"Typ: {type(exception).__name__}\n")
            f.write(f"Meldung: {str(exception)}\n\n")
            f.write("Traceback:\n")
            traceback.print_tb(exception.__traceback__, file=f)
        
        if stderr:
            f.write("\nSUBPROZESS STDERR (Standardfehlerausgabe):\n")
            f.write(stderr)
            f.write("\n")
            
        if not exception and not stderr:
            f.write("Keine spezifischen Fehlerdetails verfügbar.\n")
            
        f.write("\n================================================================================\n")
        f.write("Dieses Dokument dient der technischen Fehleranalyse und Reproduktion.\n")
        f.write("Ende des Protokolls.\n")
        
    print(f"\n[!] KRITISCHER FEHLER. Details wurden gespeichert in:\n    {filename.resolve()}\n")
