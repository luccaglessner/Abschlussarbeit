import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import glob
import numpy as np

# ----------------------------------------------- Plot Styles setzen -----------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# ----------------------------------------------- Pfade definieren -----------------------------------------------
BASE_DIR = Path(__file__).parent.parent.resolve()
POSTER_ASSETS_DIR = BASE_DIR.parent / "Posterpräsentation/assets"
DATA_DIR = BASE_DIR / "1_Acquisition/1.1_Data-Acquisition-Wrapper/Angepasste_Datenbanken/Komplette_Datenbank"

# ----------------------------------- Assets Ordner erstellen falls nicht vorhanden -----------------------------------
POSTER_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def get_latest_database():
    # ------------------ Spezifische Rohdatenbank vom 27.12. nutzen (Große Datei, unbereinigt) ------------------
    target_folder = DATA_DIR / "2025-12-27_16-32-50"
    db_path = target_folder / "Komplette_Datenbank.csv"
    
    if not db_path.exists():
        # ------------------- Fallback auf aktuellste, falls spezifische nicht gefunden (Warnung) -------------------
        print("Spezifische Rohdatenbank nicht gefunden, nutze aktuellste (evtl. gefiltert).")
        timestamp_folders = [f for f in DATA_DIR.iterdir() if f.is_dir()]
        target_folder = max(timestamp_folders, key=lambda f: f.stat().st_mtime)
        db_path = target_folder / "Komplette_Datenbank.csv"
        
    print(f"Lade Datenbank von: {db_path}")
    
    # --------------------------------------- Benötigte Spalten für Analyse ---------------------------------------
    needed_cols = [
        "temperature_in_c",
        "Na_in_mmol/L", "Ca_in_mmol/L", "Mg_in_mmol/L", 
        "Cl_in_mmol/L", "HCO3_in_mmol/L", "SO4_in_mmol/L"
    ]
    
    # ------------------------------- Header lesen um korrekte Spaltennamen zu finden -------------------------------
    header = pd.read_csv(db_path, nrows=0).columns.tolist()
    
    # --------------------------------------- Nur vorhandene Spalten laden ---------------------------------------
    use_cols = [c for c in header if any(n in c for n in ["temperature", "Na", "Ca", "Mg", "Cl", "HCO3", "SO4"])]
    
    print(f"Lese Untermenge an Spalten: {len(use_cols)} Spalten gewählt.")
    
    # --------------------------------------- Alle Zeilen laden ---------------------------------------
    print("Lade gesamten Datensatz...")
    return pd.read_csv(db_path, usecols=use_cols, low_memory=False)

def plot_temperature_histogram(df):
    # ----------------------------- Generiert ein Temperatur-Histogramm -----------------------------
    print("Generiere Temperatur Histogramm...")
    
    # ----------------------------- Filter auf realistische Temperaturen (0 bis 500) -----------------------------
    raw_vals = df['temperature_in_c'].astype(str).str.replace(',', '.', regex=False)
    temp_numeric = pd.to_numeric(raw_vals, errors='coerce')
    
    # Debug: Wie viele haben überhaupt Werte?
    valid_vals_count = temp_numeric.notna().sum()
    print(f"DEBUG: Anzahl Zeilen mit gültiger Temperatur (vor Filter): {valid_vals_count}")
    
    temp_data = temp_numeric[(temp_numeric >= 0) & (temp_numeric <= 500)]
    
    # -------------------------------- Statistiken berechnen --------------------------------
    total_count = len(temp_data)
    below_10 = len(temp_data[temp_data < 10])
    above_10 = len(temp_data[temp_data >= 10])
    
    plt.figure(figsize=(8, 6))
    
    # ----------------------------- Plotten (Geteiltes Histogramm) -----------------------------
    data_below = temp_data[temp_data < 10]
    data_above = temp_data[temp_data >= 10]
    bins = np.linspace(0, 500, 101) # Linear bis 500

    
    sns.histplot(data_below, bins=bins, color='#8d94a7', alpha=0.8, edgecolor='#555', linewidth=0.5, label='< 10°C')
    sns.histplot(data_above, bins=bins, color='#022541', alpha=1.0, edgecolor='#01182a', linewidth=0.5, label='≥ 10°C')
    
    plt.title('Globale Grundwassertemperatur Verteilung', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Temperatur [°C]', fontsize=14)
    plt.ylabel('Anzahl (Log-Skala)', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, which="both")
    
    # --------------------------------------- Mittelwert Linie hinzufügen ---------------------------------------
    mean_val = temp_data.mean()
    plt.axvline(mean_val, color='#e67e22', linestyle='--', linewidth=2, label=f'Mittelwert: {mean_val:.1f}°C')

    # --------------------------------------- 10°C Trennlinie hinzufügen ---------------------------------------
    plt.axvline(10, color='red', linestyle='-', linewidth=2, label='10°C Grenze')
    
    # --------------------------------------- Statistik Text Box hinzufügen ---------------------------------------
    # Farben anpassen / Position
    stats_text = (
        f"Analysiert Gesamt: {total_count:,}\n"
        f"≥ 10°C: {above_10:,} ({above_10/total_count:.1%})\n"
        f"< 10°C: {below_10:,} ({below_10/total_count:.1%})"
    )
    plt.text(
        0.98, 0.60, stats_text, 
        transform=plt.gca().transAxes, 
        fontsize=11, 
        verticalalignment='top', 
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#ccc')
    )
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    output_path = POSTER_ASSETS_DIR / "temp_histogram.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gespeichert: {output_path}")
    plt.close()

def plot_ibe_histogram(df):
    # ----------------------------- Generiert ein schematisches IBE Histogramm -----------------------------
    print("Generiere IBE Histogramm...")
    
    # ----------------------------- Ziel-Ionen definieren (Main-Ions-Six) -----------------------------
    ions = {
        'Na': 1, 'Ca': 2, 'Mg': 2, # Kationen
        'Cl': 1, 'HCO3': 1, 'SO4': 2 # Anionen
    }
    
    # --------------------------------------- Spalten identifizieren ---------------------------------------
    found = {}
    for ion in ions:
        matches = [c for c in df.columns if c == ion] # Exakter Match
        if not matches:
             matches = [c for c in df.columns if f"{ion}_in_" in c] # Fallback
        
        if matches:
            found[ion] = matches[0]

    if len(found) < 6:
        print("Nicht genügend Ionen-Spalten gefunden. Verwende Dummy-Daten.")
        return 
    
    # --------------------------------------- Berechnung der Summen ---------------------------------------
    cat_sum = 0
    an_sum = 0
    
    for ion, col in found.items():
        val = pd.to_numeric(df[col], errors='coerce').fillna(0)
        charge = ions[ion]
        if ion in ['Na', 'Ca', 'Mg']:
            cat_sum += val * charge
        else:
            an_sum += val * charge
            
    total = cat_sum + an_sum
    ibe = ((cat_sum - an_sum) / total) * 100
    
    # --------------------------- Filter auf Validität (Summe > 0.1 und endliche Werte) ---------------------------
    valid_mask = (total > 0.1) & np.isfinite(ibe)
    valid_ibe = ibe[valid_mask]
    
    display_range = 100
    
    # -------------------------------- Statistiken berechnen auf ALLEN validen Daten --------------------------------
    total_count = len(valid_ibe)
    good_count = len(valid_ibe[(valid_ibe >= -5) & (valid_ibe <= 5)])
    bad_count = total_count - good_count
    
    # ------------------------------------------- Outlier Analyse -------------------------------------------
    outliers = valid_ibe[(valid_ibe < -5) | (valid_ibe > 5)]
    # Extreme Outliers entfernt, da per Definition bei +/- 100 Schluss ist
    
    # --------------------------------------- Daten für Plot filtern ---------------------------------------
    data = valid_ibe[(valid_ibe >= -display_range) & (valid_ibe <= display_range)]
    print(f"Statistik: Total={total_count}, Gut={good_count}, Schlecht={bad_count}")

    plt.figure(figsize=(8, 6))
    
    # ------------------------------------------ Daten aufteilen ------------------------------------------
    good_data = data[(data >= -5) & (data <= 5)]
    outlier_data = data[(data < -5) | (data > 5)] 
    
    # --------------------------------------- Shared Bins definieren ---------------------------------------
    bins = np.linspace(-display_range, display_range, 101)

    # ---------------------------------------------- Plotten ----------------------------------------------
    # Outliers (Lighter Blue/Grey: #8d94a7)
    sns.histplot(outlier_data, bins=bins, color='#8d94a7', element='bars', alpha=0.8, edgecolor='#555', linewidth=0.5, label='Ausreißer (> 5%)')
    # Good Data (Dark Blue: #022541)
    sns.histplot(good_data, bins=bins, color='#022541', element='bars', alpha=1.0, edgecolor='#01182a', linewidth=0.5, label='Gute Daten (±5%)')
    
    # --------------------------------------- +/- 5% Bereich hervorheben ---------------------------------------
    plt.axvspan(-5, 5, color='#022541', alpha=0.05)
    
    # ----------------------------------------- Logarithmische Skala -----------------------------------------
    plt.yscale('log')
    
    # --------------------------------------- Statistik Text Box hinzufügen ---------------------------------------
    stats_text = (
        f"Analysiert Gesamt: {total_count:,}\n"
        f"Gute Daten: {good_count:,} ({good_count/total_count:.1%})\n"
        f"Ausreißer: {bad_count:,} ({bad_count/total_count:.1%})"
    )
    plt.text(
        0.98, 0.75, stats_text, 
        transform=plt.gca().transAxes, 
        fontsize=11, 
        verticalalignment='top', 
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#ccc')
    )
    
    # --------------------------------------- Formel hinzufügen ---------------------------------------
    formula_text = r"$IBE = \frac{\sum Kationen - \sum Anionen}{\sum Kationen + \sum Anionen} \cdot 100$"
    plt.text(
        0.02, 0.98, formula_text,  
        transform=plt.gca().transAxes, 
        fontsize=12, 
        verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9, edgecolor='#ccc')
    )
    
    plt.title('Ionenbilanzfehler (IBF)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Ionenbilanzfehler [%]', fontsize=14)
    plt.ylabel('Anzahl [Log-Skala]', fontsize=14)
    plt.xlim(-display_range, display_range)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    output_path = POSTER_ASSETS_DIR / "ibe_histogram.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gespeichert: {output_path}")
    plt.close()

def plot_architecture_diagram():
    # ----------------------------- Generiert ein schematisches Architektur-Diagramm -----------------------------
    print("Generiere Architektur Diagramm...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # ----------------------------------------- Styles definieren -----------------------------------------
    box_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#0d2b45', linewidth=2)
    wrapper_props = dict(boxstyle='round,pad=0.3', facecolor='#e1f5fe', edgecolor='#0277bd', linewidth=1.5)
    mediator_props = dict(boxstyle='round,pad=0.5', facecolor='#fff3e0', edgecolor='#ef6c00', linewidth=2)
    
    # -------------------------------- Level 1: Heterogene Quellen (Links) --------------------------------
    sources = ["USGS\n(USA)", "HESSEN\n(DE)", "ALPS\n(EU)", "YUKON\n(CA)"]
    y_pos = [5, 4, 3, 2]
    
    for i, source in enumerate(sources):
        ax.text(1.5, y_pos[i], source, ha='center', va='center', bbox=box_props, fontsize=10)
        
    ax.text(1.5, 5.8, "1. Heterogene Quellen", ha='center', va='center', fontweight='bold', color='#333')

    # ------------------------------------ Level 2: Wrapper (Mitte-Links) ------------------------------------
    for y in y_pos:
        ax.text(4, y, "Wrapper", ha='center', va='center', bbox=wrapper_props, fontsize=9, style='italic')
        ax.annotate("", xy=(3.5, y), xytext=(2.2, y), arrowprops=dict(arrowstyle="->", color='#555'))

    ax.text(4, 5.8, "2. Transformation", ha='center', va='center', fontweight='bold', color='#333')

    # ------------------------------ Level 3: Globales Schema / Mediator (Mitte-Rechts) ------------------------------
    ax.text(7, 3.5, "Globales Schema\n(Mediator)\n\n30 Attribute\nEinheitliche Units", 
            ha='center', va='center', bbox=mediator_props, fontsize=11, fontweight='bold')
    
    for y in y_pos:
        ax.annotate("", xy=(6, 3.5), xytext=(4.5, y), arrowprops=dict(arrowstyle="->", color='#555', lw=1))

    # ---------------------------------- Level 4: Einheitliche Datenbank (Rechts) ----------------------------------
    ax.text(9.5, 3.5, "Validierte\nDatenbasis\n(CSV)", ha='center', va='center', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f5e9', edgecolor='#2e7d32'), fontsize=10)
    
    ax.annotate("", xy=(8.8, 3.5), xytext=(8, 3.5), arrowprops=dict(arrowstyle="->", color='#2e7d32', lw=2))

    plt.tight_layout()
    output_path = POSTER_ASSETS_DIR / "architecture.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gespeichert: {output_path}")
    plt.close()

if __name__ == "__main__":
    try:
        df = get_latest_database()
        plot_temperature_histogram(df)
        plot_ibe_histogram(df)
        plot_architecture_diagram()
        print("Assets erfolgreich generiert.")
    except Exception as e:
        print(f"Fehler: {e}")
