import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------------------------- Pfade definieren -----------------------------------------------
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "1.1_Data-Acquisition-Wrapper/Angepasste_Datenbanken/Komplette_Datenbank"

def get_latest_database():
    # ------------------ Spezifische Rohdatenbank vom 27.12. nutzen (Große Datei, unbereinigt) ------------------
    target_folder = DATA_DIR / "2025-12-27_16-32-50"
    db_path = target_folder / "Komplette_Datenbank.csv"
    print(f"Lade Datenbank: {db_path}")
    
    needed_cols = ["Na_in_mmol/L", "K_in_mmol/L", "Ca_in_mmol/L", "Mg_in_mmol/L", 
                   "Cl_in_mmol/L", "HCO3_in_mmol/L", "SO4_in_mmol/L", "NO3_in_mmol/L"]
    
    # -------------------------------- Header lesen um Spalten zu finden --------------------------------
    header = pd.read_csv(db_path, nrows=0).columns.tolist()
    use_cols = [c for c in header if any(n in c for n in ["Na", "K", "Ca", "Mg", "Cl", "HCO3", "SO4", "NO3"])]
    
    return pd.read_csv(db_path, usecols=use_cols, low_memory=False)

try:
    df = get_latest_database()
    
    # --------------------------- Logik aus generate_poster_assets.py (Main-Ions-Six) ---------------------------
    # ------------------------------- Entferne K (1) und NO3 (1) aus der Berechnung ------------------------------
    ions = {'Na': 1, 'Ca': 2, 'Mg': 2, 'Cl': 1, 'HCO3': 1, 'SO4': 2}
    
    found = {}
    for ion in ions:
        matches = [c for c in df.columns if c == ion]
        if not matches: matches = [c for c in df.columns if f"{ion}_in_" in c]
        if matches: found[ion] = matches[0]
            
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
    
    valid_mask = (total > 0.1) & np.isfinite(ibe)
    valid_ibe = ibe[valid_mask]
    
    # --------------------------------------- BEISPIEL EXTRAKTION ---------------------------------------
    print(f"\n------------------------------ BEISPIELE FÜR EXTREMWERTE ------------------------------")
    
    # --- IBE zum DataFrame hinzufügen für einfacheres Filtern ---
    df['IBE'] = ibe
    df_valid = df[valid_mask].copy()
    
    # -------------------------- Beispiele +100% (Erwarte fehlende Anionen) --------------------------
    pos_examples = df_valid[df_valid['IBE'] > 99.9].head(3)
    print(f"\n[Beispiele: +100% IBE] (Erwarte fehlende Anionen)")
    cols_to_show = ['IBE'] + list(found.values())
    print(pos_examples[cols_to_show].to_string(index=False))
    
    # -------------------------- Beispiele -100% (Erwarte fehlende Kationen) --------------------------
    neg_examples = df_valid[df_valid['IBE'] < -99.9].head(3)
    print(f"\n[Beispiele: -100% IBE] (Erwarte fehlende Kationen)")
    print(neg_examples[cols_to_show].to_string(index=False))

except Exception as e:
    print(e)
