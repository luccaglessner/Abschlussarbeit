
# ----------------------- Attributes required for the Bachelor Thesis -----------------------
ATTRIBUTES = [
    "location",
    "well_or_spring_name",
    "WGS84_latitude",
    "WGS84_longitude",
    "depth_bgl_in_m",
    "rock_type",
    "stratigraphic_period",
    "temperature_in_c",
    "electrical_conductivity_25c_in_uS/cm",
    "pH",
    "redox_potential_in_mV",
    "total_dissolved_solids_in_mmol/L", 
    "O2_in_mmol/L",
    "Na_in_mmol/L",
    "Mg_in_mmol/L",
    "Ca_in_mmol/L",
    "Cl_in_mmol/L",
    "SO4_in_mmol/L",
    "HCO3_in_mmol/L",
    "Li_in_mmol/L",
    "K_in_mmol/L",
    "Sr_in_umol/L", # umol
    "NH4_in_umol/L", # umol
    "Fe_in_mmol/L",
    "Mn_in_mmol/L",
    "F_in_umol/L", # umol
    "NO3_in_mmol/L",
    "H2SiO3_in_umol/L", # umol
    "HS_in_mmol/L",
    "Database_number",
    "Database_name"
]


# ---------------------------------- 2. Column where suffixes are removed ----------------------------------
SUFFIXES = {
    "_in_mmol/L": "mmol/L",
    "_in_umol/L": "µmol/L",
    "_in_mV": "mV",
    "_in_m": "m",
    "_in_c": "°C",
    "_in_uS/cm": "µS/cm"
}


# ---------------------- Molar Masses (g/mol) according to NIST ----------------------
MOLAR_MASS = {
    "O2": 31.9988,
    "Na": 22.989769,
    "Mg": 24.305,
    "Ca": 40.078,
    "Cl": 35.45,
    "SO4": 96.06,
    "HCO3": 61.0168,
    "Li": 6.94,
    "K": 39.0983,
    "Sr": 87.62,
    "NH4": 18.038,
    "Fe": 55.845,
    "Mn": 54.938,
    "F": 18.998,
    "NO3": 62.0049,
    "H2SiO3": 78.11,
    "HS": 33.07
}
