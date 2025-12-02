# core_processing.py

"""
Central helper functions
-> already formatted .xlsx file
-> conversion to .csv following the global data schema
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd


# ---------------------- Molar masses (g/mol) – basic set ----------------------

MOLAR_MASS: Dict[str, float] = {
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
    "HS": 33.07,
}


# -------------------------------- helper functions --------------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Normalize column names:
    - trim
    - lower case
    - whitespaces -> underscore
    - reduce multiple underscores
    """
    
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("__+", "_", regex=True)
    )
    return df


def _available_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> Dict[str, str]:
   
    """
    For mapping only consider columns that actually exist
    in the DataFrame, ignore all others
    """
    return {src: tgt for src, tgt in mapping.items() if src in df.columns}


# -------------------------- remove whitespaces and N/D values ---------------------

def _clean_numeric(s: pd.Series) -> pd.Series:
    
    """
    Rough cleaning of numeric columns:
    - trims strings
    - removes 'N/D', 'N-D', etc. (case-insensitive)
    - removes comparison operators
    - converts to float (NaN on errors)
    """
    
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"(?i)\bN/?D\b", "", regex=True)
         .str.replace(r"[<>≈~]", "", regex=True)
         .replace({"": np.nan})
         .pipe(pd.to_numeric, errors="coerce")
    )


# ------------------------------- mg/L to mmol/L / µmol/L ------------------------------

def mgL_to_mmolL(series: pd.Series, M_gmol: float) -> pd.Series:
    return _clean_numeric(series) / M_gmol


def mgL_to_umolL(series: pd.Series, M_gmol: float) -> pd.Series:
    return mgL_to_mmolL(series, M_gmol) * 1000.0


# ---------------------------------- read & map Excel sheet ---------------------------------
def read_and_map(
    xls: pd.ExcelFile,
    sheet_name: str,
    mapping: Dict[str, str],
    usecols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    
    """
    Reads a single worksheet from an Excel file,
    normalizes the column names and maps them to the target schema.

    Parameters:
    # xls : pd.ExcelFile | already opened Excel file.
    # sheet_name : str | name of the worksheet.
    # mapping : Dict[str, str] | dict: {source_column_name_normalized: target_column_name}
    # keys must already correspond to the normalized form.

    Returns:
    pd.DataFrame | DataFrame with renamed columns (only the mapped ones).
    """
    
    if sheet_name not in xls.sheet_names:
        # empty DF with 0 rows
        return pd.DataFrame()

    df = pd.read_excel(xls, sheet_name=sheet_name, usecols=usecols)
    df = _normalize_columns(df)

    # error handling for typo in "dissolved oxygen" (occurred more often in practice)
    if "dissolved_oxygen_mg_per_l" in df.columns and "dissolved_oxigen_mg_per_l" not in df.columns:
        df = df.rename(columns={"dissolved_oxygen_mg_per_l": "dissolved_oxigen_mg_per_l"})

    ren = _available_mapping(df, mapping)
    if not ren:
        return pd.DataFrame()

    df = df.rename(columns=ren)
    return df[list(ren.values())]


# -------------------------- orchestration: Excel -> CSV --------------------------

def process_excel_to_csv(
    input_xlsx: Union[str, Path],
    output_csv: Union[str, Path],
    sheet_name: str,
    mapping: Dict[str, str],
    usecols: Optional[Iterable[str]] = None,
    sep: str = ",",
    encoding: str = "utf-8-sig",
) -> Path:
    
    base_dir = Path(__file__).resolve().parent

    input_xlsx = Path(input_xlsx)
    if not input_xlsx.is_absolute():
        input_xlsx = base_dir / "Input_formatted_xlsx" / input_xlsx

    output_csv = Path(output_csv)
    if not output_csv.is_absolute():
        output_csv = base_dir / "Output_cleaned_csv" / output_csv

    xls = pd.ExcelFile(input_xlsx)
    df = read_and_map(xls, sheet_name=sheet_name, mapping=mapping, usecols=usecols)

    # here is a place to implement further processing steps

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, sep=sep, encoding=encoding)
    return output_csv
