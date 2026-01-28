import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import schema_config

class GeothermalDataPipeline:
    def __init__(self, input_path: Union[str, Path], output_path: Union[str, Path], column_mapping: Dict[str, str]):
        
        # ----------------------------------------------------------------------------------------------
        """
        Initializes the pipeline with input/output paths and a column mapping.
        
        Args:
            input_path: Path to the input .xlsx file.
            output_path: Path to the output directory or file.
            column_mapping: Dictionary mapping raw column names to target schema names.
        """
        # ----------------------------------------------------------------------------------------------

        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.column_mapping = column_mapping
        
        # Load schema from config
        self.attributes = schema_config.ATTRIBUTES
        self.suffixes = schema_config.SUFFIXES
        self.molar_masses = schema_config.MOLAR_MASS

        self.df = None

    def run(self):
        # ------------------------------- Executes the full pipeline -------------------------------
        print(f"--- Starting Pipeline for {self.input_path.name} ---")
        
        self.load_data()
        self.normalize_input_columns()
        self.apply_mapping()
        self.convert_units()
        self.validate_schema()
        self.save_data()
        
        print("--- Pipeline Completed Successfully ---")
        return self.df

    def load_data(self, sheet_name: Union[str, int] = 0):
        # ------------------------------- Loads data from the input Excel file -------------------------------
        print(f"Loading data from: {self.input_path}")
        # ------------------------------- Using openpyxl explicitly as engine -------------------------------
        self.df = pd.read_excel(self.input_path, sheet_name=sheet_name)
        print(f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns.")

    def normalize_input_columns(self):

        # ----------------------------------------------------------------------------------------------
        """
        Normalizes the input dataframe columns to be lower case, stripped, and underscore-separated.
        This makes mapping easier (e.g., matching "  Temp (C) " to "temp_c").
        """
        # ----------------------------------------------------------------------------------------------

        if self.df is None: return
        
        # ------------------------------- Determine the normalization map to help the user debug mapping issues if needed -------------------------------
        self.df.columns = (
            self.df.columns
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace("__+", "_", regex=True)
        )
        print("Normalized input columns.")

    def apply_mapping(self):
        # ------------------------------- Renames columns based on the provided mapping -------------------------------
        if self.df is None: return

        # ------------------------------- Normalize the keys in the mapping to match the normalized input columns -------------------------------
        normalized_mapping = {}
        for k, v in self.column_mapping.items():
            norm_k = k.strip().lower().replace(" ", "_").replace("__", "_") 

            norm_key = k.strip().lower().replace(" ", "_").replace(r"\s+", "_")
            # ------------------------------- This regex replacement in python string need re library or use basic string methods -------------------------------
            import re
            norm_key = re.sub(r"\s+", "_", norm_key)
            norm_key = re.sub(r"__+", "_", norm_key)
             
            normalized_mapping[norm_key] = v

        # ------------------------------- Filter mapping to only columns that exist -------------------------------
        available_mapping = {src: tgt for src, tgt in normalized_mapping.items() if src in self.df.columns}
        
        missing_keys = set(normalized_mapping.keys()) - set(available_mapping.keys())
        if missing_keys:
            print(f"Warning: The following mapping keys were not found in the normalized input: {missing_keys}")
            print(f"Available normalized columns: {self.df.columns.tolist()[:10]}...")

        self.df = self.df.rename(columns=available_mapping)
    
        
        kept_columns = list(available_mapping.values())
        if not kept_columns:
            print("Warning: No columns were mapped! Result will be empty.")
            
        self.df = self.df[kept_columns]
        print(f"Applied mapping. Retained {len(self.df.columns)} columns.")

    def convert_units(self):
        # ------------------------------- Automatically converts units based on column names -------------------------------
        # Logic:
        # - If column has suffix `_mg_per_l` or `_mg_l` (or similar variants users might map to),
        #   and the target in `schema_config` is `mmol/L` or `µmol/L`.
        # - However, the standard attributes in `schema_config` already have suffixes like `_in_mmol/L`.
        # - So if the user mapped a column to `Na_in_mg/L` (which isn't in the standard list, but let's say they did),
        #   we need to convert it to `Na_in_mmol/L`.

        if self.df is None: return
        
        for col in self.df.columns:
            # ------------------------------- Check if this is a temporary mg/L column that needs conversion -------------------------------
            # We look for the "base" name (e.g., "Na") in the MOLAR_MASS dict
            
            # ------------------------------- Simple heuristic: user maps to "Na_in_mg/L". We want "Na_in_mmol/L" -------------------------------
            if "_in_mg/L" in col or "_in_mg_L" in col:
                base_name = col.replace("_in_mg/L", "").replace("_in_mg_L", "")
                
                # ------------------------------- Check if we have a molar mass for this -------------------------------
                if base_name in self.molar_masses:
                    molar_mass = self.molar_masses[base_name]
                    
                    # ------------------------------- Determine target unit from schema -------------------------------
                    # We look for a target attribute that starts with base_name
                    target_col_mmol = f"{base_name}_in_mmol/L"
                    target_col_umol = f"{base_name}_in_umol/L"
                    
                    if target_col_mmol in self.attributes:
                        # ------------------------------- Convert mg/L -> mmol/L -------------------------------
                        print(f"Converting {col} to {target_col_mmol}...")
                        self.df[target_col_mmol] = self._clean_numeric(self.df[col]) / molar_mass
                        
                    elif target_col_umol in self.attributes:
                        # ------------------------------- Convert mg/L -> µmol/L -------------------------------
                        print(f"Converting {col} to {target_col_umol}...")
                        self.df[target_col_umol] = (self._clean_numeric(self.df[col]) / molar_mass) * 1000.0
                        
                    else:
                        print(f"No target unit found for {base_name} in schema. Keeping {col}.")
                else:
                    print(f"No molar mass found for {base_name}. Cannot convert {col}.")
        
        # ------------------------------- Also handle "dissolved_oxygen" special naming if present in mapping -------------------------------

    def _clean_numeric(self, series: pd.Series) -> pd.Series:
        # ------------------------------- Coerces series to numeric, replacing errors with NaN -------------------------------
        return pd.to_numeric(series, errors='coerce')

    def validate_schema(self):
        # ------------------------------- Checks which attributes from the global schema are missing -------------------------------
        if self.df is None: return
        
        missing = [attr for attr in self.attributes if attr not in self.df.columns]
        present = [attr for attr in self.attributes if attr in self.df.columns]
        
        print(f"Schema Validation:")
        print(f"  - {len(present)} attributes found.")
        print(f"  - {len(missing)} attributes missing.")
        
    def save_data(self):
        # ------------------------------- Saves the dataframe to the output path -------------------------------
        if self.df is None: return
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.output_path.suffix == '.xlsx':
             with pd.ExcelWriter(self.output_path, engine="xlsxwriter") as writer:
                self.df.to_excel(writer, index=False, sheet_name="Data")
                self._format_excel(writer, self.df, "Data")
        elif self.output_path.suffix == '.csv':
            self.df.to_csv(self.output_path, index=False)
            
        print(f"Saved data to: {self.output_path}")

    def _format_excel(self, writer, df, sheet_name):
        # ------------------------------- Applies basic formatting to the Excel output -------------------------------
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        
        header_format = workbook.add_format({
            "bold": True,
            "text_wrap": True,
            "valign": "center",
            "fg_color": "#D3D3D3",
            "border": 1
        })
        
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        for i, col in enumerate(df.columns):
            max_len =  min(max(df[col].astype(str).map(len).max(), len(str(col))) + 2, 50)
            worksheet.set_column(i, i, max_len)
