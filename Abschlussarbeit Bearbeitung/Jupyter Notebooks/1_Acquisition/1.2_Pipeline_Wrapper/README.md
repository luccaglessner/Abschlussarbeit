# 1.2 Pipeline Wrapper - User Guide

This tool is designed to convert any geothermal dataset (`.xlsx` or `.csv`) into the standardized schema required for the Bachelor thesis. It automates loading, renaming, and unit conversion.

## Step-by-Step Instructions

### 1. Open the Notebook
Start the Jupyter Notebook **`Pipeline_Execution.ipynb`** located in this folder.

### 2. Configure Paths
In the **"1. Configuration"** section (Code Cell 2), simply enter the **filenames**.
The notebook is configured to look in the **`Input`** folder and save to the **`Output`** folder automatically.

```python
INPUT_FILENAME = "My_Raw_Data.xlsx"
OUTPUT_FILENAME = "My_Clean_Data.xlsx"
```

### 3. Define Column Mapping (The "Dictionary")
In the `COLUMN_MAPPING` dictionary, define which column in your file corresponds to which standard attribute.
The format is always: `"Name in your file" : "Target Attribute"`.

**Example:**
```python
COLUMN_MAPPING = {
    # Left (Your File)     # Right (Standard)
    "Site ID":             "location",
    "Water Temp":          "temperature_in_c",
    "pH-Wert":             "pH",
    
    # Automatic Conversion
    "Sodium (mg/l)":       "Na_in_mg/L", 
    "Sulphate":            "SO4_in_mg/L" 
}
```

### 4. Automatic Unit Conversion
The system automatically detects if a conversion from **mg/L** to **mmol/L** (or µmol/L) is necessary.

*   **Your input is already correct (mmol/L)?** 
    $\rightarrow$ Map directly to the target (e.g., `"Na_in_mmol/L"`).
    
*   **Your input is in mg/L?** 
    $\rightarrow$ Map to `"XX_in_mg/L"` (e.g., `"Na_in_mg/L"`). 
    The pipeline uses the molar mass from `schema_config.py`, converts the value, and saves it in the correct target column (`Na_in_mmol/L`).

### 5. Execute
Execute all cells in the notebook (Menu: *Run* $\rightarrow$ *Run All Cells*).
The cleaned file will be created at the specified output location.

---

## Important Files in this Folder
*   **`Pipeline_Execution.ipynb`**: The user interface for daily use.
*   **`pipeline_logic.py`**: The program code (core logic). **Do not modify.**
*   **`schema_config.py`**: List of all valid target attributes, units, and molar masses. New substances can be added here.
