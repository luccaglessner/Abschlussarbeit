from pathlib import Path
import pandas as pd
import numpy as np
from pipeline_logic import GeothermalDataPipeline
import schema_config
import os

# ----------------------- Define Paths -----------------------
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "Input"
OUTPUT_DIR = BASE_DIR / "Output"

INPUT_PATH = INPUT_DIR / "test_input.xlsx"
OUTPUT_PATH = OUTPUT_DIR / "test_output.xlsx"

# ----------------------- Make sure directories exist -----------------------
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ----------------------- Define mapping -----------------------
MAPPING = {
    "Location Name": "location",
    "Temp (C)": "temperature_in_c", 
    "pH Value": "pH",
    "Sodium (mg/L)": "Na_in_mg/L" 
}

def create_test_data():
    # ----------------------- Generates a dummy Excel file for testing -----------------------
    print(f"Creating test data at {INPUT_PATH}...")
    data = {
        "Location Name": ["Site A", "Site B", "Site C"],
        "  Temp (C) ": [25.5, 60.2, 105.1],
        "pH Value": [6.5, 7.2, 8.1],
        "Sodium (mg/L)": [150.0, 500.5, 2300.0], 
        "Unmapped Column": [1, 2, 3] 
    }
    df = pd.DataFrame(data)
    df.to_excel(INPUT_PATH, index=False)

def test_pipeline():
    # ----------------------- 1. Create Data -----------------------
    create_test_data()
    
    print("\nRunning Pipeline Verification...")
    
    # ----------------------- 2. Run Pipeline -----------------------
    pipeline = GeothermalDataPipeline(INPUT_PATH, OUTPUT_PATH, MAPPING)
    result_df = pipeline.run()
    
    # ----------------------- 3. Verify Columns -----------------------
    expected_cols = ["location", "temperature_in_c", "pH", "Na_in_mmol/L"]
    for col in expected_cols:
        assert col in result_df.columns, f"Missing expected column: {col}"
        
    print("Column verification successful.")
    
    # ----------------------- 4. Verify Unit Conversion -----------------------
    # Na: 150 mg/L -> / 22.989769 ~= 6.5246 mmol/L
    na_val = result_df.loc[0, "Na_in_mmol/L"]
    expected_val = 150.0 / 22.989769
    assert np.isclose(na_val, expected_val, atol=0.001), f"Unit conversion failed. Got {na_val}, expected {expected_val}"
    
    print("Unit conversion verification successful.")
    
    # ----------------------- 5. Verify Dropped Columns -----------------------
    assert "Unmapped Column" not in result_df.columns, "Unmapped column was not dropped."
    
    print("Unmapped column drop verification successful.")
    print("\nALL TESTS PASSED!")

if __name__ == "__main__":
    test_pipeline()
