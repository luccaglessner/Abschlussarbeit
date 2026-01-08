import os
import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime

# ----------------------------------------------- Basisverzeichnis -----------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()

import pipeline_logger

def run_notebook(notebook_path, env_vars=None):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starte: {notebook_path.name}")
    try:
        start_time = time.time()
        
        # Prepare Environment
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
            
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            str(notebook_path)
        ]
        # Stderr für Logging erfassen
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        print(f"  -> Fertig in {time.time() - start_time:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  -> FEHLER: {e.stderr}")
        pipeline_logger.log_error(f"Fehler bei Ausführung von {notebook_path.name}", exception=e, stderr=e.stderr)
        return False
    except Exception as e:
        print(f"  -> UNERWARTETER FEHLER: {e}")
        pipeline_logger.log_error(f"Unerwarteter Fehler bei {notebook_path.name}", exception=e)
        return False

def main():
    print("=== Pipeline 4: Imputation & Machine Learning ===")
    
    # Configuration for Strategies
    strategies = [
        {
            "name": "Main-Ions",
            "imputation_dir": "4.1_Imputation-Main-Ions",
            "imputation_nb": "VAE_Imputation.ipynb",
            "imputation_file": "Imputed_Data_Raw.csv",
            "output_subdir": "Output_MainIons"
        },
        {
            "name": "Everything",
            "imputation_dir": "4.2_Imputation-Everything",
            "imputation_nb": "MiniSom_Machine-Learning-Everything.ipynb", # Assuming this is intended as imputation or creates one
            "imputation_file": "Imputed_Data_Raw.csv", # Assuming a standard output name if it were an imputation
            "output_subdir": "Output_Everything"
        }
    ]

    preprocessing_nb = BASE_DIR / "4_Imputation/4.3_Preprocessing/Preprocessing.ipynb"
    ml_nb = BASE_DIR / "4_Imputation/4.4_Machine-Learning-MiniSOM/MiniSom_Machine-Learning.ipynb"

    for strat in strategies:
        print(f"\n--- Strategy: {strat['name']} ---")
        
        # 1. Imputation
        imp_dir = BASE_DIR / "4_Imputation" / strat["imputation_dir"]
        imp_nb = imp_dir / strat["imputation_nb"]
        imp_output = imp_dir / strat["imputation_file"]
        
        if imp_nb.exists():
            print(f"Running Imputation: {imp_nb.name}")
            if not run_notebook(imp_nb):
                print(f"Skipping rest of strategy {strat['name']} due to Imputation failure.")
                continue
        else:
            print(f"Warning: Imputation Notebook not found: {imp_nb}")
            # Proceed only if file exists (maybe pre-calculated)
        
        if not imp_output.exists():
            print(f"Imputation Output File not found: {imp_output}")
            print(f"Skipping Preprocessing & ML for {strat['name']} (No input data).")
            continue

        # 2. Preprocessing
        print(f"Running Preprocessing on {strat['name']} data...")
        prep_out_dir = BASE_DIR / "4_Imputation/4.3_Preprocessing" / strat["output_subdir"]
        
        env_prep = {
            "PREPROCESSING_INPUT_FILE": str(imp_output),
            "PREPROCESSING_OUTPUT_DIR": str(prep_out_dir)
        }
        
        if preprocessing_nb.exists():
            if not run_notebook(preprocessing_nb, env_vars=env_prep):
                print("Preprocessing failed.")
                continue
        else:
            print(f"Preprocessing NB not found: {preprocessing_nb}")
            continue
            
        prep_output_file = prep_out_dir / "Preprocessed_SOM_Ready.csv"
        if not prep_output_file.exists():
            print(f"Preprocessing Output not found: {prep_output_file}. Skipping ML.")
            continue
            
        # 3. Machine Learning
        print(f"Running Machine Learning on {strat['name']} data...")
        env_ml = {
            "ML_INPUT_FILE": str(prep_output_file)
        }
        
        if ml_nb.exists():
            run_notebook(ml_nb, env_vars=env_ml)
        else:
            print(f"ML Notebook not found: {ml_nb}")

    print("\n--- Pipeline 4 abgeschlossen ---")

if __name__ == "__main__":
    main()
