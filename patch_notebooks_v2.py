
import nbformat
from pathlib import Path

def patch_notebook_cell_by_id(path, cell_id, new_source):
    print(f"Patching cell {cell_id} in {path.name}...")
    nb = nbformat.read(path, as_version=4)
    patched = False
    for cell in nb.cells:
        if cell.get('id') == cell_id:
            cell.source = new_source
            patched = True
            print(f"  -> Found and patched cell by ID.")
            break
    
    if patched:
        nbformat.write(nb, path)
        print(f"  -> Successfully updated {path.name}")
    else:
        print(f"  -> WARNING: Cell with ID {cell_id} not found in {path.name}")

# Paths
BASE_DIR = Path(r"f:\Abschlussarbeit\Abschlussarbeit Bearbeitung\Jupyter Notebooks")
INF_NB = BASE_DIR / "4_Imputation" / "4.2_Inference" / "Inference.ipynb"

# --- Clean source for Inference.ipynb Cell 5 ---
clean_inf_source = '''# ------------------------- Daten laden & Preprocessing -------------------------
preprocessing_root = base_dir.parent.parent / "3_Machine-Learning" / "3.1_Preprocessing" / "Preprocessing"
latest_prep_folder = max([f for f in preprocessing_root.iterdir() if f.is_dir()], key=lambda f: f.stat().st_mtime)
df_full = pd.read_csv(latest_prep_folder / "Preprocessed_SOM_Ready.csv", low_memory=False)

if INCLUDE_INCOMPLETE:
    data_subset = df_full[features].dropna(how='all').copy()
else:
    data_subset = df_full[features].dropna().copy()

X_original_full = data_subset.values
X_train_orig, X_test_orig = train_test_split(X_original_full, test_size=0.1, random_state=42)

# ------------------------- Skalierung & NaN-Handling -------------------------
X_train_scaled = scaler.transform(np.nan_to_num(X_train_orig))
X_train_scaled[np.isnan(X_train_orig)] = np.nan

X_test_scaled = scaler.transform(np.nan_to_num(X_test_orig))
X_test_scaled[np.isnan(X_test_orig)] = np.nan

res_dir = Path("Inference_Results") / model_dir.name
res_dir.mkdir(parents=True, exist_ok=True)

# ------------------------- Inferenz-Loop über Maskierungslevel -------------------------
num_features = len(features)
for k in range(1, num_features):
    print(f"Processing Level {k}...")
    combos = list(itertools.combinations(range(num_features), k))
    
    if len(combos) > 50: combos_to_run = random.sample(combos, 50)
    else: combos_to_run = combos
    
    all_results = []
    
    # ------------------------- Hilfsfunktion für Inferenz (Batching) -------------------------
    def run_inference_on_split(X_orig_split, X_scaled_split, split_name):
        split_results = []
        for combo_indices in combos_to_run:
            X_target_scaled = X_scaled_split.copy()
            # ------------------------- Künstliche Maskierung -------------------------
            is_masked = np.zeros(X_target_scaled.shape, dtype=bool)
            for f_idx in combo_indices:
                mask_condition = ~np.isnan(X_target_scaled[:, f_idx])
                X_target_scaled[mask_condition, f_idx] = 0.0
                is_masked[mask_condition, f_idx] = True
            
            X_target_scaled = np.nan_to_num(X_target_scaled)
            
            # ------------------------- VAE Rekonstruktion -------------------------
            with torch.no_grad():
                recon, _, _ = vae(torch.from_numpy(X_target_scaled).float())
                X_recon_scaled = recon.numpy()
            
            X_recon_orig = scaler.inverse_transform(X_recon_scaled)
            
            # ------------------------- Ergebnisse sammeln -------------------------
            for f_idx in combo_indices:
                mask = is_masked[:, f_idx]
                if not mask.any(): continue
                
                o_vals = X_orig_split[mask, f_idx]
                i_vals = X_recon_orig[mask, f_idx]
                for o, i in zip(o_vals, i_vals):
                    split_results.append({
                        "Feature": features[f_idx],
                        "Original": o,
                        "Imputed": i,
                        "Split": split_name,
                        "Masking_Level": k,
                        "Masked_Combination": str([features[i] for i in combo_indices])
                    })
        return split_results

    # 1. Test-Split (Vollständig)
    all_results.extend(run_inference_on_split(X_test_orig, X_test_scaled, "Test"))
    
    # 2. Train-Split (Zusätzlich 10% Stichprobe für Visualisierung)
    n_train = len(X_train_orig)
    sample_size = max(1, n_train // 10)
    sample_indices = np.random.choice(n_train, sample_size, replace=False)
    
    X_train_sample_orig = X_train_orig[sample_indices]
    X_train_sample_scaled = X_train_scaled[sample_indices]
    
    all_results.extend(run_inference_on_split(X_train_sample_orig, X_train_sample_scaled, "Train"))
    
    # ------------------------- Speichern der Resultate -------------------------
    if all_results:
        df_res = pd.DataFrame(all_results)
        df_res.to_csv(res_dir / f"Imputation_Results_Run_{TARGET_RUN_INDEX:03d}_Level_{k}.csv", index=False, float_format='%.4f')

print("Done.")'''

if INF_NB.exists(): patch_notebook_cell_by_id(INF_NB, "5", clean_inf_source)
