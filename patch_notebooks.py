
import nbformat
from pathlib import Path

def patch_notebook(path, target_text, replacement_text):
    print(f"Patching {path.name}...")
    nb = nbformat.read(path, as_version=4)
    patched = False
    for cell in nb.cells:
        if cell.cell_type == 'code' and target_text in cell.source:
            # We use a more robust replacement that handles multiple lines
            if target_text in cell.source:
                cell.source = cell.source.replace(target_text, replacement_text)
                patched = True
                print(f"  -> Found and patched cell.")
    
    if patched:
        nbformat.write(nb, path)
        print(f"  -> Successfully updated {path.name}")
    else:
        print(f"  -> WARNING: No match found in {path.name}")

# Paths
BASE_DIR = Path(r"f:\Abschlussarbeit\Abschlussarbeit Bearbeitung\Jupyter Notebooks")
INF_NB = BASE_DIR / "4_Imputation" / "4.2_Inference" / "Inference.ipynb"
EVAL_NB = BASE_DIR / "4_Imputation" / "4.3_Evaluation" / "Evaluation.ipynb"
KNN_NB = BASE_DIR / "5_kNN" / "5.1_kNearest-Neighbors.ipynb"

# --- 1. Patch Inference.ipynb ---
inf_target = '# ------------------------- Skalierung & NaN-Handling -------------------------'
inf_replacement = '''# ------------------------- Skalierung & NaN-Handling -------------------------
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
    
    all_results.extend(run_inference_on_split(X_train_sample_orig, X_train_sample_scaled, "Train"))'''

# --- 2. Patch Evaluation.ipynb ---
eval_target = 'ax_scat.legend(handles, ["Train-Split", "Test-Split"], loc=\'upper left\', fontsize=8)'
# I'll use a regex-like or simpler match for Evaluation as handles might have different quotes
# Actually, I'll just look for a unique part.
eval_target_unique = 'ax_scat.legend(handles, ["Train-Split", "Test-Split"]'
eval_replacement = """# Dynamische Legende basierend auf vorhandenen Splits
                              existing_splits = subset_plot['Split'].unique()
                              handles, labels = ax_scat.get_legend_handles_labels()
                              # Mapping für schönere Labels
                              legend_labels = [f"{l}-Split" for l in labels]
                              ax_scat.legend(handles, legend_labels, loc='upper left', fontsize=8, frameon=True)"""

# --- 3. Patch kNearest-Neighbors.ipynb ---
knn_target = 'all_results = []\n    for combo_indices in combos_to_run:'
knn_replacement = '''all_results = []
    
    # ------------------------- Hilfsfunktion für Inferenz -------------------------
    def run_knn_on_split(X_orig_split, X_scaled_split, split_name):
        split_results = []
        for combo_indices in combos_to_run:
            X_target_scaled = X_scaled_split.copy()
            X_target_orig = X_orig_split.copy()
            
            # ------------------------- Maskierung -------------------------
            is_masked = np.zeros(X_target_scaled.shape, dtype=bool)
            for f_idx in combo_indices:
                mask_condition = ~np.isnan(X_target_scaled[:, f_idx])
                X_target_scaled[mask_condition, f_idx] = np.nan
                is_masked[mask_condition, f_idx] = True
            
            # ------------------------- Imputieren -------------------------
            X_imputed_scaled = imputer.transform(X_target_scaled)
            X_imputed = scaler.inverse_transform(X_imputed_scaled)
            
            # ------------------------- Ergebnisse sammeln -------------------------
            for f_idx in combo_indices:
                 mask = is_masked[:, f_idx]
                 if not mask.any(): continue
                 
                 feat_name = features[f_idx]
                 orig_masked = X_target_orig[mask, f_idx]
                 imp_masked = X_imputed[mask, f_idx]
                 
                 for o, im in zip(orig_masked, imp_masked):
                     split_results.append({
                         "Feature": feat_name,
                         "Original": o,
                         "Imputed": im,
                         "Split": split_name,
                         "Masking_Level": k,
                         "Masked_Combination": str([features[i] for i in combo_indices])
                     })
        return split_results

    # 1. Test-Split (Vollständig)
    all_results.extend(run_knn_on_split(X_test_orig, X_test_scaled, "Test"))
    
    # 2. Train-Split (10% Stichprobe)
    n_train = len(X_train_orig)
    sample_size = max(1, n_train // 10)
    sample_indices = np.random.choice(n_train, sample_size, replace=False)
    X_train_sample_orig = X_train_orig[sample_indices]
    X_train_sample_scaled = X_train_scaled[sample_indices]
    all_results.extend(run_knn_on_split(X_train_sample_orig, X_train_sample_scaled, "Train"))'''

# Running patches
if INF_NB.exists(): patch_notebook(INF_NB, inf_target, inf_replacement)
if EVAL_NB.exists(): patch_notebook(EVAL_NB, eval_target_unique, eval_replacement)
if KNN_NB.exists(): patch_notebook(KNN_NB, knn_target, knn_replacement)
