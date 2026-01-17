#!/usr/bin/env python
# coding: utf-8

# # 4.3
# 
# <div class="alert alert-info"> VAE Imputation: Evaluation
# 
# ## Ziel
# Bewertung der Imputationsgüte. 
# Es wird verglichen, wie gut die vom VAE rekonstruierten Werte ("Imputed") mit den tatsächlichen Werten ("Original") übereinstimmen.
# 
# ## Metriken
# - **RMSE**: Root Mean Squared Error (Niedriger ist besser)
# - **Verteilungs-Check**: Visueller Vergleich der Dichtefunktionen (KDE).
# 
# ## Output
# - Grafiken zur Imputationsqualität pro Feature.
# - Zusammenfassung der Fehlerstatistik.
# </div>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

sns.set_theme(style="whitegrid")


# In[ ]:


# ------------------------- Evaluation (PDF Report) -------------------------
import time
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.backends.backend_pdf import PdfPages

sns.set_theme(style="whitegrid")

base_dir = Path.cwd()
inference_root = base_dir.parent / "4.2_Inference" / "Inference_Results"
def log(msg): print(msg)
log(f"Suche Ergebnisse in: {inference_root}")

# ------------------------- Warten auf Ordnererstellung -------------------------
results_folder = None
wait_counter = 0
max_wait_seconds = 1800 

while results_folder is None:
    if not inference_root.exists():
        log("Inferenzordner fehlt noch")
        time.sleep(2)
        continue
        
    timestamp_folders = [f for f in inference_root.iterdir() if f.is_dir()]
    if timestamp_folders:
        candidate = max(timestamp_folders, key=lambda f: f.stat().st_mtime)
        age = time.time() - candidate.stat().st_mtime
        
        if age < 7200:
             results_folder = candidate
             log(f"Nutze Ergebnisse aus: {results_folder.name}")
             break
        else:
             pass
             
    time.sleep(5)
    wait_counter += 5
    if wait_counter >= max_wait_seconds:
        log("Timeout!")
        raise TimeoutError("Keine neuen Ergebnisse von 4.2 gefunden.")

# ------------------------- Setup Loop -------------------------
models_dir_path = results_folder.parent.parent.parent / "4.1_VAE_Imputation" / "Models" / results_folder.name
signal_file_path = models_dir_path / "DONE_TRAINING"
out_dir_eval = base_dir / "Evaluation_Results" / results_folder.name
out_dir_eval.mkdir(parents=True, exist_ok=True)
log(f"Ergebnis gespeichert in: {out_dir_eval}")

processed_files = set()
summary_stats = []
all_metrics = []
done_training = False

log("\nStarte Monitoring Loop...")

while True:
    
    current_files = list(results_folder.glob("Imputation_Results_*.csv"))
    # ------------------------- Valide Dateien herausfiltern -------------------------
    valid_files = [f for f in current_files if f.stat().st_size > 0]
    new_files = [f for f in valid_files if f.name not in processed_files]

    def get_file_index(p):
        name = p.stem
        parts = name.split("_")
        for part in parts:
             if part.isdigit(): return int(part)
        return 99999
        
    new_files.sort(key=get_file_index)
    
    for file_path in new_files:
        clean_name = file_path.stem.replace("Imputation_Results_Model_", "").replace("Imputation_Results_Run_", "")
        if "Trennlinie" in clean_name:
             processed_files.add(file_path.name)
             continue

        log(f"Evaluation: {clean_name}")
        
        pdf_path = out_dir_eval / f"Analysis_Run_{clean_name}.pdf"
        
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                processed_files.add(file_path.name)
                continue
                
            features = df['Feature'].unique()
            rmse_per_feat = {}
            
            with PdfPages(pdf_path) as pdf:
                # =========================================================================
                # ------------------------- SEITE 1: DASHBOARD -------------------------
                # =========================================================================
                fig = plt.figure(figsize=(11.69, 8.27)) # A4 Quer
                gs = fig.add_gridspec(3, 2, height_ratios=[0.25, 0.45, 0.3], wspace=0.3, hspace=0.4)
                
                # ------------------------- Header (Titel) -------------------------
                ax_header = fig.add_subplot(gs[0, :])
                ax_header.axis('off')
                ax_header.text(0.5, 0.7, f"Evaluation Report", ha='center', va='center', fontsize=24, weight='bold')
                ax_header.text(0.5, 0.35, f"Run: {clean_name}", ha='center', va='center', fontsize=18)
                ax_header.text(0.5, 0.1, f"Timestamp: {time.strftime('%Y-%m-%d %H:%M')}", ha='center', va='center', fontsize=10, color='gray')

                # ------------------------- Metadaten Laden -------------------------
                meta_filename = f"Model_Run_{clean_name}_meta.json"
                if "Test-Run" in clean_name and not (models_dir_path / meta_filename).exists(): 
                     meta_filename = f"Model_{clean_name}_meta.json"
                if not (models_dir_path / meta_filename).exists():
                     meta_filename = f"{file_path.stem.replace('Imputation_Results_', '')}_meta.json"

                meta_path = models_dir_path / meta_filename
                
                # ------------------------- Standard Texte -------------------------
                vae_info_text = "VAE Info unavailalbe."
                cv_folds = []
                cv_mean = "N/A"

                if meta_path.exists():
                    try:
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                        
                        # ------------------------- VAE Info Text -------------------------
                        n_feat = len(meta.get('features_mapped', []))
                        feat_list_str = ", ".join([f.replace("_in_mmol/L_log_gauss", "") for f in meta.get('features_mapped', [])[:8]])
                        if n_feat > 8: feat_list_str += "..."
                        
                        vae_info_text = (
                            f"Architecture:\n"
                            f"  • Input Dim: {n_feat}\n"
                            f"  • Latent Dim: {meta.get('latent_dim', 16)}\n"
                            f"  • Hidden Dim: {meta.get('hidden_dim', 256)}\n"
                            f"  • Epochs: {meta.get('epochs_trained', 150)}\n\n"
                            f"Features ({n_feat}):\n"
                            f"  {feat_list_str}"
                        )
                        
                        # CV Data
                        cv_mean = meta.get('cv_mean_score', meta.get('cv_score', 'N/A'))
                        cv_folds = meta.get('cv_fold_scores', [])
                        
                    except Exception as e:
                        vae_info_text = f"Error loading meta: {e}"

                # ------------------------- Linke Seite: Methodik -------------------------
                ax_method = fig.add_subplot(gs[1, 0])
                ax_method.axis('off')
                ax_method.set_title("1. Methodology & Model", fontsize=12, loc='left', weight='bold')
                
                method_text = (
                    f"Variational Autoencoder (VAE):\n"
                    f"Compression of input data into significant latent\n"
                    f"features to reconstruct missing values.\n\n"
                    f"{vae_info_text}"
                )
                ax_method.text(0, 0.9, method_text, va='top', fontsize=10, linespacing=1.6, family='monospace')

                # ------------------------- Rechte Spalte: Kreuzvalidierung -------------------------
                ax_val = fig.add_subplot(gs[1, 1])
                ax_val.set_title("2. Cross-Validation (CV) Performance", fontsize=12, loc='left', weight='bold')
                
                if cv_folds and isinstance(cv_folds, list) and len(cv_folds) > 1:
                    # ------------------------- Balkendiagramm erstellen -------------------------
                    x_labels = [f"Fold {i+1}" for i in range(len(cv_folds))]
                    sns.barplot(x=x_labels, y=cv_folds, ax=ax_val, color='#4c72b0')
                    ax_val.set_ylabel("Reconstruction Loss (MSE + KLD)")
                    ax_val.set_ylim(bottom=0)
                    # ------------------------- CV Schnitt -------------------------
                    if isinstance(cv_mean, (int, float)):
                        ax_val.axhline(cv_mean, color='red', linestyle='--', label=f'Mean: {cv_mean:.4f}')
                        ax_val.legend()
                    ax_val.grid(axis='y', alpha=0.5)
                else:
                    # ------------------------- Fallback Text -------------------------
                    ax_val.axis('off')
                    fallback_text = "Detaillierter Bericht nicht möglich (keine CV Daten)?"
                    if cv_mean != "N/A":
                         fallback_text += f"\n\nReported Mean CV Score: {cv_mean:.4f}"
                    ax_val.text(0.5, 0.5, fallback_text, ha='center', va='center', fontsize=10, 
                                bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="gray"))

                # ------------------------- Metriken -------------------------
                ax_metrics = fig.add_subplot(gs[2, :])
                ax_metrics.axis('off')
                ax_metrics.set_title("3. Evaluation Metrics (Test Set 10%)", fontsize=12, loc='left', weight='bold')
                
                metrics_text = (
                    f"Performance on global hold-out Test Set (completely unseen during training/CV):\n"
                    f"• R² Score (Coefficient of Determination): 1.0 is perfect prediction. < 0 means worse than mean-baseline.\n"
                    f"• RMSE (Root Mean Squared Error): Average deviation in original units (mmol/L etc.). Lower is better.\n"
                    f"• Plots: Comparing Original values (x-axis) vs Imputed values (y-axis)."
                )
                ax_metrics.text(0, 0.8, metrics_text, va='top', fontsize=10, linespacing=1.6)

                pdf.savefig(fig)
                plt.close()
                
                # ------------------------- SEITE 2+: FEATURE PLOTS -------------------------
                for feature in features:
                    try:
                        subset = df[df['Feature'] == feature]
                        y_true = subset['Original']
                        y_pred = subset['Imputed']
                        if len(y_true) < 2: continue
                        mse = mean_squared_error(y_true, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_true, y_pred)
                        rmse_per_feat[feature] = rmse
                        all_metrics.append({'Run_ID': clean_name, 'Feature': feature, 'R2': r2, 'RMSE': rmse})
                        
                        # ------------------------- Plot -------------------------
                        fig, ax = plt.subplots(1, 2, figsize=(11.69, 5))
                        
                        # ------------------------- Scatter -------------------------
                        sns.scatterplot(x=y_true, y=y_pred, ax=ax[0], alpha=0.5, hue=subset['Split'], palette={'Train': 'blue', 'Test': 'red'})
                        low = min(y_true.min(), y_pred.min()) - 0.1
                        high = max(y_true.max(), y_pred.max()) + 0.1
                        ax[0].plot([low, high], [low, high], 'k--', alpha=0.7, label='Ideal')
                        ax[0].set_title(f"Scatter Plot (R²={r2:.3f})")
                        ax[0].set_xlabel("Original Value (Scaled)")
                        ax[0].set_ylabel("Imputed Value (Scaled)")
                        ax[0].legend()
                        
                        # ------------------------- KDE -------------------------
                        sns.kdeplot(y_true, ax=ax[1], label="Original", fill=True, alpha=0.3, color='blue')
                        sns.kdeplot(y_pred, ax=ax[1], label="Imputed", fill=True, alpha=0.3, color='orange')
                        ax[1].set_title(f"Distribution (RMSE={rmse:.3f})")
                        ax[1].set_xlabel("Value (Scaled)")
                        ax[1].legend()
                        
                        plt.suptitle(f"Feature: {feature}", fontsize=16, weight='bold')
                        plt.tight_layout()
                        
                        pdf.savefig(fig)
                        plt.close(fig)
                        
                    except Exception as e:
                        log(f"!!! Fehler beim Plotten von Feature {feature}: {e}")
                        try: plt.close(fig)
                        except: pass
                        continue

            avg_rmse = np.mean(list(rmse_per_feat.values())) if rmse_per_feat else 0
            summary_stats.append({
                "Run": clean_name,
                "Avg_RMSE": avg_rmse,
                "Features": list(rmse_per_feat.keys())
            })
            processed_files.add(file_path.name)
            log(f"Erstelle PDF: {pdf_path.name}")
            
        except Exception as e:
            log(f"Fehler bei {file_path.name}: {e}")
            processed_files.add(file_path.name)

    if signal_file_path.exists():
        if not done_training:
             log("Signal DONE_TRAINING erkannt.")
             done_training = True
             
        if not new_files:
             current_check = [f for f in list(results_folder.glob("Imputation_Results_*.csv")) if f.stat().st_size > 0]
             remaining = [f for f in current_check if f.name not in processed_files]
             if not remaining:
                 # ------------------------- Globales .pdf-Ergebnis -------------------------
                 if summary_stats:
                    df_summary = pd.DataFrame(summary_stats)
                    csv_path = out_dir_eval / "Summary_Evaluation.csv"
                    df_summary.to_csv(csv_path, index=False)
                    
                    # ------------------------- .pdf Zusammenfassung -------------------------
                    sum_pdf_path = out_dir_eval / "Evaluation_Summary_Report.pdf"
                    with PdfPages(sum_pdf_path) as pdf:
                        # ------------------------- Tabelle -------------------------
                        fig, ax = plt.subplots(figsize=(11.69, max(4, len(df_summary)*0.3 + 2)))
                        ax.axis('tight')
                        ax.axis('off')
                        ax.table(cellText=df_summary[['Run', 'Avg_RMSE']].values, colLabels=['Run', 'Avg_RMSE'], loc='center')
                        ax.set_title("Overall Performance Summary", fontsize=16)
                        pdf.savefig(fig)
                        plt.close()
                        
                        # ------------------------- Balkendiagramm erstellen -------------------------
                        fig2, ax2 = plt.subplots(figsize=(11.69, 6))
                        sns.barplot(data=df_summary, x='Run', y='Avg_RMSE', ax=ax2)
                        plt.xticks(rotation=90)
                        plt.tight_layout()
                        pdf.savefig(fig2)
                        plt.close()
                        
                    log(f"Global Summary PDF created: {sum_pdf_path.name}")

                 log("Alle Evaluierungen abgeschlossen.")
                 break
                 
    time.sleep(5)


# In[ ]:


# ------------------------- Finalen Bericht sichern -------------------------
if all_metrics:
    df_metrics = pd.DataFrame(all_metrics)
    summary_path = eval_output_dir / "Evaluation_Summary.csv"
    df_metrics.to_csv(summary_path, index=False)
    print(f"\nZusammenfassung gespeichert: {summary_path}")
    
    # ------------------------- Top 5 anzeigen -------------------------
    print(df_metrics.groupby('Run_ID')['R2'].mean().sort_values(ascending=False).head())

    # ------------------------- PDF Bericht Erstellung -------------------------
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    pdf_path = eval_output_dir / "Evaluation_Report.pdf"
    print(f"Generating PDF Report: {pdf_path.name}...")
    
    with PdfPages(pdf_path) as pdf:

        # ------------------------- PDF Bericht Erstellung -------------------------
        fig, ax = plt.subplots(figsize=(11.69, 8.27)) # A4
        ax.axis('off')
        ax.text(0.5, 0.95, "VAE Imputation - Evaluation Report", ha='center', fontsize=24, weight='bold')
        ax.text(0.5, 0.90, f"Date: {time.strftime('%Y-%m-%d %H:%M')}", ha='center', fontsize=12)
        
        # ------------------------- Top 10 Modelle -------------------------
        agg_funcs = {'R2': 'mean', 'RMSE': 'mean', 'Features_List': 'first'}
        top_n = df_metrics.groupby('Run_ID').agg(agg_funcs).sort_values('R2', ascending=False).head(10).reset_index()
        top_n['R2'] = top_n['R2'].round(4)
        top_n['RMSE'] = top_n['RMSE'].round(4)
        
        table_data = [top_n.columns.values.tolist()] + top_n.values.tolist()
        table = ax.table(cellText=table_data, colLabels=top_n.columns, loc='center', cellLoc='center', bbox=[0.1, 0.4, 0.8, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        ax.text(0.5, 0.82, "Top 10 Models (by Mean R²)", ha='center', fontsize=16)
        
        pdf.savefig(fig)
        plt.close()
        
        # ------------------------- Perfomanz Ergebnisse -------------------------
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        sns.boxplot(data=df_metrics, x='Feature', y='R2', ax=ax)
        ax.set_title("R² Score Distribution per Feature across all Runs", fontsize=16)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylim(-0.1, 1.1)
        pdf.savefig(fig)
        plt.close()
        
        # ------------------------- Details zum besten Modell -------------------------
        best_run_id = top_n.iloc[0]['Run_ID']
        best_run_file = results_root / f"Imputation_Results_{best_run_id}.csv"
        
        if best_run_file.exists():
            df_best = pd.read_csv(best_run_file)
            features = df_best['Feature'].unique()
            
            # ------------------------- Scatter Plots für Modell -------------------------
            cols = 3
            rows = (len(features) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(11.69, 4*rows))
            fig.suptitle(f"Best Model: {best_run_id} - Original vs Imputed", fontsize=16)
            axes = axes.flatten()
            
            for i, feat in enumerate(features):
                ax = axes[i]
                subset = df_best[df_best['Feature'] == feat]
                
                # ------------------------- Scatter -------------------------
                sns.scatterplot(data=subset, x='Original', y='Imputed', ax=ax, alpha=0.3, s=10)
                
                # ------------------------- Ideallinie [Original = Imputed] -------------------------
                min_val = min(subset['Original'].min(), subset['Imputed'].min())
                max_val = max(subset['Original'].max(), subset['Imputed'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5)
                
                # ------------------------- Metrik -------------------------
    
                r2_vals = df_metrics[(df_metrics['Run_ID'] == best_run_id) & (df_metrics['Feature'] == feat)]['R2'].values
                r2_val = r2_vals[0] if len(r2_vals) > 0 else 0.0

                ax.set_title(f"{feat} (R²: {r2_val:.2f})")
                
            for j in range(i+1, len(axes)): axes[j].axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close()
            
    print(f"PDF Saved.")
