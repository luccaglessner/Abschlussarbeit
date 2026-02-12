import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import math
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
from datetime import datetime

# ------------------------- Konfiguration -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_4_1 = BASE_DIR / "4_Imputation" / "4.1_VAE_Imputation" / "VAE_Imputation.ipynb"
NOTEBOOK_4_2 = BASE_DIR / "4_Imputation" / "4.2_Inference" / "Inference.ipynb"
NOTEBOOK_4_3 = BASE_DIR / "4_Imputation" / "4.3_Evaluation" / "Evaluation.ipynb"

# ------------------------- Globaler Status für die Historie -------------------------
results_history = []

def get_model_folders(limit=30):
    models_root = NOTEBOOK_4_1.parent / "Models"
    if not models_root.exists():
        print(f"Fehler: Models Ordner {models_root} nicht gefunden.")
        return []
        
    # ------------------------- Alle Ordner holen und nach Datum sortieren -------------------------
    folders = [f for f in models_root.iterdir() if f.is_dir()]
    folders.sort(key=lambda f: f.stat().st_mtime)
    
    # ------------------------- Die letzten 'limit' Ordner zurückgeben -------------------------
    return folders[-limit:]

def generate_progress_pdf(output_folder):
    if not results_history: return
    
    df = pd.DataFrame(results_history)
    
    # ------------------------- Index hinzufügen (basierend auf der Position in der Liste) -------------------------
    df["Index"] = range(1, len(df) + 1)
    
    pdf_path = output_folder / "Incremental_Progress.pdf"
    
    print(f"    -> Generiere PDF: {pdf_path.name}")
    
    # ------------------------- Plotting Setup -------------------------
    sns.set_theme(style="whitegrid")
    
    # ------------------------- Multipage PDF speichern -------------------------
    from matplotlib.backends.backend_pdf import PdfPages
    try:
        with PdfPages(pdf_path) as pdf:
            
            # ------------------------- Seite 1: Tabellarische Auflistung -------------------------
            fig1, ax1 = plt.subplots(figsize=(11.69, 8.27))
            ax1.axis('off')
            ax1.set_title("Incremental Run History (Regenerated)", fontsize=18, weight='bold')
            
            # ------------------------- Spalten für Tabelle auswählen -------------------------
            display_cols = ["Index", "Latent_Dim", "Hidden_Dim", 
                            "Avg_RMSE_Train", "Avg_RMSE_Test", 
                            "Avg_R2_Train", "Avg_R2_Test"]
            
            # ------------------------- Formatieren für Anzeige -------------------------
            df_display = df.copy()
            for col in display_cols[3:]: # Ab RMSE alle formatieren
                 if col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
            
            # ------------------------- Tabelle zeichnen -------------------------
            if not df_display.empty:
                table_data = [display_cols] + df_display[display_cols].values.tolist()
                table = ax1.table(cellText=table_data, colLabels=display_cols, loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.5)
            
            pdf.savefig(fig1)
            plt.close(fig1)
            
            # ------------------------- Seite 2: Plots (RMSE & R2) -------------------------
            fig2, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
            fig2.suptitle("Performance Metrics Progression (Train vs Test)", fontsize=16)
            
            # ------------------------- Plot Helper Funktion -------------------------
            def plot_metric(ax, metric_train, metric_test, title, higher_better=False):
                if metric_train in df.columns and not df[metric_train].isna().all():
                    sns.lineplot(data=df, x="Index", y=metric_train, marker="o", ax=ax, color="blue", label="Train")
                if metric_test in df.columns and not df[metric_test].isna().all():
                    sns.lineplot(data=df, x="Index", y=metric_test, marker="o", ax=ax, color="red", label="Test")
                
                ax.set_title(title)
                if ax.get_legend(): ax.legend()
                ax.grid(True)
            
            # ------------------------- 1. RMSE Plot (Oben Links) -------------------------
            plot_metric(axes[0, 0], "Avg_RMSE_Train", "Avg_RMSE_Test", "Average RMSE (Lower is better)")
            
            # ------------------------- 2. R2 Plot (Oben Rechts) -------------------------
            plot_metric(axes[0, 1], "Avg_R2_Train", "Avg_R2_Test", "Average R² (Higher is better)", higher_better=True)
            
            # ------------------------- 3. CV Loss (Unten Links) -------------------------
            ax_cv = axes[1, 0]
            if "CV_Mean_Loss (Train)" in df.columns and not df["CV_Mean_Loss (Train)"].isna().all():
                sns.lineplot(data=df, x="Index", y="CV_Mean_Loss (Train)", marker="x", linestyle="--", ax=ax_cv, label="CV Mean (Train)", color="green")
            if "Masked_CV_Loss (Train)" in df.columns and not df["Masked_CV_Loss (Train)"].isna().all():
                sns.lineplot(data=df, x="Index", y="Masked_CV_Loss (Train)", marker="x", linestyle="--", ax=ax_cv, label="Masked CV (Train)", color="orange")
                
            ax_cv.set_title("Cross-Validation Loss (Training Data Only)")
            if ax_cv.get_legend(): ax_cv.legend()
            ax_cv.grid(True)
            
            # ------------------------- 4. Leer (Unten Rechts) -------------------------
            axes[1, 1].axis("off")
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig2)
            plt.close(fig2)
            
            # ------------------------- Seite 3: Erklärung -------------------------
            fig3, ax3 = plt.subplots(figsize=(11.69, 8.27))
            ax3.axis('off')
            
            explanation_text = (
                "Metric Explanation\n\n"
                "CV Mean Loss (Train):\n"
                "Measures how well the model learns the general data structure during training.\n"
                "It includes both reconstruction error and latent space regularity (KLD).\n"
                "Lower is better.\n\n"
                "Masked CV Loss (Train):\n"
                "A stress test during training where the model must guess 20% hidden values.\n"
                "Measures pure imputation capability on known data.\n"
                "Lower is better.\n\n"
                "RMSE (Test):\n"
                "The critical metric for generalization. It measures how accurately the model fills gaps\n"
                "in data it has never seen before (Test Set).\n"
                "If this increases while Training RMSE decreases, the model is overfitting.\n"
                "Lower is better.\n\n"
                "R² Score (Test):\n"
                "Indicates how much of the variance in the unseen test data is explained\n"
                "by the model's predictions.\n"
                "Higher is better (closer to 1.0)."
            )
            
            ax3.text(0.1, 0.8, explanation_text, fontsize=14, va='top', family='monospace', linespacing=1.8)
            ax3.set_title("Metrics Definition", fontsize=18, weight='bold')
            
            pdf.savefig(fig3)
            plt.close(fig3)
            
    except Exception as e:
        print(f"Error saving PDF {pdf_path}: {e}")

def process_single_run(model_dir):
    print(f"  -> Verarbeite: {model_dir.name}")
    
    # ------------------------- 1. Metadaten lesen (Training) -------------------------
    meta_files = list(model_dir.glob("*_meta.json"))
    if not meta_files:
        print("    -> Keine Metadata gefunden. Überspringe.")
        return None
        
    meta_path = meta_files[0]
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except:
        print("    -> Fehler beim Lesen der Metadaten.")
        return None

    # ------------------------- Versuch, Run-Label aus Meta oder Dateiname zu ermitteln -------------------------
    # ------------------------- Fallback auf Default-Werte, falls nicht vorhanden -------------------------
    
    latent = meta.get("latent_dim", "?")
    hidden = meta.get("hidden_dim", "?")
    
    # ------------------------- Run-Label aus Dateinamen der .pth Dateien extrahieren -------------------------
    pth_files = list(model_dir.glob("*.pth"))
    run_label = "Unknown"
    if pth_files:
        # ------------------------- Extrahiere Label aus Dateiname, da run_label hier nicht direkt verfügbar ist -------------------------
        # ------------------------- Konstruktion aus Latent/Hidden Dimensionen -------------------------
        run_label = f"L{latent}_H{hidden}"
    
    entry = {
        "Run_Label": run_label,
        "Timestamp": model_dir.name,
        "Latent_Dim": latent,
        "Hidden_Dim": hidden,
        "CV_Mean_Loss (Train)": meta.get("cv_mean_score", None),
        "Masked_CV_Loss (Train)": meta.get("cv_masked_score", None),
    }

    # ------------------------- 2. Metriken berechnen (Inference Results) -------------------------
    # ------------------------- Pfad: 3_Machine-Learning/4_Imputation/4.2_Inference/Inference_Results/<TIMESTAMP> -------------------------
    inf_results_root = NOTEBOOK_4_2.parent / "Inference_Results" / model_dir.name
    
    train_rmse_list = []
    test_rmse_list = []
    train_r2_list = []
    test_r2_list = []
    
    if inf_results_root.exists():
         csv_files = list(inf_results_root.glob("Imputation_Results_*.csv"))
         # print(f"    -> {len(csv_files)} CSVs gefunden.")
         
         for csv_file in csv_files:
             if csv_file.stat().st_size == 0: continue
             
             try:
                 df_csv = pd.read_csv(csv_file)
                 if "Split" not in df_csv.columns: continue
                 
                 for split_name in ["Train", "Test"]:
                     subset = df_csv[df_csv["Split"] == split_name]
                     if subset.empty: continue
                     
                     y_true = subset["Original"]
                     y_pred = subset["Imputed"]
                     
                     if len(y_true) > 0:
                         mse = mean_squared_error(y_true, y_pred)
                         rmse = math.sqrt(mse)
                         r2 = r2_score(y_true, y_pred)
                         
                         if split_name == "Train":
                             train_rmse_list.append(rmse)
                             train_r2_list.append(r2)
                         else:
                             test_rmse_list.append(rmse)
                             test_r2_list.append(r2)
             except Exception:
                 pass
    else:
        print(f"    -> Kein Inference Ordner gefunden: {inf_results_root}")

    entry["Avg_RMSE_Train"] = np.mean(train_rmse_list) if train_rmse_list else None
    entry["Avg_RMSE_Test"] = np.mean(test_rmse_list) if test_rmse_list else None
    entry["Avg_R2_Train"] = np.mean(train_r2_list) if train_r2_list else None
    entry["Avg_R2_Test"] = np.mean(test_r2_list) if test_r2_list else None
    
    print(f"    -> Stats: RMSE Train={entry['Avg_RMSE_Train']:.4f}, Test={entry['Avg_RMSE_Test']:.4f}")
    
    return entry

def main():
    print("=========================================================")
    print("   RE-PLOT INCREMENTAL ERROR (Pipeline 4 Restoration)    ")
    print("=========================================================")
    
    folders = get_model_folders(limit=30)
    
    if not folders:
        print("Keine Models gefunden.")
        return

    print("\nVerfügbare Zeitstempel-Ordner (Letzte 30):")
    for i, f in enumerate(folders):
        # Versuche Meta für Details zu laden für bessere Anzeige
        info = ""
        try:
            mf = list(f.glob("*_meta.json"))[0]
            with open(mf, "r") as j:
                m = json.load(j)
                info = f"(L={m.get('latent_dim')}, H={m.get('hidden_dim')})"
        except: pass
        
        print(f"  [{i}] {f.name} {info}")
        
    print("\nAb welchem Ordner soll der Plot neu generiert werden?")
    print("Hinweis: Alle Ordner DAVOR werden ignoriert. Alle Ordner AB (inclusive) diesem Index werden geplottet.")
    
    try:
        idx_str = input(f"Bitte Index eingeben (0-{len(folders)-1}): ").strip()
        start_idx = int(idx_str)
        if start_idx < 0 or start_idx >= len(folders):
            raise ValueError
    except:
        print("Ungültige Eingabe. Abbruch.")
        return

    selected_folders = folders[start_idx:]
    print(f"\nVerarbeite {len(selected_folders)} Ordner ab Index {start_idx}...\n")
    
    # Loop
    for folder in selected_folders:
        # ------------------------- Datensatz erstellen -------------------------
        entry = process_single_run(folder)
        if entry:
            results_history.append(entry)
            
            # ------------------------- PDF am Zielort erstellen (In Evaluation Results) -------------------------
            # ------------------------- Pfad: 3_Machine-Learning/4_Imputation/4.3_Evaluation/Evaluation_Results/<TIMESTAMP> -------------------------
            eval_results_root = NOTEBOOK_4_3.parent / "Evaluation_Results" / folder.name
            
            if eval_results_root.exists():
                generate_progress_pdf(eval_results_root)
            else:
                print(f"    -> ACHTUNG: Evaluation Ordner {eval_results_root} existiert nicht. Erstelle ihn...")
                try:
                    eval_results_root.mkdir(parents=True, exist_ok=True)
                    generate_progress_pdf(eval_results_root)
                except Exception as e:
                    print(f"    -> Konnte Ordner nicht erstellen: {e}")

    print("\nFertig. Die PDFs wurden in den jeweiligen Evaluation_Results Ordnern aktualisiert.")

if __name__ == "__main__":
    main()
