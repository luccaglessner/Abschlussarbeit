#!/usr/bin/env python
# coding: utf-8

# # 4.2 VAE Imputation: Inferenz & Anwendung
# 
# <div class="alert alert-info">
# 
# ## Ziel
# Anwendung der trainierten VAE-Modelle aus 4.1.
# Hier wird die "Masking"-Strategie angewandt: Ein Feature wird aus den **vollständigen Daten** künstlich entfernt, und der VAE versucht, es zu rekonstruieren.
# 
# ## Vorgehen
# 1. **Modelle laden**: Den neuesten Run-Ordner aus 4.1 identifizieren.
# 2. **Iteration**: Für jedes trainierte Modell (Run_001 bis Run_030):
#     - Laden von Modell, Scaler und Metadaten.
#     - Laden der entsprechenden vollständigen Daten.
# 3. **Masking & Imputation**:
#     - Für jedes Feature wird eine Kopie des Datensatzes erstellt.
#     - Das Feature wird auf 0 (oder Mean) gesetzt (maskiert).
#     - Das Modell rekonstruiert den Input.
#     - Der rekonstruierte Wert für das maskierte Feature wird gespeichert.
# 4. **Output**: Eine CSV-Datei pro Run mit `Original` und `Imputed` Werten für die Evaluation in 4.3.
# </div>

# In[ ]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import os

from pathlib import Path
from datetime import datetime

# ------------------------- Reproduzierbarkeit -------------------------
torch.manual_seed(42)
np.random.seed(42)

# ------------------------- Hardware- und Ressourcenoptimierung -------------------------
cpu_limit = max(1, int(os.cpu_count() * 0.80))
torch.set_num_threads(cpu_limit)
print(f'Limiting PyTorch to {cpu_limit} threads')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# In[2]:


# ------------------------- VAE Modell Definition (Höhere Kapazität) -------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, hidden_dim=512):
        super(VAE, self).__init__()
        
        # ------------------------- Encoder -------------------------
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # ------------------------- Decoder -------------------------
        self.dec1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.dec2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.dec3 = nn.Linear(hidden_dim, input_dim)
        
        self.leaky_relu = nn.LeakyReLU(0.2)

    def encode(self, x):
        x = self.leaky_relu(self.enc1(x))
        x = self.leaky_relu(self.enc2(x))
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = self.leaky_relu(self.dec1(z))
        z = self.leaky_relu(self.dec2(z))
        return self.dec3(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# In[3]:


# ------------------------- Pfade finden (Robust + Log) -------------------------
import time
import sys
from pathlib import Path


def log(msg): print(msg)
log("Starte Inferenznotebook:")

base_dir = Path.cwd()
models_root = base_dir.parent / "4.1_VAE_Imputation" / "Models"

log(f"Base Dir: {base_dir}")
log(f"Suche Modelle in: {models_root}")

# ------------------------- Auf Ordner-Generation warten -------------------------
latest_model_folder = None
wait_counter = 0
max_wait_seconds = 600 # ------------------------- 10 Minuten warten auf 4.1 (gab Probleme) -------------------------

while latest_model_folder is None:
    if not models_root.exists():
        log("Modell fehlt. Warten...")
        time.sleep(2)
        continue
        
    timestamp_folders = [f for f in models_root.iterdir() if f.is_dir()]
    if timestamp_folders:
        # ------------------------- Neusten Zeitstempel Ordner finden -------------------------
        candidate = max(timestamp_folders, key=lambda f: f.stat().st_mtime)
        age = time.time() - candidate.stat().st_mtime
        log(f"Candidate: {candidate.name} Age: {age:.1f}s")
        
        # ------------------------- Check: Darf nicht älter als 1h sein (gab Probleme damit) -------------------------
        if age < 3600:
             latest_model_folder = candidate
             log(f"Nutze Modelle aus: {latest_model_folder.name}")
             break
        else:
             log("Krandidat zu alt. Bitte erneut versuchen")
             
    time.sleep(5)
    wait_counter += 5
    if wait_counter % 30 == 0:
        log(f"Warte auf Modell-Ordner... ({wait_counter}s)")
        
    if wait_counter >= max_wait_seconds:
        log("Timeout!")
        raise TimeoutError("Keine neuen Modelle von 4.1 in 10 Minuten gefunden.")

# ------------------------- Datenquelle (Preprocessing 3.1) -------------------------
log("Lade Daten...")
try:
    preprocessing_root = base_dir.parent.parent / "3_Machine-Learning" / "3.1_Preprocessing" / "Preprocessing"
    latest_prep_folder = max([f for f in preprocessing_root.iterdir() if f.is_dir()], key=lambda f: f.stat().st_mtime)
    input_path_prep = latest_prep_folder / "Preprocessed_SOM_Ready.csv"
    df_full = pd.read_csv(input_path_prep, low_memory=False)
    log(f"Daten geladen: {latest_prep_folder.name}")
except Exception as e:
    log(f"Fehler beim Laden der Daten: {e}")
    raise e


# In[ ]:


# ------------------------- Monitoring einrichten -------------------------
signal_file_path = latest_model_folder / "DONE_TRAINING"
# ------------------------- Inferenz Monitoring Loop -------------------------
import time

log("Starte Monitoring auf neue Modelle...")
log(f"Modell-Ordner: {latest_model_folder.name}")
log(f"Signal-File erwartet: {signal_file_path.name}")

# ------------------------- Vorbereitung Output  -------------------------
out_dir_inference = base_dir / "Inference_Results" / latest_model_folder.name
out_dir_inference.mkdir(parents=True, exist_ok=True)
log(f"Speichere Ergebnisse in: {out_dir_inference}")

processed_models = set()
done_training = False

while True:
    
    # ------------------------- Neue Dateien suchen -------------------------
    current_files = list(latest_model_folder.glob("*_vae.pth"))
    new_files = [f for f in current_files if f.name not in processed_models]
    
    # ------------------------- Sortieren für geordnete Abarbeitung -------------------------
    def get_model_index(p):
        name = p.stem
        parts = name.split("_")
        for part in parts:
             if part.isdigit(): return int(part)
        return 99999
        
    new_files.sort(key=get_model_index)
    
    # ------------------------- Abarbeiten -------------------------
    for model_path in new_files:
        
        # ------------------------- Metadaten prüfen -------------------------
        meta_path = model_path.with_name(model_path.name.replace("_vae.pth", "_meta.json"))
        scaler_path = model_path.with_name(model_path.name.replace("_vae.pth", "_scaler.joblib"))
        
        # ------------------------- Warten bis Metadata auch wirklich da ist -------------------------
        retries = 5
        while not meta_path.exists() and retries > 0:
            time.sleep(1)
            retries -= 1
            
        if not meta_path.exists():
            log(f"Überspringe {model_path.name}: Metadaten fehlerhaft")
            continue

        # ------------------------- Verarbeitung -------------------------
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            
            run_id = meta.get("run_id", "Unknown")
            features_mapped = meta["features_mapped"]
            test_indices_list = meta.get("test_indices", [])
            test_indices_set = set(test_indices_list)
            
            log(f"Processing {model_path.name} | ID: {run_id}")
            
            # ------------------------- Modell Laden -------------------------
            input_dim = len(features_mapped)
            latent_dim = meta["latent_dim"]
            hidden_dim = meta["hidden_dim"]
            
            vae = VAE(input_dim, latent_dim, hidden_dim)
            vae.load_state_dict(torch.load(model_path))
            vae.to(device)
            vae.eval()
            
            scaler = joblib.load(scaler_path)
            
            # ------------------------- Daten Subset -------------------------
            data_subset = df_full[features_mapped].dropna()
            
            if len(data_subset) > 10:
                X_original = data_subset.values
                X_scaled = scaler.transform(X_original)
                
                results_list = []
                for f_idx, feature_name in enumerate(features_mapped):
                    # ------------------------- Maskierung für einzelne Spalte -------------------------
                    X_masked = X_scaled.copy()
                    X_masked[:, f_idx] = 0.0
                    
                    with torch.no_grad():
                        X_tensor = torch.from_numpy(X_masked).float().to(device)
                        recon_x, _, _ = vae(X_tensor)
                        X_recon_scaled = recon_x.cpu().numpy()
                        
                    X_recon_original = scaler.inverse_transform(X_recon_scaled)
                    
                    original_vals = X_original[:, f_idx]
                    subset_indices = data_subset.index.values
                    imputed_vals = X_recon_original[:, f_idx]
                    
                    for idx_val, val_orig, val_imp in zip(subset_indices, original_vals, imputed_vals):
                        results_list.append({
                            "Feature": feature_name,
                            "Original": val_orig,
                            "Imputed": val_imp,
                            "Run_ID": run_id,
                            "Split": "Test" if idx_val in test_indices_set else "Train"
                        })
                
                # ------------------------- Speichern -------------------------
                df_res = pd.DataFrame(results_list)
                safe_name = model_path.stem.replace("_vae", "")
                out_csv = out_dir_inference / f"Imputation_Results_{safe_name}.csv"
                df_res.to_csv(out_csv, index=False)
                
            else:
                 log(f"  Zu wenige Samples für Validierung ({len(data_subset)}).")

            # ------------------------- Als fertig markieren -------------------------
            processed_models.add(model_path.name)
            
        except Exception as e:
            log(f"Fehler bei {model_path.name}: {e}")
            processed_models.add(model_path.name)
            
    # ------------------------- Abbruchbedingung prüfen -------------------------
    if signal_file_path.exists():
        if not done_training:
             log("Signal DONE_TRAINING erkannt. Verarbeite restliche Dateien...")
             done_training = True
             
        # ------------------------- Überprüfen ob alle Modelle verarbeitet worden sind -------------------------
        if not new_files:
             current_check = list(latest_model_folder.glob("*_vae.pth"))
             remaining = [f for f in current_check if f.name not in processed_models]
             if not remaining:
                 log("Alle Modelle verarbeitet. Beende Monitoring.")
                 break
    
    # ------------------------- Warten -------------------------
    time.sleep(2)
