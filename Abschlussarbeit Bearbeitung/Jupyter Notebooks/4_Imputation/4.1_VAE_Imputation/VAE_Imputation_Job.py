#!/usr/bin/env python
# coding: utf-8

# # 4.1 VAE Imputation: Training
# 
# <div class="alert alert-info">
# 
# <strong>Ziel</strong><br>
# Training von Variational Autoencoders (VAE) auf <strong>vollständigen Datenzeilen</strong>.
# 
# Es werden wie bereits zuvor in 3.2 verschiedene Feature-Kombinationen durchiteriert
# <br><br>
# <strong>Vorgehen</strong>
# <ol>
#     <li><strong>Daten laden</strong>: Aus dem neuesten Preprocessing-Ordner (3.1)</li>
#     <li><strong>Feature-Auswahl</strong>: Iteration über Basis-Features + Optionale Features (bis zu 30 Kombinationen aktuell, anpassbar)</li>
#     <li><strong>Filterung</strong>: Nutzung ausschließlich vollständiger Datensätze für die gewählten Spalten</li>
#     <li><strong>Training</strong>: Ein VAE pro Kombination wird trainiert und gespeichert</li>
#     <li><strong>Output</strong>: Gespeicherte Modelle im <code>Models/</code> Unterordner für spätere Inferenz (4.2)</li>
# </ol>
# </div>

# In[1]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools
import joblib

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from pathlib import Path
from datetime import datetime

# Reproduzierbarkeit
torch.manual_seed(42)
np.random.seed(42)
sns.set_theme(style="whitegrid")

# ------------------------- Device Configuration -------------------------
# ------------------------- Resource Optimization -------------------------
import os
# Limit CPU threads to avoid 100% load (leave ~20% for OS)
cpu_limit = max(1, int(os.cpu_count() * 0.80))
torch.set_num_threads(cpu_limit)
print(f'Limiting PyTorch to {cpu_limit} threads (max 80% of {os.cpu_count()} cores)')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if device.type == 'cuda':
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True


# In[2]:


# ------------------------- Konfiguration -------------------------

FIXED_BASE_FEATURES = [
    "Na_in_mmol/L",
    "Mg_in_mmol/L",
    "Ca_in_mmol/L",
    "Cl_in_mmol/L",
    "SO4_in_mmol/L",
    "HCO3_in_mmol/L",
]

OPTIONAL_FEATURES_POOL = [
    # ------------------------- Physikalische Parameter -------------------------
    "temperature_in_c",
    "pH",
    "electrical_conductivity_25c_in_uS/cm",
    "redox_potential_in_mV",
    "total_dissolved_solids_in_mmol/L",

    # ------------------------- Weitere Ionen / Spurenelemente -------------------------
    "K_in_mmol/L",
    "NO3_in_mmol/L",
    "Li_in_mmol/L",
    "Fe_in_mmol/L",
    "Mn_in_mmol/L",
    "HS_in_mmol/L",
    "O2_in_mmol/L",
    "Sr_in_umol/L",
    "NH4_in_umol/L",
    "F_in_umol/L",
    "H2SiO3_in_umol/L",
    
    # ------------------------- Metadaten (nur numerisch nutzen) -------------------------
    "depth_bgl_in_m",
    # "stratigraphic_period",
    # "rock_type"
]


# <div class="alert alert-info">
# 
# <strong>Parameter</strong><br>
# <br><br>
# <strong>Erklärung</strong>
# <ol>
#     <li><strong>MAX_ITERATIONS</strong>: Anzahl unabhängiger Trainingsläufe, sortiert nach Anzahl' kompletter Datenzeilen</li>
#     <li><strong>BATCH_SIZE</strong>: Batch-Größe für das Training</li>
#     <li><strong>EPOCHS</strong>: Anzahl der Training-Epochen pro VAE</li>
#     <li><strong>LATENT_DIM</strong>: Dimension des latenten Raums</li>
#     <li><strong>HIDDEN_DIM</strong>: Dimension der versteckten Layer</li>
#     <li><strong>KLD_WEIGHT</strong>: Gewichtung des KLD-Loss</li>
# </ol>
# </div>

# In[ ]:


# ------------------------- Hyperparameter -------------------------
MAX_ITERATIONS = 30
BATCH_SIZE = 256
EPOCHS = 150
KLD_WARMUP_EPOCHS = 50

# ------------------------- Variablen -------------------------
LATENT_DIM = 16
HIDDEN_DIM = 256
KLD_WEIGHT = 0.5 


# In[4]:


if __name__ == "__main__":
    # ------------------------- Daten Laden -------------------------
    base_dir = Path.cwd()
    
    
    # ------------------------- Suche nach dem neuesten Preprocessing-Ordner -------------------------
    preprocessing_root = base_dir.parent.parent / "3_Machine-Learning" / "3.1_Preprocessing" / "Preprocessing"
    timestamp_folders = [f for f in preprocessing_root.iterdir() if f.is_dir()]
    if not timestamp_folders:
        raise FileNotFoundError(f"Keine Preprocessing-Ordner in {preprocessing_root} gefunden.")
    
    latest_folder = max(timestamp_folders, key=lambda f: f.stat().st_mtime)
    print(f"Verwendeter Preprocessing-Ordner: {latest_folder.name}")
    
    input_path_prep = latest_folder / "Preprocessed_SOM_Ready.csv"
    df_full = pd.read_csv(input_path_prep, low_memory=False)
    print(f"Daten geladen. Shape: {df_full.shape}")


# In[5]:


# ------------------------- Features finden -------------------------
def get_training_features(user_selection, df_columns):
    training_features = []
    mapping_report = {}
    
    for user_col in user_selection:
        found = False
        candidates = [c for c in df_columns if c.startswith(user_col) and "_gauss" in c]
        if candidates:

            # ------------------------- Priorisiere _log_gauss -------------------------
            best_match = next((c for c in candidates if "_log_gauss" in c), candidates[0])
            training_features.append(best_match)
            mapping_report[user_col] = best_match
            found = True
        else:
            if user_col in df_columns:
                training_features.append(user_col)
                mapping_report[user_col] = f"{user_col} (Raw)"
                found = True
        
        if not found:
             # ------------------------- Fallback -------------------------
             fuzzy = [c for c in df_columns if c.startswith(user_col) and "_gauss" in c]
             if fuzzy:
                 best_match = fuzzy[0]
                 training_features.append(best_match)
                 mapping_report[user_col] = f"{best_match} (Fuzzy)"
                 found = True
                 
        if not found:
             print(f"Warnung: Feature '{user_col}' nicht gefunden!")
             
    return training_features, mapping_report


# In[ ]:


# ------------------------- VAE Modell Definition -------------------------
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

# ------------------------- Verlustfunktion -------------------------
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + (beta * KLD)


# In[ ]:


if __name__ == "__main__":
    # ------------------------- Ordner für Modelle -------------------------
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    models_dir = base_dir / "Models" / current_time
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Modelle werden gespeichert in: {models_dir}")
    
    # ------------------------- Feature-Kombinationen generieren -------------------------
    combos_to_run = []
    labels = []
    
    # ------------------------- Testdurchlauf (Basis ohne pH) -------------------------
    # Leere Liste = Nur FIXED_BASE_FEATURES werden genutzt
    combos_to_run.append([]) 
    labels.append("Base_without_pH") 
    
    # ------------------------- Alle Kombinationen generieren -------------------------
    max_r = 3 
    pool = OPTIONAL_FEATURES_POOL
    
    for r in range(1, max_r + 1):
        for subset in itertools.combinations(pool, r):
            combos_to_run.append(list(subset))
            
            # ------------------------- Label generieren -------------------------
            addon_names = [n.split('_in_')[0] for n in subset]
            lbl_temp = "Plus_" + "-".join(addon_names)
            
            # ------------------------- Mapping Korrektur für Base_with_pH -------------------------
            if lbl_temp == "Plus_pH":
                lbl_temp = "Base_with_pH"
                
            labels.append(lbl_temp)
    
    print(f"Generierte Kombinationen: {len(combos_to_run)}")
    
    
    # ------------------------- Priorisierung & Sortierung -------------------------
    print("\nBerechne Datenverfügbarkeit für Sortierung...")
    
    combo_stats = []
    
    for idx, combo in enumerate(combos_to_run):
        lbl = labels[idx]
        
        # ------------------------- Hardcoded Prioritäten (äquivalent zu 3.2) -------------------------
        if lbl == "Base_without_pH":
            count = 999999999
        elif lbl == "Base_with_pH":
            count = 999999998
        elif lbl == "Plus_pH-Fe":
            count = 999999997
        elif lbl == "Plus_pH-K-Fe":
            count = 999999996
        else:
            # ------------------------- Datengestützte Sortierung (äquivalet zu 2.3) -------------------------
            current_selection = FIXED_BASE_FEATURES + list(combo)
            t_cols, _ = get_training_features(current_selection, df_full.columns)
            # ------------------------- Nur vollständige Zeilen zählen 
            count = df_full[t_cols].dropna().shape[0]
    
        combo_stats.append({
            'combo': combo,
            'label': lbl,
            'count': count
        })
    
    # ------------------------- Höchste Anzahl an Reihen zuerst -------------------------
    combo_stats.sort(key=lambda x: x['count'], reverse=True)
    
    # ------------------------- Listen neu aufbauen -------------------------
    sorted_combos = [x['combo'] for x in combo_stats]
    sorted_labels = [x['label'] for x in combo_stats]
    
    
    # ------------------------- Labels anpassen & Separatoren einfügen -------------------------
    
    # ------------------------- Umbenennung des ersten Runs -------------------------
    if len(sorted_labels) > 0 and sorted_labels[0] == "Base_without_pH":
        sorted_labels[0] = "Test-Run"
    
    # ------------------------- Seperator nach Testdurchlauf einfügen -------------------------
    if len(sorted_combos) >= 1:
        sorted_combos.insert(1, ["SEPARATOR"])
        sorted_labels.insert(1, "Basis_Durchlaeufe")
    
    # ------------------------- Seperator nach Basisdurchläufen einfügen -------------------------
    if len(sorted_combos) >= 5:
         sorted_combos.insert(5, ["SEPARATOR"])
         sorted_labels.insert(5, "Zusatzinformationen")
    
    
    # ------------------------- Ausführungs Loop -------------------------
    model_counter = 0
    history_log = []
    
    print(f"Start Training Loop (Max Models: {MAX_ITERATIONS})...\n")
    
    for i, combo in enumerate(sorted_combos):
        
        # ------------------------- Abbruchbedingung -------------------------
        if model_counter >= MAX_ITERATIONS:
            print(f"Limit von {MAX_ITERATIONS} Modellen erreicht.")
            break
    
        run_label = sorted_labels[i]
        run_id = f"{i:03d}_{run_label}"
        
        # ------------------------- Separator überprüfen -------------------------
        if combo == ["SEPARATOR"]:
            print(f"--- {run_id} (Separator) ---")
            sep_path = models_dir / f"Model_{run_id}_Trennlinie.txt"
            with open(sep_path, "w") as f:
                f.write(f"Trennlinie: {run_label}")
            continue
            
        print(f"--- Start Run: {run_id} ---")
    
        # ------------------------- Features zusammenstellen -------------------------
        current_selection = FIXED_BASE_FEATURES + list(combo)
        
        # ------------------------- Mapping auf Spalten -------------------------
        train_cols, mapping = get_training_features(current_selection, df_full.columns)
        
        # ------------------------- Daten Filtern (Nur vollständige Zeilen) -------------------------
        df_run = df_full[train_cols].dropna().copy()
        
        n_samples = len(df_run)
        
        # ------------------------- Check nach zu wenig Daten -------------------------
        if n_samples < 50:
            print(f"[{run_id}] Zu wenige Daten ({n_samples}). Skip.")
            continue
            
        print(f"Features: {len(train_cols)} | Samples: {n_samples}")
        print(f"Combo: {combo}")
    
    
        # ------------------------- Train / Test Split (10% Test) -------------------------
        X_full = df_run.values
        indices_full = df_run.index.values
        
        X_train_raw, X_test_raw, idx_train, idx_test = train_test_split(
            X_full, indices_full, test_size=0.1, random_state=42
        )
        
        # ------------------------- Skalierung (Fit nur auf Train!) -------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        
        # ------------------------- Cross Validation (5-Fold) auf Train Datensatz -------------------------
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        print("  Run CV...", end="")
        for fold, (train_idx_cv, val_idx_cv) in enumerate(kf.split(X_train_scaled)):
            # ------------------------- Train / Validation Split -------------------------
            X_fold_train = X_train_scaled[train_idx_cv]
            X_fold_val = X_train_scaled[val_idx_cv]
            
            # ------------------------- Optimized DataLoader -------------------------
            loader_kwargs = {
                'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': min(4, os.cpu_count() // 2),
                'pin_memory': True if device.type == 'cuda' else False,
                'persistent_workers': True if device.type == 'cuda' and min(4, os.cpu_count() // 2) > 0 else False
            }
            # ------------------------- DataLoader -------------------------
            fold_loader = DataLoader(TensorDataset(torch.from_numpy(X_fold_train).float()), **loader_kwargs)
            
            # ------------------------- Modell Initialisierung -------------------------
            input_dim = len(train_cols)
            vae_cv = VAE(input_dim=input_dim, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM).to(device)
            opt_cv = optim.Adam(vae_cv.parameters(), lr=1e-3)
            scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda')) # AMP Scaler
            
            # ------------------------- Training -------------------------
            vae_cv.train()
            for epoch in range(EPOCHS):
                if (epoch + 1) % 50 == 0:
                    print(f"  [CV Fold {fold+1}/5] Epoch {epoch+1}/{EPOCHS}")
                
                if epoch < KLD_WARMUP_EPOCHS: current_beta = KLD_WEIGHT
                else: current_beta = KLD_WEIGHT
                
                for batch in fold_loader:
                    x_batch = batch[0].to(device)
                    opt_cv.zero_grad()
                    
                    # AMP Context
                    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                        recon_x, mu, logvar = vae_cv(x_batch)
                        loss = loss_function(recon_x, x_batch, mu, logvar, beta=current_beta)
                    
                    scaler_amp.scale(loss).backward()
                    scaler_amp.step(opt_cv)
                    scaler_amp.update()
            
            # ------------------------- Validierung -------------------------
            vae_cv.eval()
            with torch.no_grad():
                x_val = torch.from_numpy(X_fold_val).float().to(device)
                
                # AMP Context für Val
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    r_val, m_val, l_val = vae_cv(x_val)
                    # ------------------------- CV Loss -------------------------
                    val_loss = loss_function(r_val, x_val, m_val, l_val, beta=KLD_WEIGHT).item() / len(x_val)
                
                cv_scores.append(val_loss)
                
        avg_cv_loss = np.mean(cv_scores)
        print(f" Done. Avg CV Loss: {avg_cv_loss:.4f}")
        
        # ------------------------- Optimized DataLoader -------------------------
        loader_kwargs = {
            'batch_size': BATCH_SIZE,
            'shuffle': True,
            'num_workers': min(4, os.cpu_count() // 2),
            'pin_memory': True if device.type == 'cuda' else False,
            'persistent_workers': True if device.type == 'cuda' and min(4, os.cpu_count() // 2) > 0 else False
        }
        # ------------------------- Finales Training auf ganzem Train-Set -------------------------
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_scaled).float()), **loader_kwargs)
        
    
        # ------------------------- Modell Initialisierung -------------------------
        input_dim = len(train_cols)
        vae = VAE(input_dim=input_dim, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=1e-3)
        scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda')) # AMP Scaler
    
        # ------------------------- Training -------------------------
        vae.train()
        final_loss = 0
        for epoch in range(EPOCHS):
            # ------------------------- KLD Aufwärmen -------------------------
            if epoch < KLD_WARMUP_EPOCHS:
                current_beta = KLD_WEIGHT
            else:
                current_beta = KLD_WEIGHT
                
            epoch_loss = 0
            for batch in train_loader:
                x_batch = batch[0].to(device)
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    recon_x, mu, logvar = vae(x_batch)
                    loss = loss_function(recon_x, x_batch, mu, logvar, beta=current_beta)
                
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 50 == 0:
                print(f"{epoch + 1} von {EPOCHS} Epochen durchgeführt")
                
            if epoch == EPOCHS - 1:
                final_loss = epoch_loss / n_samples
                
        print(f"  -> Final Loss: {final_loss:.4f}\n")
        
        # ------------------------- Modelle Zählen -------------------------
        model_counter += 1
    
        # ------------------------- Speichern -------------------------
        safe_label = run_label.replace("/", "_").replace("\\", "_")
        
        model_path = models_dir / f"Model_{i:03d}_{safe_label}_vae.pth"
        scaler_path = models_dir / f"Model_{i:03d}_{safe_label}_scaler.joblib"
        meta_path = models_dir / f"Model_{i:03d}_{safe_label}_meta.json"
        
        torch.save(vae.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)
        
    
        # ------------------------- Metadaten speichern für Inferenz -------------------------
        metadata = {
            "run_id": run_id,
            "run_index": i,
            "label": run_label,
            "features_user": current_selection,
            "features_mapped": train_cols,
            "mapping_report": mapping,
            "n_samples": n_samples,
            "final_loss": final_loss,
            "latent_dim": LATENT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "cv_score": avg_cv_loss,
            "test_indices": idx_test.tolist(),
        }
        import json
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        history_log.append(metadata)
    
    print("\nTraining beendet")


# In[8]:


# In[8]:


if __name__ == "__main__":
    
    # ------------------------- SIGNAL DONE -------------------------
    signal_file = models_dir / "DONE_TRAINING"
    signal_file.touch()
    print(f"Signal file created: {signal_file.name}")
