# VAE Modell-Bericht

## 1. Modell-Architektur
Das verwendete Netzwerk ist ein **Beta-Variational Autoencoder (Beta-VAE)** mit symmetrischem Encoder und Decoder.

*   **Eingabedimension (Input):** 9 Neuronen
    *   *Features:* TDS, Na, Mg, Ca, Cl, SO4, HCO3, K, Fe
*   **Encoder:**
    *   Hidden Layer 1: **256 Neuronen** (Linear + ReLU Aktivierung)
    *   Latent Space (Bottleneck): **16 Neuronen** (2x Linear für $\mu$ und $\log(\sigma^2)$)
*   **Decoder:**
    *   Hidden Layer 2: **256 Neuronen** (Linear + ReLU Aktivierung)
    *   Ausgabedimension (Output): **9 Neuronen** (Linear, keine Aktivierung am Ende, da die Daten skaliert sind)

## 2. Daten-Preprocessing
Da VAEs eine Normalverteilung im Latent Space annehmen, wurden die Eingabedaten transformiert:
*   **Imputation (Vorverarbeitung für Training):** `SimpleImputer(strategy='mean')` (Füllen von Lücken in den Trainingsdaten).
*   **Skalierung:** `QuantileTransformer(output_distribution='normal')`.
    *   Transformiert die Eingabedaten in eine Gaußsche Glockenkurve (Normalverteilung, $\mu=0, \sigma=1$). Dies eliminiert Ausreißer und optimiert die Datenverteilung für den VAE.

## 3. Training & Hyperparameter
Das Modell wurde speziell für "Data Rescue" (Imputation) optimiert.

*   **Optimizer:** Adam (`lr=1e-3`)
*   **Epochen:** 150
*   **Batch Size:** 64
*   **Verlustfunktion (Loss Function):**
    *   $Loss = MSE + \beta \cdot KLD$
    *   **MSE (Mean Squared Error):** Misst den Rekonstruktionsfehler.
    *   **KLD (Kullback-Leibler Divergenz):** Misst die Abweichung des Latent Space von der Normalverteilung (Regularisierung).
*   **Beta-Schedule:**
    *   Lineares Warmup von 0 auf **0.05** über die ersten 50 Epochen.
    *   **Max Beta:** 0.05. Dieser niedrige Wert priorisiert die **Rekonstruktionsgenauigkeit** gegenüber der Generierung neuer Daten.

## 4. Funktionsweise der Imputation
1.  Rohdaten werden initial mit dem Mittelwert gefüllt (Dummy-Fill).
2.  Der VAE projiziert die Daten in den Latent Space (Kompression auf 16 Dimensionen) und rekonstruiert sie.
3.  Durch die Kompression lernt das Modell die geochemischen Zusammenhänge.
4.  Lücken in den Originaldaten werden mit den rekonstruierten Werten überschrieben (Data Rescue).

## 5. Lern-Kategorie
Es handelt sich um **Unüberwachtes Lernen (Unsupervised Learning)** (spezifischer: Self-Supervised Learning).
*   **Grund:** Das Modell nutzt die Eingabedaten ($X$) selbst als Zielgröße (Target) für die Rekonstruktion ($X \approx \hat{X}$). Es werden keine externen Labels oder Zielvariablen benötigt.
