# Hydrochemical Data Analysis and Machine Learning Pipeline

This repository contains the complete codebase and research pipeline developed for my Bachelor Thesis. The project focuses on the automated acquisition, analysis, and processing of hydrochemical data using modern Machine Learning techniques like Self-Organizing Maps (SOM) and Variational Autoencoders (VAE).

## Project Overview

The core objective is to create a robust workflow for handling large-scale groundwater monitoring data, from initial scraping to advanced clustering and missing value imputation.

## Repository Structure

The project is structured into logical phases, mirroring the research workflow:

### 1. Data Acquisition (`1_Acquisition`)
Responsible for harvesting raw data. It includes specialized wrappers and scraping scripts to pull hydrochemical parameters from online databases and official monitoring stations.

### 2. Exploratory Data Analysis (`2_Analysis`)
A deep dive into the dataset. This section covers:
-   **Statistical EDA**: General data distribution and completeness checks.
-   **Domain-Specific Analysis**: Detailed studies on rock-type influence, temperature gradients, and ionic balance errors (IBE).
-   **Data Quality**: Automated filtering and cleaning based on hydrochemical consistency (e.g., IBE < 5%).

### 3. Machine Learning & Clustering (`3_Machine-Learning`)
Implementation of unsupervised learning methods:
-   **Preprocessing**: Normalization, log-scaling, and Gaussian transformations.
-   **MiniSom**: Training Self-Organizing Maps to identify hydrochemical facies and spatial patterns.
-   **Profiling**: Automated generation of cluster profiles and characteristic reports.

### 4. Data Imputation (`4_Imputation`)
Advanced handling of missing data using deep learning:
-   **VAE Architecture**: A Variational Autoencoder designed to learn the latent distribution of groundwater chemistry.
-   **Beta-VAE**: Optimized Beta-scheduling for better disentanglement and reconstruction.
-   **Comparison**: Benchmarking VAE results against traditional statistical methods.

### 5. Validation and Inference (`5_kNN`)
Final evaluation using k-Nearest Neighbors (kNN) and other inference models to validate the learned representations and the quality of the imputed data.

---

## 🚀 How to Run

The project uses an automated pipeline system located in `Pipeline_Scripts/`.

-   **`run_pipeline_All.py`**: The main entry point that orchestrates the entire workflow.
-   **`run_pipeline_2_3.py`**: Focuses on Analysis and SOM training.
-   **`run_pipeline_4-5.py`**: Handles VAE Imputation and subsequent validation.

### Requirements
-   Python 3.x
-   Jupyter Notebook
-   Core libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `minisom`, `pytorch` / `tensorflow`.

## Results
The pipeline automatically generates detailed PDF reports for each stage, located in their respective `*_Results/` directories. These reports provide visual insights into cluster stability, chemistry profiles, and imputation accuracy.

---
*Created as part of the Bachelor Thesis research by Lucca Glessner.*
