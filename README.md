# Hydrochemical Data Analysis and Machine Learning Pipeline

> [!NOTE]
> The Bachelor Thesis was written in German and the project was initiated and continued in a German context. A translation of the entire repository may occur in the future, but was not part of the thesis submission as of March 9, 2026. However the ReadMe is already translated to english to enable easier introduction to this project for english users.

This repository contains the complete codebase and research pipeline developed for my Bachelor Thesis. The project focuses on the automated acquisition, analysis, and processing of hydrochemical data using modern Machine Learning techniques like Self-Organizing Maps (SOM) and Variational Autoencoders (VAE).

## Project Overview

The core objective is to create a robust workflow for handling large-scale groundwater monitoring data, from initial scraping to advanced clustering and missing value imputation.

## Repository Structure

The project is structured into logical phases, mirroring the research workflow:

### 1. Data Acquisition (`1_Acquisition`)
Responsible for harvesting raw data from hydrochemical databases.
-   **Subdirectories**: `1.1_Data-Acquisition-Wrapper`, `1.2_Pipeline_Wrapper`
-   **Results**: Raw databases are stored in `1.1_Data-Acquisition-Wrapper/Gesammelte_Datenbanken/`.
-   **Experimental Feature**: `1.2_Pipeline_Wrapper` contains an experimental tool (`Pipeline_Execution.ipynb`) designed for future publications. It allows users to easily convert their own custom datasets into the project's standardized schema. A first functional version is already implemented and includes its own detailed English [User Guide](Abschlussarbeit%20Bearbeitung/Jupyter%20Notebooks/1_Acquisition/1.2_Pipeline_Wrapper/README.md).

### 2. Exploratory Data Analysis (`2_Analysis`)
A deep dive into the hydrochemical dataset to ensure quality and understand patterns.
-   **Subdirectories**: `2.1_Explorative-Datenanalyse`, `2.2_Rock-Type_Analysis`, `2.3_Temperature_Analysis`, `2.4_Data-Quality_Ionic-Balance-Error`, `2.5_Full-Datasets-Analysis`.
-   **Results**: Cleaned datasets and IBE-filtered data can be found in `2.4_Data-Quality_Ionic-Balance-Error/` as CSV files and PDF reports.

### 3. Machine Learning & Clustering (`3_Machine-Learning`)
Implementation of unsupervised learning methods to identify hydrochemical facies.
-   **Subdirectories**: `3.1_Preprocessing`, `3.2_Machine-Learning` (MiniSom).
-   **Results**: Preprocessed datasets in `3.1_Preprocessing/Preprocessing/` and SOM clustering reports in `3.2_Machine-Learning/SOM_Results/`.

### 4. Data Imputation (`4_Imputation`)
Deep learning for missing value imputation using Variational Autoencoders.
-   **Subdirectories**: `4.1_VAE_Imputation`, `4.2_Inference`, `4.3_Evaluation`.
-   **Results**: Model weights in `4.1_VAE_Imputation/Models/`, imputed datasets in `4.2_Inference/Inference_Results/`, and evaluation reports in `4.3_Evaluation/Evaluation_Results/`.

### 5. Validation and Inference (`5_kNN`)
Comparative validation of VAE results using standard imputation techniques.
-   **Subdirectories**: `5.1_kNearest-Neighbors.ipynb`.
-   **Results**: Validation metrics and comparison results are stored in `5_kNN/Inference_Results/`.

## Data Preparation

The hydrochemical datasets required for this project are stored on Google Drive due to their size:
[Download data from Google Drive](https://drive.google.com/drive/folders/1UW6X8MoAbbR1Ye8dZCb-GDjokEJhqimu?usp=drive_link)

**Instructions:**
1. Download all files from the link above.
2. Place these files into the following directory:
   `Abschlussarbeit Bearbeitung\Jupyter Notebooks\1_Acquisition\1.1_Data-Acquisition-Wrapper\Gesammelte_Datenbanken`
3. This is required for the Data-Acquisition-Wrapper to process the data correctly.

The following 4 files (.pdfs) are stored in the "Bachelor Thesis Examples" subfolder on the Drive and represent the results used in the Bachelor's thesis:
- `SOM-DISCRIMINATION_Bachelor-Thesis.pdf`
- `SOM-REPORT_Bachelor-Thesis.pdf`
- `VAE-CLIPPING_Bachelor-Thesis.pdf`
- `VAE-NO-CLIPPING-FULL_Bachelor-Thesis.pdf`


---

## How to Run

The project uses an automated pipeline system located in `Pipeline_Scripts/`.

-   **`run_pipeline_All.py`**: The main entry point that orchestrates the entire workflow.
-   **`run_pipeline_1.py`**: Automates the Data Acquisition process.
-   **`run_pipeline_2_3.py`**: Focuses on Analysis and SOM training.
-   **`run_pipeline_4.py`**: Handles VAE Imputation, Inference and Evaluation.

### Requirements
-   **Python 3.10.8** or **3.12.10** (tested versions; other versions may cause compatibility issues)
-   Jupyter Notebook
-   All required Python packages and their exact versions are listed in `requirements.txt`. Install them using:
    ```bash
    pip install -r requirements.txt
    ```

## Results
The pipeline automatically generates detailed PDF reports for each stage, located in their respective `*_Results/` directories. These reports provide visual insights into cluster stability, chemistry profiles, and imputation accuracy.

## Reproducibility
To ensure full scientific reproducibility of the results across all model trainings and pipeline executions, a fixed **Random Seed of `42`** is used throughout the codebase.

This applies specifically to:
1.  **Self-Organizing Maps (SOM)**: To guarantee the same initial weights and data presentation order during clustering (`3.2_Machine-Learning/MiniSom/MiniSom_Machine-Learning.ipynb`).
2.  **Variational Autoencoders (VAE)**: To ensure consistent weight initialization, data shuffling, and identical sampling of the latent space via the reparameterization trick (`4.1_VAE_Imputation/VAE_Imputation.py` & `.ipynb`).

The seed is globally enforced via `numpy.random.seed(42)` and `torch.manual_seed(42)`.

