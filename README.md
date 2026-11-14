# Adaptive Risk-Aware Adversarial Defense Framework

A research implementation of an adaptive, risk-aware defense framework for a machine-learning-based Network Intrusion Detection System (NIDS).

## Project Overview

The framework combines a risk-assessment model with adaptive routing:

- **Risk assessment:** an XGBoost model estimates the risk of each network-flow record.
- **Fast path:** records classified as benign with low uncertainty can bypass the deeper model.
- **Deep path:** records requiring further inspection are processed using feature squeezing and an adversarially trained MLP.
- **Input format:** network-flow CSV files containing the model's 68 expected numerical features.

This repository is intended for academic research, reproducibility checks, and peer review. It is a CSV-based workflow; it does not directly capture live network traffic or accept raw PCAP files in the dashboard.

## Requirements

- Windows 10 or 11
- Python 3.11
- Git
- Git LFS
- VS Code (recommended)

The dashboard is configured to run on CPU. A GPU is not required.

## Installation

Open PowerShell and run these commands from the directory where you want the project:

```powershell
git lfs install
git clone https://github.com/Hayatullah166/FYP_Adaptive_Defense_Review.git
cd FYP_Adaptive_Defense_Review

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, run the following command in the same PowerShell window, then activate the environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Git LFS is required because the repository contains large datasets, model artifacts, and an archive. After cloning, allow Git LFS to finish downloading the tracked files.

## Run the Dashboard

From the repository root, with the virtual environment activated:

```powershell
streamlit run app_dashboard.py
```

Streamlit will print a local address in the terminal (typically `http://localhost:8501`; the port may differ if it is already in use).

## Repository Structure

- `experiments/` — research experiment scripts (23 scripts).
- `preprocessing/` — data-preparation scripts (11 scripts), including the `cicids2018_raw/` directory.
- `data/processed/` — processed datasets and train/validation/test splits.
- `data/processed/grouped/` — grouped train, validation, and test datasets, including model-specific feature representations.
- `models/` — trained model artifacts.
- `app_dashboard.py` — Streamlit dashboard.
- `requirements.txt` — pinned Python dependencies.
- `FYP_Adaptive_Defense.zip` — project archive.

## Dashboard Model Artifacts

The dashboard expects the following artifacts at these repository-relative paths:

- `models/risk_assessment/xgboost_risk_model.joblib`
- `models/risk_assessment/risk_model_feature_names.joblib`
- `models/mlp/mlp_adversarial_best.pth`
- `data/processed/grouped/models/standard_scaler_grouped.joblib`

Keep the directory structure intact when running the dashboard.

## Dataset Input

The dashboard accepts network-flow CSV files, not raw PCAP files.

The uploaded CSV must contain all **68 numerical input features** expected by the trained models. An optional binary `Label` column (0 or 1) can be included for evaluation. The dashboard validates the required features and their values before inference.

Do not remove, rename, or reorder feature columns in a way that changes their meaning. Use the feature definitions and preprocessing scripts in this repository as the reference.

## Research Reproduction and Evaluation

Run experiment scripts from the repository root unless a script's instructions specify otherwise. Review each script's configuration and input paths before execution.

Some experiments may require substantial RAM, disk space, and processing time. Large datasets and model artifacts are tracked using Git LFS.

The dashboard may display previously generated experiment results. Uploading a new CSV performs inference on that input; it does not retrain the models or automatically regenerate every research experiment.

For a meaningful peer review, please distinguish between:
- results reproduced by running the experiment scripts;
- results displayed from previously generated reports or dashboard artifacts; and
- results obtained by uploading a new CSV for inference.

## Peer Review Checklist

Please review the following areas:

1. Installation and reproducibility.
2. Data preprocessing and train/validation/test split integrity.
3. Risk assessment and Fast/Deep routing behavior.
4. Clean and adversarial evaluation methodology.
5. Model performance, resource use, and efficiency.
6. Dashboard functionality and usability.

When reporting an issue, please include the script or file name, the command you ran, and the complete error message. Please also mention your Python version and operating system.

## Project Team

- Hayatullah
- Muhammad Roohullah
- Sudharth Kumar

Academic project shared for research review and feedback.

## Dataset Attribution

This project uses the CSE-CIC-IDS2018 dataset associated with the Canadian Institute for Cybersecurity, University of New Brunswick, and the Communications Security Establishment.

**Dataset source:** https://registry.opendata.aws/cse-cic-ids2018/

**Official dataset documentation:** https://www.unb.ca/cic/datasets/ids-2018.html

**Research citation:**

Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", ICISSP, 2018.

Please consult the dataset provider's terms and documentation before redistributing dataset files.
