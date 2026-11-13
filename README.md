# Adaptive Risk-Aware Adversarial Defense Framework

## Project
Machine Learning-Based Network Intrusion Detection System (NIDS).

This research project implements:
- XGBoost-based risk assessment.
- Adaptive Fast and Deep routing.
- Feature squeezing.
- Adversarially trained MLP.
- Network-flow intrusion detection.

## Requirements
- Windows 10 or 11
- Python 3.11
- Git
- Git LFS
- VS Code (recommended)

GPU is optional. CPU execution is supported.

## Installation

Clone the repository:

    git lfs install
    git clone https://github.com/Hayatullah166/FYP_Adaptive_Defense.git
    cd FYP_Adaptive_Defense

Create a virtual environment:

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install --upgrade pip
    pip install -r requirements.txt

## Run Dashboard

    streamlit run app_dashboard.py

## Project Folders

- experiments: 23 research experiment scripts.
- preprocessing: 11 preprocessing scripts.
- data/raw: Original network-flow dataset.
- data/processed: Cleaned and split datasets.
- data/reports: Experiment results and evaluation reports.
- models: Trained models and preprocessing artifacts.

## Dataset Input

The dashboard accepts CSV network-flow data, not raw PCAP files.

Uploaded CSV files must contain the 68 numerical features expected by the trained models. An optional binary Label column can be provided for evaluation.

## Research Reproduction

Run experiment scripts from the repository root. Check each script's input paths and configuration before execution.

Some experiments require substantial RAM, storage and processing time.

Dashboard charts may show previously generated research results. Uploading a new CSV does not retrain models or regenerate all experiment results.

## Peer Review

Please evaluate:
1. Installation and reproducibility.
2. Data preprocessing and split integrity.
3. Fast/Deep routing.
4. Clean and adversarial evaluation.
5. Model performance and efficiency.
6. Dashboard usability.

Please report errors with the script name and complete error message.

## Project Team

- Hayatullah
- Muhammad Roohullah
- Sudharth Kumar

Academic project shared for research review and feedback.

## Dataset Attribution and License

This project uses the CSE-CIC-IDS2018 dataset provided by the Canadian Institute for Cybersecurity (CIC), University of New Brunswick, in collaboration with the Communications Security Establishment (CSE).

The dataset may be redistributed under the conditions stated by its provider. Any redistribution must retain the dataset citation and official AWS source link.

**Dataset source:** https://registry.opendata.aws/cse-cic-ids2018/

**Official dataset documentation:** https://www.unb.ca/cic/datasets/ids-2018.html

**Required research citation:**

Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", ICISSP, 2018.
