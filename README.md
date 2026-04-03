# **`README.md`**

# Robust Financial Risk Intelligence: Compositional Bankruptcy Prediction

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.24215-b31b1b.svg)](https://arxiv.org/abs/2603.24215)
[![Journal](https://img.shields.io/badge/Journal-ArXiv%20Preprint-003366)](https://arxiv.org/abs/2603.24215)
[![Year](https://img.shields.io/badge/Year-2026-purple)](https://github.com/chirindaopensource/adapting_altmans_bankruptcy_prediction_model)
[![Discipline](https://img.shields.io/badge/Discipline-Financial%20Econometrics%20%7C%20Corporate%20Solvency-00529B)](https://github.com/chirindaopensource/adapting_altmans_bankruptcy_prediction_model)
[![Data Sources](https://img.shields.io/badge/Data-SABI%20(Iberian%20Balance%20Sheet%20Analysis%20System)-lightgrey)](https://sabi.bvdinfo.com/)
[![Core Method](https://img.shields.io/badge/Method-Compositional%20Data%20(CoDa)%20%7C%20Aitchison%20Simplex-orange)](https://github.com/chirindaopensource/adapting_altmans_bankruptcy_prediction_model)
[![Analysis](https://img.shields.io/badge/Analysis-Machine%20Learning%20%7C%20Logit%20%7C%20k--NN%20%7C%20Random%20Forest-red)](https://github.com/chirindaopensource/adapting_altmans_bankruptcy_prediction_model)
[![Validation](https://img.shields.io/badge/Validation-Sensitivity%20%7C%20Balanced%20Accuracy-green)](https://github.com/chirindaopensource/adapting_altmans_bankruptcy_prediction_model)
[![Robustness](https://img.shields.io/badge/Robustness-No%20Outlier%20Pruning%20%7C%20Log--Ratio%20EM%20Imputation-yellow)](https://github.com/chirindaopensource/adapting_altmans_bankruptcy_prediction_model)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-%23000000.svg?style=flat)](https://www.statsmodels.org/)
[![rpy2](https://img.shields.io/badge/rpy2-R%20Bridge-198CE7.svg?style=flat&logo=R)](https://rpy2.github.io/)
[![YAML](https://img.shields.io/badge/YAML-%23CB171E.svg?style=flat&logo=yaml&logoColor=white)](https://yaml.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen)](https://github.com/chirindaopensource/adapting_altmans_bankruptcy_prediction_model)

**Repository:** `https://github.com/chirindaopensource/adapting_altmans_bankruptcy_prediction_model`

**Owner:** 2026 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2026 paper entitled **"Adapting Altman's bankruptcy prediction model to the compositional data methodology"** by:

*   **Fatemeh Keivani** (Universitat de Girona, Department of Economics)
*   **Germà Coenders** (Universitat de Girona, Department of Economics)
*   **Geòrgia Escaramís** (Universitat de Girona, Department of Economics)

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, highly optimized pipeline that executes the entire research workflow: from the rigorous ingestion of raw accounting data and the execution of the log-ratio Expectation-Maximization (EM) algorithm for zero imputation, to the geometric transformation of financial statements into the Aitchison simplex, culminating in the training and evaluation of machine learning classifiers (Logistic Regression, k-NN, Random Forests) optimized for extreme class imbalance.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `execute_robust_financial_risk_intelligence_pipeline`](#key-callable-execute_robust_financial_risk_intelligence_pipeline)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the analytical framework presented in Keivani et al. (2026). The core of this repository is the iPython Notebook `adapting_altmans_bankruptcy_prediction_model_draft.ipynb`, which contains a comprehensive suite of 20+ orchestrated tasks to replicate the paper's findings.

The pipeline addresses a critical vulnerability in classical financial econometrics: the use of standard accounting ratios (e.g., Current Assets / Current Liabilities) as predictors. These ratios inherently suffer from extreme outliers, severe asymmetry, and non-normality, often forcing researchers to arbitrarily prune up to one-third of their sample, thereby destroying representativeness.

The codebase operationalizes the proposed solution—the **Compositional Data (CoDa) methodology**:
-   **Imputes** rounded zeros in raw accounting parts using the Palarea-Albaladejo & Martín-Fernández (2008) log-ratio EM algorithm.
-   **Transforms** the $D=7$ fundamental accounting figures into scale-invariant pairwise log-ratios (plr), mapping the data from the Aitchison simplex into a mathematically stable Euclidean space.
-   **Evaluates** the predictive superiority of the CoDa features against standard Altman ratios using parametric (GLM) and non-parametric (k-NN, RF) architectures.
-   **Validates** the economic utility of the models by prioritizing **Sensitivity** (the correct identification of bankrupt firms) over naive accuracy in a heavily imbalanced dataset.

## Theoretical Background

The implemented methods combine techniques from Corporate Solvency Jurisprudence, Compositional Geometry, and Supervised Machine Learning.

**1. The Failure of Standard Ratios:**
Classical models rely on ratios that exhibit extreme kurtosis. For example, the Return on Equity ($X_9$) can approach infinity if Net Worth approaches zero:
$$ X_9 = \frac{OR - OE}{(NCA + CA) - (NCL + CL)} $$

**2. The Aitchison Simplex and Pairwise Log-Ratios (plr):**
The CoDa methodology treats the $D=7$ accounting figures as parts of a whole. Relative information is extracted via log-ratio transformations. For parametric models (Logistic Regression), a non-redundant spanning tree of $D-1=6$ log-ratios is used to prevent perfect collinearity:
$$ plr_k = \log\left(\frac{x_i}{x_j}\right) \quad \text{e.g., } \log\left(\frac{RE}{NCL}\right) $$

**3. Distance Equivalence for Non-Parametric Models:**
For distance-based methods (k-NN) and feature-selecting ensembles (Random Forests), the exhaustive set of $D(D-1)/2 = 21$ pairwise log-ratios is utilized. This guarantees that the Euclidean distance in the transformed space is strictly proportional to the **Aitchison distance** in the simplex:
$$ d_A(\mathbf{x}, \mathbf{y}) = \left( \frac{1}{2D} \sum_{i=1}^{D} \sum_{j=1}^{D} \left[ \log\left(\frac{x_i}{x_j}\right) - \log\left(\frac{y_i}{y_j}\right) \right]^2 \right)^{1/2} $$

**4. Class Imbalance Mitigation:**
Given the rarity of bankruptcy (97 out of 31,131 firms), the training set is randomly downsampled to a strict 1:1 ratio (69 bankrupt to 69 healthy firms) to optimize the classifiers for sensitivity, while the validation set remains imbalanced to reflect real-world base rates.

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/adapting_altmans_bankruptcy_prediction_model/blob/main/adapting_altmans_bankruptcy_prediction_model_ipo_main.png" alt="Robust Financial Risk Intelligence Architecture" width="100%">
</div>

## Features

The provided iPython Notebook (`adapting_altmans_bankruptcy_prediction_model_draft.ipynb`) implements the full research pipeline, including:

-   **Zero-Leakage State Machine:** The pipeline utilizes a strict state-routing architecture, ensuring that missingness audits occur *before* arithmetic operations, and that infinity-capping for standard ratios relies strictly on training-set bounds to absolutely guarantee zero look-ahead bias.
-   **R-Bridge Integration:** Implements the true Palarea-Albaladejo & Martín-Fernández (2008) log-ratio EM algorithm by bridging Python to the R `zCompositions` package via `rpy2`, ensuring exact subcompositional coherence during zero imputation.
-   **Rigorous Econometric Rounding:** Replaces standard Python Banker's rounding with `decimal.ROUND_HALF_UP` to ensure distributional diagnostics (Table 1) and performance metrics (Table 2) match strict academic reporting standards.
-   **Algorithmic Compliance Enforcement:** Overrides `scikit-learn`'s default lexicographical tie-breaking in k-NN with a custom prediction loop that strictly enforces the manuscript's "predict class 1 on ties" rule during hyperparameter tuning.
-   **Cryptographic Archival:** Serializes the entire in-memory replication dossier to disk using a custom `NumpyEncoder` to prevent type-crashes, generating SHA-256 checksums for every artifact to ensure perfect auditability.
-   **Configuration-Driven Design:** All study parameters, temporal boundaries, and optimization constraints are managed in an external `config.yaml` file, ensuring strict methodological reproducibility.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Data Engineering (Tasks 1-6):** Ingests raw SABI accounting data. Enforces strict cohort filters (Spain, NACE 46XX, employees $\ge 5$). Computes the inactivity conjunction rule to exclude non-operating firms without pruning statistical outliers.
2.  **Data Quality & Imputation (Tasks 7-8):** Audits the raw data for `NaN`s and negative values. Executes the log-ratio EM algorithm to replace rounded zeros in the $D=7$ predictor-year parts, verifying the strict positivity contract required by the Aitchison simplex.
3.  **Feature Engineering (Tasks 9-11):** Constructs the 10 standard Altman ratios and the 6/21 compositional pairwise log-ratios. Computes Fisher-Pearson skewness and excess kurtosis (Table 1) to prove the statistical superiority of the CoDa transformation.
4.  **Partitioning & Downsampling (Tasks 12-13):** Executes a deterministic seed search to match the manuscript's exact bankrupt counts. Splits the data 70/30 and performs 1:1 random downsampling on the training set.
5.  **Model Fitting (Tasks 14-17):** Trains Logistic Regression (handling expected collinearity crashes in standard ratios), tunes and fits k-NN ($k=5$), and trains Random Forests ($B=100$, $m_{try}=4/7$), extracting Mean Decrease in Gini for variable importance.
6.  **Evaluation & Audit (Tasks 18-20):** Computes the confusion matrices and populates the performance metrics (Table 2). Algorithmically evaluates fidelity to the manuscript's central claims and archives the reproducibility package.

## Core Components (Notebook Structure)

The notebook is structured as a logical pipeline with modular orchestrator functions for each of the 20 major tasks. All functions are self-contained, fully documented with strict type hints and comprehensive docstrings, and designed for professional-grade execution.

## Key Callable: `execute_robust_financial_risk_intelligence_pipeline`

The project is designed around a single, top-level user-facing interface function:

-   **`execute_robust_financial_risk_intelligence_pipeline`:** This apex orchestrator function runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project, managing data validation, EM imputation, feature engineering, model training, metric evaluation, and the final cryptographic fidelity audit.

## Prerequisites

-   Python 3.10+
-   R 4.0+ (Required for `zCompositions` package)
-   Core Python dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `rpy2`, `pyyaml`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/adapting_altmans_bankruptcy_prediction_model.git
    cd adapting_altmans_bankruptcy_prediction_model
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy scipy scikit-learn statsmodels rpy2 pyyaml faker
    ```

4.  **Install R dependencies (Run within an R console):**
    ```R
    install.packages("zCompositions")
    ```

## Input Data Structure

The pipeline requires a single primary data structure, strictly validated at runtime:

1.  **`df_raw` (pd.DataFrame):** Contains the raw accounting figures and metadata extracted from the SABI database. It must contain one row per firm, with two consecutive annual snapshots (year $t$ and year $t-1$) to support the inactivity exclusion rule.
    *   **Metadata:** `firm_id`, `nace_code`, `employees`, `label_year`, `feature_year`, `status_t` (1=Bankrupt, 0=Healthy).
    *   **Accounting Parts ($t$ and $t-1$):** `NCA`, `CA`, `RE`, `NCL`, `CL`, `OR`, `OE`.

*Note: The pipeline includes a high-fidelity synthetic data generator for testing purposes if access to the proprietary SABI database is unavailable.*

## Usage

The notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell, which demonstrates how to load the configuration, generate synthetic data with exact pathological zero-densities, and use the top-level orchestrator to execute the pipeline:

```python
import os
import yaml
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

# 1. Load the master configuration from the YAML file.
# (Assumes config.yaml is in the working directory)
def load_study_configuration(filepath: str = "config.yaml") -> Dict[str, Any]:
    if not isinstance(filepath, str):
        raise TypeError(f"filepath must be a string, got {type(filepath)}.")
    try:
        with open(filepath, "r") as file:
            config = yaml.safe_load(file)
        print(f"Successfully loaded configuration from {filepath}")
        return config
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: {filepath} not found in the working directory.")
        raise e

raw_config = load_study_configuration("config.yaml")

# 2. Load raw datasets (Example using synthetic generator provided in the notebook)
# In production, load from CSV/Parquet: pd.read_parquet("data/sabi_extract_raw.parquet")
# Note: The synthetic generator (defined in the notebook) injects exact zero densities 
# (e.g., 22.58% for NCL) to rigorously test the log-ratio EM imputation.
df_raw = generate_synthetic_sabi_data(n_total=31131, n_bankrupt=97)

# 3. Execute the entire replication study.
if __name__ == "__main__":
    output_directory = "./replication_output_dossier"
    
    if not df_raw.empty and raw_config:
        print("\nInitiating Robust Financial Risk Intelligence Pipeline...")
        
        pipeline_results = execute_robust_financial_risk_intelligence_pipeline(
            df_raw=df_raw,
            raw_config=raw_config,
            output_dir_str=output_directory
        )
        
        # 4. Access results
        if pipeline_results["status"] == "Success":
            print("\n" + "="*80)
            print("STUDY EXECUTION COMPLETE: ARTIFACTS SECURED")
            print("="*80)
            
            dossier = pipeline_results["dossier"]
            
            print("\n[Table 2: Predictive Performance (Rounded %)]")
            print(dossier["evaluation"]["table2_rounded_pct"].to_string())
            
            print("\n[Compositional Random Forest: Top 3 Predictors]")
            rf_comp_importance = dossier["model_results"]["random_forest"]["variable_importance"]["compositional_rf"]
            print(rf_comp_importance.head(3).to_string(index=False))
            
            print("\n" + "="*80)
            print("FINAL FIDELITY AUDIT VERDICT")
            print("="*80)
            print(f"Verdict: {dossier['evaluation']['fidelity_assessment']['overall_fidelity_verdict']}")
            
            print(f"\nCryptographically verified archive saved to: {pipeline_results['archive_path']}")
```

## Output Structure

The pipeline returns a master dictionary (`ReplicationDossier`) and serializes it to a `.tar.gz` archive containing:
-   **`00_config/`**: The frozen configuration and toolchain manifest.
-   **`01_data_lineage/`**: Audit trails for cohort filtering, inactivity exclusion, and the pre/post-imputation Parquet matrices.
-   **`02_features/`**: The materialized standard (10), plr6, and plr21 feature matrices.
-   **`03_diagnostics/`**: Table 1 (Skewness/Kurtosis) and the counterfactual pruning burden report.
-   **`04_partitioning/`**: The frozen, deterministic split indices for training, validation, and downsampling.
-   **`05_models/`**: Coefficient tables, tuning results, and Random Forest variable importance rankings.
-   **`06_evaluation/`**: Table 2 (Performance Metrics), confusion matrices, and the algorithmic fidelity assessment.
-   **`07_audit/`**: The ambiguity register and the SHA-256 cryptographic checksums for all artifacts.

## Project Structure

```
adapting_altmans_bankruptcy_prediction_model/
│
├── adapting_altmans_bankruptcy_prediction_model_draft.ipynb    # Main implementation notebook
├── config.yaml                                                 # Master configuration file
├── requirements.txt                                            # Python package dependencies
│
├── LICENSE                                                     # MIT Project License File
└── README.md                                                   # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify study parameters such as:
-   **Cohort Definition:** Adjust the NACE prefix, employee thresholds, or temporal lag.
-   **Sampling:** Modify the train/validation split ratio or the downsampling ratio (e.g., from 1:1 to 2:1).
-   **Hyperparameters:** Alter the number of trees ($B$), the $m_{try}$ parameter, or the $k$-grid for nearest neighbors.
-   **Evaluation:** Change the classification threshold for the logistic regression or prioritize a different metric (e.g., F1-Score) over Sensitivity.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, strict type hinting, and the 1:1 inline comment-to-code-line ratio is required.

## Recommended Extensions

Future extensions, as suggested by the authors, could include:
-   **Advanced Deep Learning Architectures:** Adapting the CoDa feature space for Neural Networks, Deep Learning, or hybrid CNN-LSTM-GRU models.
-   **Gradient Boosting:** Implementing XGBoost, LightGBM, or CatBoost, which may offer superior handling of the compositional feature space compared to standard Random Forests.
-   **Explainable AI (XAI):** Integrating SHAP (Shapley Additive Explanations) or LIME to provide post-hoc interpretability for the ensemble models, making the "black box" predictions usable by financial auditors.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{keivani2026adapting,
  title={Adapting Altman's bankruptcy prediction model to the compositional data methodology},
  author={Keivani, Fatemeh and Coenders, Germ{\`a} and Escaram{\'\i}s, Ge{\`o}rgia},
  journal={arXiv preprint arXiv:2603.24215},
  year={2026}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2026). Robust Financial Risk Intelligence: Compositional Bankruptcy Prediction.
GitHub repository: https://github.com/chirindaopensource/adapting_altmans_bankruptcy_prediction_model
```

## Acknowledgments

-   Credit to **Fatemeh Keivani, Germà Coenders, and Geòrgia Escaramís** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, particularly the **SciPy**, **Scikit-Learn**, **Statsmodels**, and **Pandas** contributors, as well as the maintainers of the **rpy2** bridge.

--

*This README was generated based on the structure and content of the `adapting_altmans_bankruptcy_prediction_model_draft.ipynb` notebook and follows best practices for research software documentation.*
