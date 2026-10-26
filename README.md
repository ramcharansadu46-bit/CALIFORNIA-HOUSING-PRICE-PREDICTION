# 🏠 California Housing Price Prediction
### *End-to-End Machine Learning Regression Pipeline — LPU Predictive Analysis CA2*

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-006400?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-3.0.2-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.4.4-013243?style=for-the-badge&logo=numpy&logoColor=white)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

> **Best Model: XGBoost Regressor** | Test RMSE **$40,591** | R² **0.874** | Within-20%: **78.5%**  
> Outperforms a naïve mean baseline (RMSE $114,165) by **64.4%**

---

## 📋 Table of Contents

| Section | Description |
|---------|-------------|
| [Overview](#-overview) | Project context and objectives |
| [Project Structure](#-project-structure) | Directory layout |
| [Quick Start](#-quick-start) | Environment setup & running |
| [Web Dashboard](#-web-dashboard) | Interactive prediction app |
| [Research Objectives](#-research-objectives) | Six measurable ML goals |
| [Model Benchmark](#-model-benchmark) | Seven-model comparison table |
| [Key Results](#-key-results) | Performance summary |
| [Figures Gallery](#-figures-gallery) | 20 publication charts |
| [Reproduce Results](#-reproduce-results) | Full pipeline execution |
| [Project Report](#-project-report) | Academic deliverables |
| [Tech Stack](#-tech-stack) | Libraries & versions |

---

## 🎯 Overview

This project performs a **rigorous end-to-end predictive analysis** of the **1990 California Census District House Values** dataset (20,640 census tracts). It was built for the LPU Predictive Analysis CA2 assessment and covers:

- **Comprehensive data audit**: missing value patterns, outliers, censored targets, duplicates
- **Leakage-safe ML pipelines**: KNN imputation, feature engineering, and scaling all fitted on training folds only
- **Seven tuned regressors** compared with 5-fold cross-validation, Nadeau-Bengio corrected t-test, and paired bootstrap significance testing
- **Geographic feature engineering**: distance vectors to 5 major California urban hubs (SF, LA, SD, San Jose, Sacramento) to address spatial autocorrelation (Moran's I = 0.79)
- **Ratio feature engineering**: rooms/household, bedrooms-ratio, population/household — normalised for district size
- **Error analysis & explainability**: TreeSHAP beeswarm plots, Partial Dependence Plots, permutation importance
- **Interactive web dashboard**: live inference API + publication figure gallery

---

## 📁 Project Structure

```
CA2_Predictive_Analysis_Project/
│
├── data/
│   ├── raw/
│   │   ├── housing.csv              # Original dataset (unchanged) — 20,640 rows × 10 cols
│   │   └── SHA256.txt               # Integrity checksum
│   ├── processed/
│   │   ├── train.csv                # 80% income-stratified split (16,512 rows, seed 42)
│   │   └── test.csv                 # 20% held-out test set (4,128 rows)
│   └── DATA_DOCUMENTATION.md        # Provenance, data dictionary, known issues
│
├── notebooks/
│   └── California_Housing_Predictive_Analysis.ipynb   # Main 61-cell notebook (executed)
│
├── src/                             # Production source modules
│   ├── __init__.py
│   ├── config.py                    # Paths, seeds, hub coordinates, constants
│   ├── features.py                  # build_preprocessor() — KNN, ratios, spatial, OHE
│   ├── evaluate.py                  # regression_metrics() — RMSE, MAE, R², MedAPE, Within20%
│   └── predict.py                   # CLI: python -m src.predict data/processed/test.csv
│
├── models/
│   ├── best_model.joblib            # Serialised XGBoost pipeline (sklearn 1.8.0 + xgboost 3.4.1)
│   └── model_card.json             # Pipeline metadata, hyperparameters, test metrics
│
├── results/
│   ├── metrics.json                 # All numeric results from the notebook (every table in report)
│   ├── model_comparison.csv         # 7-model CV + test benchmark
│   ├── cv_fold_rmse.csv             # Per-fold RMSE for each model
│   ├── test_predictions.csv         # Row-level predictions on test set
│   └── data_dictionary.csv          # Machine-readable feature definitions
│
├── figures/                         # 20 × publication-quality PNG figures (200 dpi)
│   ├── fig01_data_quality.png
│   ├── fig02_distributions.png
│   ├── ...                          # (20 figures total)
│   └── fig20_partial_dependence.png
│
├── report/
│   ├── CA2_Predictive_Analysis_Report.docx   # Full academic report (editable Word)
│   └── CA2_Predictive_Analysis_Report_preview.pdf  # PDF preview with all figures
│
├── docs/
│   └── SCREENSHOT_CHECKLIST.md      # GitHub/viva screenshot submission guide
│
├── web/                             # Interactive prediction web app
│   ├── index.html                   # Glassmorphism dashboard UI
│   ├── style.css                    # Design system (dark mode, responsive)
│   └── app.js                       # Live predictor, gallery, presets, lightbox
│
├── tools/                           # Build & utility scripts
│   ├── build_notebook.py            # Regenerate notebook cells from code builder
│   ├── run_notebook.py              # Execute notebook headlessly via nbclient
│   ├── build_report.py              # Generate DOCX report
│   ├── build_docs.py                # Update README/docs from results/metrics.json
│   ├── build_viva.py                # Viva preparation guide generator
│   ├── train_best_model.py          # Retrain + serialize production XGBoost pipeline
│   ├── serve_dashboard.py           # Python HTTP server for web dashboard (port 5050)
│   ├── docx_helpers.py              # Word document formatting utilities
│   ├── report_front.py              # Report cover/front-matter builder
│   ├── report_s1_3.py               # Report sections 1–3 builder
│   ├── report_s4a.py                # Report section 4a builder
│   ├── report_s4b.py                # Report section 4b builder
│   ├── report_end.py                # Report conclusion/references builder
│   └── tune_cover.py                # Cover page customization tool
│
├── requirements.txt                 # Minimum version requirements
├── requirements-exact.txt           # Pinned versions (exact reproduction)
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ramcharansadu46-bit/CALIFORNIA-HOUSING-PRICE-PREDICTION.git
cd CALIFORNIA-HOUSING-PRICE-PREDICTION
```

### 2. Create & Activate Virtual Environment

```bash
# Using uv (recommended — fast)
pip install uv
uv venv --python 3.12 .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# Install exact dependencies for full reproduction
uv pip install -r requirements-exact.txt notebook
```

> ⚠️ **Requires Python 3.12** and the exact scikit-learn / XGBoost versions listed in `requirements-exact.txt` to load the pre-serialized model without warnings.

### 3. Register Jupyter Kernel

```bash
.venv\Scripts\python.exe -m ipykernel install --user --name python3 --display-name "Python 3 (ipykernel)"
```

### 4. Run a Prediction

```bash
.venv\Scripts\python.exe -m src.predict data/processed/test.csv
```

**Expected output (first 5 rows):**

```
   longitude  latitude  housing_median_age  ...  actual  predicted
0    -118.39     34.12                29.0  ...  500001    495799
1    -120.42     34.89                24.0  ...  162500    183069
2    -118.45     34.25                36.0  ...  204600    191468
3    -118.10     33.91                35.0  ...  159700    171339
4    -117.07     32.77                38.0  ...  184000    227261
```

---

## 🌐 Web Dashboard

Launch the full interactive prediction dashboard:

```bash
.venv\Scripts\python.exe tools/serve_dashboard.py
```

Then open: **[http://127.0.0.1:5050](http://127.0.0.1:5050)**

### Dashboard Features

| Feature | Description |
|---------|-------------|
| 🎛️ **Live Predictor** | Adjust 9 district parameters; get instant XGBoost inference |
| 🏙️ **District Presets** | One-click load for SF Bay, LA Coastal, Silicon Valley, San Diego, Sacramento, Inland |
| 🎲 **Random Test Sample** | Load real test records and compare predicted vs. actual ground truth |
| 📊 **Model Benchmark Table** | Full 7-model comparison with CV RMSE, Test R², Within-20% |
| 🖼️ **Figure Gallery** | All 20 publication charts with lightbox zoom |
| 📥 **Report Downloads** | DOCX report, PDF preview, JSON model card |

---

## 🔬 Research Objectives

| # | Objective | Finding |
|---|-----------|---------|
| 1 | **Data Audit & KNN Imputation** | 207 rows missing `total_bedrooms` (1.00%). KNN (k=5) preserves bivariate covariance vs. median imputation, without leaking test data. |
| 2 | **Income Non-Linearity & Price Caps** | Median income is the single strongest predictor (r = 0.69). Right-censoring at $500,001 affects 4.3% of test rows but drives 16% of squared error. |
| 3 | **Geographic Spatial Features** | Moran's I = 0.79 (severe clustering). Distance vectors to 5 urban hubs + coastal proximity directly reduce spatial residuals. |
| 4 | **Ratio Feature Engineering** | `rooms_per_household`, `bedrooms_ratio`, and `population_per_household` normalise raw aggregates for district size. Top-5 permutation importance features. |
| 5 | **Multi-Model Hyperparameter Tuning** | 7 models tuned with 3-fold grid/randomized search and validated with 5-fold CV. XGBoost histogram-binned trees win on all metrics. |
| 6 | **Error Diagnostics & TreeSHAP** | Heteroskedastic residuals around capped values. TreeSHAP identifies `median_income` and `dist_SanFrancisco` as dominant drivers globally and locally. |

---

## 📊 Model Benchmark

Evaluated on a held-out 20% test set (4,128 districts), never seen during training or tuning.

| Model | 5-Fold CV RMSE | Test RMSE | Test MAE | Test R² | Within 20% | Inference/4k |
|-------|:--------------:|:---------:|:--------:|:-------:|:----------:|:------------:|
| 🏆 **XGBoost (F3 + Ratios + Spatial)** | $42,926 ± $1,064 | **$40,591** | **$26,217** | **0.874** | **78.5%** | 0.11s |
| HistGradientBoosting | $44,996 ± $1,097 | $43,024 | $28,095 | 0.858 | 76.2% | 0.09s |
| Random Forest | $45,488 ± $996 | $43,613 | $28,205 | 0.854 | 75.9% | 0.25s |
| k-Nearest Neighbors | $56,184 ± $1,743 | $53,768 | $36,199 | 0.778 | 64.9% | 0.29s |
| Ridge Regression | $63,389 ± $1,063 | $62,098 | $45,173 | 0.704 | 51.8% | 0.03s |
| Ridge (log-target) | $74,289 ± $3,659 | $71,014 | $43,775 | 0.613 | 58.9% | 0.04s |
| Dummy (mean baseline) | $115,699 ± $1,194 | $114,165 | $90,807 | −0.000 | 26.4% | 0.03s |

> **Statistical significance:** XGBoost vs HistGB test RMSE difference = −$2,044 (95% bootstrap CI: [−2,687, −1,411], *p* < 0.001, Nadeau-Bengio corrected t-test).

---

## 📈 Key Results

```
🏆  XGBoost Best Model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Test RMSE         :  $40,591
  Test MAE          :  $26,217
  Test R²           :   0.874  (87.4% variance explained)
  Median APE        :   9.44%
  Within 20% Error  :  78.5% of test predictions
  Training rows     :  16,512 (80% income-stratified)
  Test rows         :   4,128 (20% held-out)

Pipeline: KNN Imputer (k=5) → F3 Ratios + 5 Hub Distances + Coastal
          → Log1p / StandardScaler / OHE → XGBoost Histogram (n=800, depth=8, lr=0.03)
```

---

## 🖼️ Figures Gallery

All 20 publication-quality figures are in [`figures/`](figures/):

| Figure | Title |
|--------|-------|
| `fig01` | Data Quality & Missingness Audit |
| `fig02` | Feature Distributions & Skewness |
| `fig03` | Pearson Correlation Matrix |
| `fig04` | Geographic Price Heatmap |
| `fig05` | KNN vs Median Imputation Comparison |
| `fig06` | Non-Linear Income–Price Relationship |
| `fig07` | Residuals by Income Deciles |
| `fig08` | Ocean Proximity Stratification |
| `fig09` | Moran's I Spatial Autocorrelation Scatter |
| `fig10` | Spatial Feature Engineering Gain |
| `fig11` | Engineered Ratio Feature Distributions |
| `fig12` | Multi-Model Benchmark Comparison |
| `fig13` | Predicted vs Actual Parity Plot |
| `fig14` | XGBoost Learning Curves |
| `fig15` | Residual Diagnostic Plots (Q-Q, Homoskedasticity) |
| `fig16` | Segmented Sub-group Error Analysis |
| `fig17` | Spatial Residual Error Map |
| `fig18` | Permutation Feature Importance |
| `fig19` | SHAP Beeswarm Value Summary |
| `fig20` | Partial Dependence Plots (PDP) |

---

## 🔄 Reproduce Results

### Option A — Execute the Full Notebook (Recommended)

```bash
# Regenerate notebook from builder (optional)
.venv\Scripts\python.exe tools/build_notebook.py

# Execute all 61 cells headlessly (~15-30 min, 1 CPU)
.venv\Scripts\python.exe tools/run_notebook.py \
  notebooks/California_Housing_Predictive_Analysis.ipynb \
  notebooks/California_Housing_Predictive_Analysis.ipynb

# Rebuild docs and report from fresh metrics
.venv\Scripts\python.exe tools/build_docs.py
.venv\Scripts\python.exe tools/build_report.py
```

> Set `QUICK=1` environment variable for a fast smoke test (sub-sampled, **never** use these numbers for the report).

### Option B — Retrain the Best Model Only

```bash
.venv\Scripts\python.exe tools/train_best_model.py
```

### Option C — Open in Jupyter Notebook

```bash
.venv\Scripts\python.exe -m notebook notebooks/California_Housing_Predictive_Analysis.ipynb
```

---

## 📄 Project Report

| Deliverable | Location | Description |
|-------------|----------|-------------|
| 📝 DOCX Report | [`report/CA2_Predictive_Analysis_Report.docx`](report/CA2_Predictive_Analysis_Report.docx) | Full academic report (Sections 1–6, editable) |
| 📕 PDF Preview | [`report/CA2_Predictive_Analysis_Report_preview.pdf`](report/CA2_Predictive_Analysis_Report_preview.pdf) | Print-ready PDF with all embedded figures |
| 📓 Notebook | [`notebooks/California_Housing_Predictive_Analysis.ipynb`](notebooks/California_Housing_Predictive_Analysis.ipynb) | Executed 61-cell Jupyter notebook |
| 📦 Model Card | [`models/model_card.json`](models/model_card.json) | Pipeline metadata, hyperparameters, environment |
| 📊 All Metrics | [`results/metrics.json`](results/metrics.json) | Every numeric result in the report |

---

## 🛠 Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.12.13 | Core runtime |
| NumPy | 2.4.4 | Numerical arrays |
| Pandas | 3.0.2 | DataFrame manipulation |
| SciPy | 1.17.1 | Statistical tests |
| scikit-learn | 1.8.0 | Pipelines, transformers, CV |
| XGBoost | 3.4.1 | Gradient boosted trees |
| SHAP | 0.53.0 | TreeSHAP explainability |
| Matplotlib | 3.10.8 | Publication figures |
| Seaborn | 0.13.2 | Statistical visualizations |
| statsmodels | 0.15.0 | Moran's I, regression stats |
| joblib | 1.5.3 | Model serialization |
| Jupyter Notebook | 7.6.3 | Interactive analysis |
| nbconvert | 7.17.2 | Headless notebook execution |

---

## 📌 Important Notes

- The dataset reflects **1990 California census conditions** — do not use this model to value current property.
- The target variable (`median_house_value`) is **right-censored at $500,001** — 4.3% of the dataset.
- The **pre-trained model** in `models/best_model.joblib` requires `scikit-learn==1.8.0` and `xgboost==3.4.1` to load cleanly. Run `tools/train_best_model.py` to retrain from scratch in any environment.
- All randomness uses **seed 42** for full reproducibility.
- The web dashboard (`tools/serve_dashboard.py`) serves the prediction REST API locally — **no internet connection required**.

---

## 👤 Author

**Ramcharan Sadu**  
LPU — B.Tech Computer Science  
Predictive Analysis CA2 Project  

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with Python 3.12 · scikit-learn 1.8.0 · XGBoost 3.4.1 · Seed 42 · LPU CA2 2026</sub>
</div>
