"""Generates README.md, data documentation, screenshot checklist, .gitignore and src/predict.py from results/metrics.json."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import os; R = json.load(open(os.environ.get("RESULTS_JSON", ROOT / "results" / "metrics.json")))
o5, o6, q, env = R["o5"], R["o6"], R["quality"], R["environment"]
best = o5["best_model"]; te = o5["test_table"][best]; run = o5["runner_up"]

(ROOT / "src" / "predict.py").write_text('''"""Predict median house value for new districts with the saved pipeline.

Usage (from the project root):
    python -m src.predict data/processed/test.csv            # prints first 10 predictions
Input CSV must contain the 9 predictor columns (the target column is ignored if present).
"""
import sys
import joblib
import pandas as pd
from .config import MODEL_DIR, TARGET


def main(path):
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET], errors="ignore")
    out = df.copy()
    out["predicted_median_house_value"] = model.predict(X)
    print(out.head(10).to_string())
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/processed/test.csv")
''')

(ROOT / ".gitignore").write_text("__pycache__/\n*.pyc\n.ipynb_checkpoints/\n.venv/\nreport/_tmp/\n.DS_Store\n# personal study material - keep out of a public repository\ndocs/VIVA_PREPARATION_GUIDE.docx\n")

(ROOT / "data" / "DATA_DOCUMENTATION.md").write_text(f"""# Dataset documentation - California Housing Prices

| Item | Detail |
|---|---|
| Content | 20,640 census block groups (districts) from the 1990 US Census; 9 predictors + 1 target |
| Prediction target | `median_house_value` (USD), right-censored at 500,001 |
| Original source | StatLib repository; Pace, R. K. and Barry, R. (1997), *Statistics & Probability Letters* 33(3), 291-297 |
| Kaggle listing | https://www.kaggle.com/datasets/camnugent/california-housing-prices |
| OpenML listing | https://api.openml.org/d/43705/json (dataset ID 43705) |
| File used here | https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv |
| SHA-256 of `raw/housing.csv` | `8a3727f4cf54ac1a327f69b1d5b4db54c5834ea81c6e4efc0d163300022a685e` |
| Licence | Check the licence shown on the Kaggle page before redistributing |

**How the file was obtained.** Direct Kaggle download was not possible in the environment where the project was built, so the same file was downloaded from the public GitHub repository of A. Geron (2019). The Kaggle page itself could not be opened from that environment; size and origin were cross-checked against the Kaggle description as republished by a third-party mirror (Baselight), OpenML metadata and Keras documentation. **Please compare the checksum with a file downloaded from Kaggle.** To verify against Kaggle yourself, download the file there and compare checksums (`sha256sum housing.csv` / `certutil -hashfile housing.csv SHA256` on Windows).

## Data dictionary
| Variable | Type | Unit | Description |
|---|---|---|---|
| longitude | float | degrees | Block-group longitude |
| latitude | float | degrees | Block-group latitude |
| housing_median_age | float | years | Median age of houses (capped at 52) |
| total_rooms | float | count | Total rooms in the block group |
| total_bedrooms | float | count | Total bedrooms ({q['missing']['total_bedrooms']} missing) |
| population | float | persons | Block-group population |
| households | float | count | Number of households |
| median_income | float | 10,000 USD | Median household income (capped at 15.0001) |
| median_house_value | float | USD | **Target** (capped at 500,001) |
| ocean_proximity | categorical | - | <1H OCEAN, INLAND, NEAR OCEAN, NEAR BAY, ISLAND |

## Known quality issues (see report Section 3.2)
{q['missing']['total_bedrooms']} missing `total_bedrooms`; {q['target_capped_n']} capped targets ({q['target_capped_pct']:.2f} %); {q['age_capped_n']} capped ages ({q['age_capped_pct']:.2f} %); {q['income_capped_n']} capped incomes; no duplicates.

`processed/train.csv` and `processed/test.csv` are the 80/20 income-stratified split (seed 42) produced by the notebook.
""")

(ROOT / "README.md").write_text(f"""# Predictive Analysis of California Census-District House Values (CA2 Project)

End-to-end predictive-analysis project (regression) for the LPU **Predictive Analysis CA2**: data audit, EDA, six measurable research objectives, seven tuned models, error analysis and explainability. **Best model: {best}** - test RMSE **${te['RMSE']:,.0f}**, MAE ${te['MAE']:,.0f}, R^2 {te['R2']:.3f}.

## 1. Project structure
```
CA2_Project/
|-- data/raw/housing.csv            original dataset (unchanged) + SHA256.txt
|-- data/processed/                 train.csv / test.csv (80/20, income-stratified, seed 42)
|-- data/DATA_DOCUMENTATION.md      provenance, dictionary, known issues
|-- notebooks/California_Housing_Predictive_Analysis.ipynb   executed notebook (all objectives)
|-- src/                            config.py, features.py (imputer + feature engineering + pipeline), evaluate.py, predict.py
|-- figures/                        20 publication-quality figures (PNG, 200 dpi)
|-- results/                        metrics.json (every number in the report), model_comparison.csv, cv_fold_rmse.csv, test_predictions.csv, data_dictionary.csv
|-- models/                         best_model.joblib (full pipeline) + model_card.json
|-- report/                         CA2_Predictive_Analysis_Report.docx (editable) + PDF preview
|-- docs/                           SCREENSHOT_CHECKLIST.md (the viva guide is kept local, see .gitignore)
|-- tools/                          scripts that build the notebook, run it, and generate the report/docs
`-- requirements.txt
```

## 2. Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m ipykernel install --user --name python3
```
Exact versions used for the reported results are in `requirements-exact.txt` (`pip install -r requirements-exact.txt`). Tested with Python {env['python']}, pandas {env['pandas']}, NumPy {env['numpy']}, scikit-learn {env['scikit-learn']}, XGBoost {env['xgboost']}.

## 3. Reproduce the results
```bash
python tools/build_notebook.py                     # (optional) regenerate the notebook from its builder
jupyter nbconvert --to notebook --execute --inplace notebooks/California_Housing_Predictive_Analysis.ipynb   # ~15-30 min on 1 CPU
python tools/build_docs.py                         # README / data docs from results/metrics.json
python tools/build_report.py                       # DOCX report (needs LibreOffice for page numbers in the contents)
```
Set `QUICK=1` for a fast smoke test (sub-sampled; **never** use its numbers). All randomness uses seed {env['random_state']}.

Use the saved model (run from the project root so that `src` is importable):
```bash
python -m src.predict data/processed/test.csv
```

## 4. Methodology (summary)
1. **Audit** the raw file (missing values, duplicates, consistency, censoring, outliers) - nothing is deleted silently.
2. **Split first** (80/20, stratified on income bands), then explore the training set only.
3. **Leakage-safe pipelines**: imputer -> feature engineering -> log1p/scale/one-hot, fitted on training folds only; test set used once.
4. **Six objectives** (data quality & imputation; income relationship; geography & spatial features; ratio features; model comparison; error analysis & explainability).
5. **Models**: Dummy mean, Ridge, Ridge (log-target), k-NN, Random Forest, HistGradientBoosting, XGBoost; tuned with 3-fold CV (grid/randomised search), selected by 5-fold CV RMSE, compared with the Nadeau-Bengio corrected t-test and a paired bootstrap on the test set.

## 5. Key results
| Model | CV RMSE | Test RMSE | Test MAE | Test R^2 |
|---|---|---|---|---|
""" + "\n".join(f"| {m} | ${o5['cv_table'][m]['CV_RMSE']:,.0f} | ${o5['test_table'][m]['RMSE']:,.0f} | ${o5['test_table'][m]['MAE']:,.0f} | {o5['test_table'][m]['R2']:.3f} |" for m in o5["rank_by_cv"]) + f"""

* {best} vs runner-up {run}: test RMSE difference {o5['significance'][run]['test_RMSE_diff']:,.0f} USD (95 % bootstrap CI {o5['significance'][run]['boot_ci_lo']:,.0f} to {o5['significance'][run]['boot_ci_hi']:,.0f}).
* Capped (500,001) districts are {o6['censoring']['pct_test_capped']:.1f} % of the test set but cause {o6['censoring']['rmse_share_from_capped_rows_pct']:.0f} % of the squared error.
* Strong spatial autocorrelation of prices (Moran's I = {R['o3']['moran_I']['I']:.2f}) means random CV is optimistic for *new regions* - see report Section 5.3.

## 6. Upload to GitHub (requires your own account)
```bash
cd CA2_Project
git init
git add .
git commit -m "CA2 Predictive Analysis: California housing price prediction"
git branch -M main
# create an EMPTY repository on github.com first (no README), then:
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
After pushing, check that the notebook renders on GitHub and that figures appear. Then follow `docs/SCREENSHOT_CHECKLIST.md`.

## 7. Notes
* The dataset reflects 1990 conditions; do not use the model to value current property.
* The report's cover, declaration, certificate and acknowledgement are filled in with typed names only; print and sign by hand wherever LPU requires an actual signature.
* Review the whole project so that you can explain every step; follow your university's academic-integrity and AI-use policy.
""")

(ROOT / "docs" / "SCREENSHOT_CHECKLIST.md").write_text("""# Screenshot checklist (genuine screenshots only)

Nothing below has been created for you: screenshots must show **your own** machine, accounts and activity. Do not edit, mock up or reuse other people's screenshots. Save images as PNG in `docs/screenshots/` using the suggested file names.

## A. Genuine project screenshots (your computer)
- [ ] `A01_project_folder.png` - project folder tree in File Explorer / VS Code showing the structure.
- [ ] `A02_environment_setup.png` - terminal showing `pip install -r requirements.txt` finishing without errors.
- [ ] `A03_notebook_start.png` - the notebook opened in Jupyter/VS Code (title and objectives table visible).
- [ ] `A04_data_inspection.png` - output of the data-quality audit cells (missing values, capped counts).
- [ ] `A05_eda_figures.png` - one or two EDA figures rendered in the notebook (e.g. correlation heat-map).
- [ ] `A06_objective_results.png` - one result table per objective (O1-O6), at least one screenshot each.
- [ ] `A07_model_tuning.png` - the tuning/CV output cell (timings visible).
- [ ] `A08_model_comparison.png` - the model-comparison table and Figure 12.
- [ ] `A09_shap_pdp.png` - SHAP summary and partial-dependence outputs.
- [ ] `A10_saved_model_test.png` - the cell printing "saved model reproduces test predictions: OK".
- [ ] `A11_predict_script.png` - terminal output of `python -m src.predict data/processed/test.csv`.
- [ ] `A12_report_docx.png` - the DOCX open in Word showing cover page, table of contents (after updating fields) and one analysis page.

## B. GitHub repository screenshots (your GitHub account)
- [ ] `B01_repo_home.png` - repository front page with README rendered and the file list.
- [ ] `B02_repo_structure.png` - folders `data/`, `notebooks/`, `src/`, `figures/`, `results/`, `models/`, `report/`.
- [ ] `B03_notebook_rendered.png` - the notebook rendered on GitHub (figures visible).
- [ ] `B04_commit_history.png` - Commits page showing your real commit(s) and date.
- [ ] `B05_readme_results.png` - README results table.
- [ ] `B06_repo_url.png` - browser address bar showing your repository URL (put the URL in the report/README).

## C. LinkedIn screenshots (your LinkedIn account)
Only if you actually publish them:
- [ ] `C01_project_section.png` - the project added under Profile -> Projects (title, description, GitHub link).
- [ ] `C02_post.png` - your own post about the project (short summary, 1-2 figures, repository link).
- [ ] `C03_post_url.png` - the post URL and date visible.
Suggested post angle: the problem, the dataset, the three main findings (best model and RMSE, spatial effect, censoring effect), and a link to the repository. Do not claim results that are not in `results/metrics.json`.

## D. Where to use them
The LPU format has no appendix for screenshots; attach them after the references as an "Annexure" only if your supervisor asks, or submit them as separate evidence.
""")
print("docs written")
