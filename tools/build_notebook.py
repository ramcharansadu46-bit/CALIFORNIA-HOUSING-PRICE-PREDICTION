"""Builds notebooks/California_Housing_Predictive_Analysis.ipynb (executed separately)."""
import nbformat as nbf

cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ───────────────────────────── 0. TITLE ─────────────────────────────
md(r'''
# Predictive Analysis of Median House Values in California Census Districts
**CA2 Project – Predictive Analysis | Lovely Professional University**

**Dataset:** California Housing Prices (1990 US Census, StatLib; Kaggle `camnugent/california-housing-prices`) – 20,640 districts × 10 variables  
**Prediction task:** supervised **regression** – predict `median_house_value` (US$) of a census block-group ("district").

## Research objectives
| # | Objective (measurable) | Key methods | Success measure |
|---|---|---|---|
| **O1** | Audit data quality (missingness, censoring/capping, outliers) and choose an imputation strategy for `total_bedrooms` | Mann-Whitney U, χ², masked-imputation experiment, CV | Masked-imputation RMSE; downstream CV RMSE; counts of capped/missing values |
| **O2** | Quantify how strongly and in what functional form median income determines house value | Pearson/Spearman + bootstrap CI, OLS, F-test, Breusch-Pagan, log-log elasticity | Correlation CI, R², elasticity, BP test p-value |
| **O3** | Test whether geography (ocean proximity, spatial position) explains price and whether spatial features improve prediction | Kruskal-Wallis, Holm-corrected pairwise tests, Moran's I, feature ablation | ε², Moran's I, ΔRMSE (95 % CI) |
| **O4** | Test whether engineered per-household ratio features add predictive value beyond raw counts | Spearman comparison, feature ablation with corrected resampled t-test | ΔRMSE, ΔR², Nadeau-Bengio p-value |
| **O5** | Compare ≥ 4 ML models fairly and select the best with statistical evidence | Dummy, Ridge, KNN, Random Forest, HistGB, XGBoost; tuned CV; paired bootstrap | CV & test RMSE, MAE, R²; CI of RMSE difference |
| **O6** | Diagnose model errors (incl. effect of target censoring) and explain predictions | Residual analysis, segment errors, Moran's I of residuals, permutation importance, SHAP, PDP | Segment RMSE/MedAPE, importance concentration, residual autocorrelation |

**Leakage-prevention principles used throughout:** (1) the 80/20 train/test split is made *before* any EDA or modelling decision; (2) every statistic (medians, scalers, imputers, encoders) is learned **inside scikit-learn Pipelines**, i.e. on training folds only; (3) the test set is used **once**, after model selection, for the final evaluation; (4) spatial features use fixed public coordinates of cities, never the target.
''')

md("## 0. Setup")
code(r'''
import os, sys, json, time, warnings, platform
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.nonparametric.smoothers_lowess import lowess

import sklearn, xgboost, joblib
from sklearn.model_selection import (train_test_split, KFold, RepeatedKFold, RandomizedSearchCV,
                                     GridSearchCV, cross_validate, learning_curve)
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from xgboost import XGBRegressor

from src.config import *
from src.features import BedroomImputer, FeatureEngineer, build_preprocessor
from src.evaluate import regression_metrics, nadeau_bengio_ttest, paired_bootstrap_rmse_diff, morans_i

QUICK = os.environ.get("QUICK") == "1"      # QUICK=1 -> tiny smoke-test run (never used for reported results)
FIG_DIR.mkdir(exist_ok=True); RESULT_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
np.random.seed(RANDOM_STATE)

# publication-style plotting defaults
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"], "axes.titlesize": 11,
                     "axes.labelsize": 10, "legend.fontsize": 9, "axes.titleweight": "bold"})
PALETTE = sns.color_palette("deep")
def savefig(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=200); plt.show(); plt.close(fig)

R = {}                                                   # every number quoted in the report is stored here
def dump():
    with open(RESULT_DIR / "metrics.json", "w") as f: json.dump(R, f, indent=2, default=float)
R["environment"] = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                    "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
                    "random_state": RANDOM_STATE, "quick_mode": QUICK}
print(R["environment"])
''')

# ───────────────────────────── 1. INSPECTION ─────────────────────────────
md(r'''
## 1. Data inspection, dictionary and quality assessment
The raw file is loaded **unchanged**. Its SHA-256 checksum is stored in `data/raw/SHA256.txt` so the exact bytes used can be verified.
''')
code(r'''
raw = pd.read_csv(DATA_RAW)
print("shape:", raw.shape)
display(raw.head())
display(raw.dtypes.to_frame("dtype").T)
''')
code(r'''
data_dict = pd.DataFrame([
 ("longitude","float","degrees","Block-group longitude (more negative = further west)"),
 ("latitude","float","degrees","Block-group latitude (higher = further north)"),
 ("housing_median_age","float","years","Median age of houses in the block group (capped at 52)"),
 ("total_rooms","float","count","Total rooms in all houses of the block group"),
 ("total_bedrooms","float","count","Total bedrooms in the block group (207 missing)"),
 ("population","float","persons","Block-group population"),
 ("households","float","count","Number of households"),
 ("median_income","float","10,000 USD","Median household income (capped at 15.0001)"),
 ("median_house_value","float","USD","TARGET - median house value (capped at 500,001)"),
 ("ocean_proximity","categorical","-","<1H OCEAN, INLAND, NEAR OCEAN, NEAR BAY, ISLAND")],
 columns=["variable","type","unit","description"])
display(data_dict)
data_dict.to_csv(RESULT_DIR / "data_dictionary.csv", index=False)
display(raw.describe().T.round(2))
''')
md("### 1.1 Data-quality audit (missing values, duplicates, inconsistencies, censoring, outliers)")
code(r'''
q = {}
miss = raw.isna().sum(); q["missing"] = {k:int(v) for k,v in miss[miss>0].items()}
q["missing_pct_total_bedrooms"] = float(raw.total_bedrooms.isna().mean()*100)
q["duplicates"] = int(raw.duplicated().sum())
q["n_rows"], q["n_cols"] = raw.shape
q["ocean_categories"] = raw.ocean_proximity.value_counts().to_dict()
q["ocean_whitespace_issues"] = int((raw.ocean_proximity != raw.ocean_proximity.str.strip()).sum())

# logical consistency checks
q["households_le_0"]       = int((raw.households <= 0).sum())
q["bedrooms_gt_rooms"]     = int((raw.total_bedrooms > raw.total_rooms).sum())
q["population_le_0"]       = int((raw.population <= 0).sum())
q["lat_lon_out_of_bounds"] = int((~raw.latitude.between(32, 43) | ~raw.longitude.between(-125, -114)).sum())

# censoring / capping
q["target_capped_n"]   = int((raw[TARGET] >= CAP_VALUE).sum()); q["target_capped_pct"] = float((raw[TARGET] >= CAP_VALUE).mean()*100)
q["age_capped_n"]      = int((raw.housing_median_age >= raw.housing_median_age.max()).sum()); q["age_cap_value"] = float(raw.housing_median_age.max())
q["age_capped_pct"]    = float((raw.housing_median_age >= raw.housing_median_age.max()).mean()*100)
q["income_capped_n"]   = int((raw.median_income >= raw.median_income.max()).sum()); q["income_cap_value"] = float(raw.median_income.max())
q["income_min"]        = float(raw.median_income.min()); q["target_min"] = float(raw[TARGET].min())

# IQR outlier counts (Tukey 1.5 x IQR) – retained, because they are real districts (e.g. resorts)
out = {}
for c in raw.select_dtypes("number").columns:
    q1, q3 = raw[c].quantile([.25, .75]); iqr = q3 - q1
    out[c] = int(((raw[c] < q1 - 1.5*iqr) | (raw[c] > q3 + 1.5*iqr)).sum())
q["iqr_outliers"] = out
R["quality"] = q
display(pd.Series({k: v for k, v in q.items() if not isinstance(v, dict)}).to_frame("value"))
display(pd.Series(out, name="IQR outliers").to_frame().T)
dump()
''')
md(r'''
**Findings and cleaning decisions**

* **No duplicates, no inconsistent category labels, no impossible coordinates.** Logical checks (households ≤ 0, bedrooms > rooms) are reported above.
* **207 missing `total_bedrooms`** (≈1 %) → imputed *inside the pipeline* (strategy chosen in O1).
* **Censoring:** the target is right-censored at \$500,001, `housing_median_age` at 52 years and `median_income` at 15.0001. These are *measurement limits*, not errors, so the rows are **kept**; their impact is quantified in O1/O6 instead of silently deleted.
* **Outliers** (e.g. 140+ rooms per household in resort/vacation districts) are genuine; they are kept and handled through log-transforms and tree-based models rather than removal.
''')

md("### 1.2 Train / test split (done *before* EDA and modelling)\nStratified on income bands (income is the strongest predictor, so both sets must cover all income levels proportionally).")
code(r'''
income_band = pd.cut(raw.median_income, bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])
train_df, test_df = train_test_split(raw, test_size=0.20, random_state=RANDOM_STATE, stratify=income_band)
if QUICK: train_df = train_df.sample(3000, random_state=RANDOM_STATE)
train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)
train_df.to_csv(DATA_PROCESSED / "train.csv", index=False); test_df.to_csv(DATA_PROCESSED / "test.csv", index=False)

X_train, y_train = train_df.drop(columns=TARGET), train_df[TARGET]
X_test,  y_test  = test_df.drop(columns=TARGET),  test_df[TARGET]
R["split"] = {"n_train": len(train_df), "n_test": len(test_df), "test_fraction": 0.20, "stratified_on": "income band (5 bins)"}
bands_tr = pd.cut(train_df.median_income, [0,1.5,3,4.5,6,np.inf], labels=[1,2,3,4,5]).value_counts(normalize=True).sort_index()*100
bands_te = pd.cut(test_df.median_income,  [0,1.5,3,4.5,6,np.inf], labels=[1,2,3,4,5]).value_counts(normalize=True).sort_index()*100
display(pd.DataFrame({"train %": bands_tr.round(2), "test %": bands_te.round(2)}).T)
print(R["split"]); dump()
''')

# ───────────────────────────── 2. EDA ─────────────────────────────
md("## 2. Exploratory data analysis (training set only)")
code(r'''
display(train_df.describe().T.round(2))
R["eda"] = {"target_skew": float(y_train.skew()), "target_kurtosis": float(y_train.kurt()),
            "target_mean": float(y_train.mean()), "target_median": float(y_train.median()),
            "target_std": float(y_train.std())}
num_cols = NUM_BASE + [TARGET]
skew = train_df[num_cols].skew().round(2); display(skew.to_frame("skewness").T)
R["eda"]["skewness"] = skew.to_dict()
''')
code(r'''
# Figure 1 – data-quality overview: missingness, target censoring, age censoring
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
m = raw.isna().sum().sort_values(ascending=False); m = m[m > 0]
ax[0].barh(m.index, m.values, color=PALETTE[3]); ax[0].set_title("(a) Missing values per column"); ax[0].set_xlabel("count of missing values")
for i, v in enumerate(m.values): ax[0].text(v + 3, i, str(v), va="center")
ax[1].hist(raw[TARGET], bins=60, color=PALETTE[0], edgecolor="white"); ax[1].axvline(CAP_VALUE, color="red", ls="--", lw=1)
ax[1].set_title("(b) Target censored at $500,001"); ax[1].set_xlabel("median house value (USD)"); ax[1].set_ylabel("districts")
ax[1].annotate(f"{R['quality']['target_capped_n']} capped\n({R['quality']['target_capped_pct']:.1f}%)", xy=(CAP_VALUE, 900), xytext=(280000, 800), arrowprops=dict(arrowstyle="->"))
ax[2].hist(raw.housing_median_age, bins=52, color=PALETTE[2], edgecolor="white")
ax[2].set_title("(c) Housing age censored at 52 years"); ax[2].set_xlabel("housing median age (years)"); ax[2].set_ylabel("districts")
fig.tight_layout(); savefig(fig, "fig01_data_quality")
''')
code(r'''
# Figure 2 – distributions of all numeric variables
fig, axes = plt.subplots(3, 3, figsize=(12, 8.5))
for a, c in zip(axes.ravel(), num_cols):
    sns.histplot(train_df[c], bins=50, kde=True, ax=a, color=PALETTE[0]); a.set_title(f"{c}  (skew={train_df[c].skew():.2f})", fontsize=9); a.set_xlabel("")
fig.suptitle("Distributions of numeric variables (training set)", fontweight="bold", y=1.0); fig.tight_layout()
savefig(fig, "fig02_distributions")
''')
code(r'''
# Figure 3 – Spearman correlation heat-map + multicollinearity (VIF)
corr = train_df[num_cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, cbar_kws={"label": "Spearman ρ"}, ax=ax)
ax.set_title("Spearman correlation matrix (training set)"); savefig(fig, "fig03_correlation")
R["eda"]["spearman_with_target"] = corr[TARGET].drop(TARGET).round(3).to_dict()

Z = train_df[NUM_BASE].copy(); Z["total_bedrooms"] = Z["total_bedrooms"].fillna(Z["total_bedrooms"].median())
for c in ["total_rooms", "total_bedrooms", "population", "households"]: Z[c] = np.log1p(Z[c])
Z = sm.add_constant((Z - Z.mean()) / Z.std())
vif = pd.Series([variance_inflation_factor(Z.values, i) for i in range(1, Z.shape[1])], index=Z.columns[1:]).round(1)
display(vif.sort_values(ascending=False).to_frame("VIF").T)
R["eda"]["vif"] = vif.to_dict()
display(train_df.groupby("ocean_proximity")[TARGET].agg(["count", "mean", "median"]).round(0))
dump()
''')
code(r'''
# Figure 4 – geographic distribution of districts coloured by house value
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
s = ax[0].scatter(train_df.longitude, train_df.latitude, c=train_df[TARGET], s=train_df.population/150+1, cmap="viridis", alpha=.45, linewidths=0)
plt.colorbar(s, ax=ax[0], label="median house value (USD)"); ax[0].set_title("(a) House value (marker size ∝ population)")
for n, (la, lo) in HUBS.items(): ax[0].plot(lo, la, "r*", ms=11); ax[0].annotate(n, (lo, la), xytext=(4, 4), textcoords="offset points", fontsize=8, color="red")
for k, g in train_df.groupby("ocean_proximity"): ax[1].scatter(g.longitude, g.latitude, s=3, alpha=.5, label=k)
ax[1].legend(markerscale=4, title="ocean_proximity"); ax[1].set_title("(b) Ocean-proximity categories")
for a in ax: a.set_xlabel("longitude"); a.set_ylabel("latitude")
fig.tight_layout(); savefig(fig, "fig04_geography")
''')

# ───────────────────────────── helpers for CV ─────────────────────────────
md(r'''
## 3. Shared evaluation utilities
All comparisons below use **the same cross-validation folds** (fixed seed), so differences between variants are *paired*.  
Because CV training sets overlap, a plain paired *t*-test is too liberal; we use the **Nadeau–Bengio corrected resampled t-test** (`src/evaluate.py`).
''')
code(r'''
from sklearn.model_selection import cross_val_score
K_SPLITS, K_REPEATS = (3, 1) if QUICK else (5, 2)
CV_ABL = RepeatedKFold(n_splits=K_SPLITS, n_repeats=K_REPEATS, random_state=RANDOM_STATE)
N_TR_FOLD = int(len(X_train) * (K_SPLITS - 1) / K_SPLITS); N_TE_FOLD = len(X_train) - N_TR_FOLD

def cv_rmse(estimator, X=None, y=None, cv=CV_ABL):
    """Per-fold RMSE (positive numbers) for a (pre-processing + model) pipeline."""
    X = X_train if X is None else X; y = y_train if y is None else y
    return -cross_val_score(estimator, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)

def paired_summary(a, b):
    """Summary of RMSE(a) - RMSE(b) across the same folds (negative => a is better)."""
    d = np.asarray(a) - np.asarray(b); k = len(d)
    se = d.std(ddof=1) / np.sqrt(k); tcrit = stats.t.ppf(0.975, k - 1)
    t, p = nadeau_bengio_ttest(d, N_TR_FOLD, N_TE_FOLD)
    return {"mean_diff": float(d.mean()), "ci_lo": float(d.mean() - tcrit * se), "ci_hi": float(d.mean() + tcrit * se), "NB_t": t, "NB_p": p}

def hgb(): return HistGradientBoostingRegressor(random_state=RANDOM_STATE)
print("CV scheme:", CV_ABL, "| train rows:", len(X_train))
''')

# ───────────────────────────── O1 ─────────────────────────────
md(r'''
---
# Objective 1 – Data quality, censoring and imputation of `total_bedrooms`
**Research question.** Is the missingness in `total_bedrooms` ignorable, and which imputation strategy recovers the true values best without harming downstream prediction?  
**Relevance.** Real housing data is rarely complete; a poor imputer can bias valuation models. The capped target/age/income values also limit what any model can learn.  
**Success measures.** (i) Mann-Whitney / χ² evidence on the missingness mechanism, (ii) RMSE/MAE of imputed vs. true values in a *masked-imputation experiment*, (iii) paired CV-RMSE difference of the downstream model.
''')
code(r'''
# 1a. Is the missingness related to other variables?  (missing vs. observed groups)
tb = X_train.copy(); tb["median_house_value"] = y_train
miss_mask = tb.total_bedrooms.isna()
rows = []
for c in ["housing_median_age", "total_rooms", "population", "households", "median_income", "median_house_value"]:
    a, b = tb.loc[miss_mask, c], tb.loc[~miss_mask, c]
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    rb = 2 * U / (len(a) * len(b)) - 1                       # rank-biserial correlation (effect size)
    rows.append({"variable": c, "median_missing_group": a.median(), "median_observed_group": b.median(), "U": U, "p_value": p, "rank_biserial_r": rb})
mm = pd.DataFrame(rows)
ct_tab = pd.crosstab(miss_mask, tb.ocean_proximity)
chi2, p_chi, dof, _ = stats.chi2_contingency(ct_tab)
display(mm.round(4)); display(ct_tab)
print(f"chi-square (missing vs ocean_proximity): chi2={chi2:.2f}, dof={dof}, p={p_chi:.4g}")
R["o1"] = {"n_missing_train": int(miss_mask.sum()), "missing_tests": mm.to_dict("records"), "chi2": float(chi2), "chi2_p": float(p_chi),
           "missing_by_ocean": ct_tab.loc[True].to_dict()}
''')
code(r'''
# 1b. Masked-imputation experiment: hide 5 % of OBSERVED values, impute, compare with the truth
STRATS = ["median", "group_median", "ratio", "knn", "iterative"]
N_TRIALS = 2 if QUICK else 5
def masked_trial(seed, frac=0.05):
    rng = np.random.default_rng(seed)
    idx = rng.choice(X_train.index[X_train.total_bedrooms.notna()], size=int(frac * X_train.total_bedrooms.notna().sum()), replace=False)
    Xm = X_train.copy(); truth = Xm.loc[idx, "total_bedrooms"].copy(); Xm.loc[idx, "total_bedrooms"] = np.nan
    out = {}
    for s in STRATS:
        Xi = BedroomImputer(s).fit(Xm).transform(Xm); err = Xi.loc[idx, "total_bedrooms"] - truth
        out[s] = (float(np.sqrt((err ** 2).mean())), float(err.abs().mean()), float((err.abs() / truth).median() * 100))
    return out
trials = [masked_trial(s) for s in range(N_TRIALS)]
imp_tab = pd.DataFrame({s: {"RMSE": np.mean([t[s][0] for t in trials]), "RMSE_sd": np.std([t[s][0] for t in trials], ddof=1),
                            "MAE": np.mean([t[s][1] for t in trials]), "MedAPE_%": np.mean([t[s][2] for t in trials])} for s in STRATS}).T
display(imp_tab.round(2))
BEST_IMPUTE = imp_tab["RMSE"].idxmin()
R["o1"]["masked_experiment"] = imp_tab.round(4).to_dict("index"); R["o1"]["n_trials"] = N_TRIALS; R["o1"]["best_imputer"] = BEST_IMPUTE
print("Best imputer by masked RMSE:", BEST_IMPUTE)
''')
code(r'''
# 1c. Downstream effect: does the imputation strategy change CV-RMSE of a gradient-boosting model?
down = {s: cv_rmse(Pipeline([("prep", build_preprocessor(s)), ("m", hgb())])) for s in STRATS}
base = down["median"]
dtab = pd.DataFrame({s: {"mean_CV_RMSE": v.mean(), "sd": v.std(ddof=1), **paired_summary(v, base)} for s, v in down.items()}).T
display(dtab.round(3))
R["o1"]["downstream"] = dtab.round(5).to_dict("index")
dump()
''')
code(r'''
# Figure 5 – imputation strategies: accuracy on masked values and downstream impact
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
order = imp_tab.sort_values("RMSE").index
ax[0].bar(order, imp_tab.loc[order, "RMSE"], yerr=imp_tab.loc[order, "RMSE_sd"], capsize=4, color=[PALETTE[2] if s == BEST_IMPUTE else PALETTE[0] for s in order])
ax[0].set_ylabel("RMSE of imputed total_bedrooms (rooms)"); ax[0].set_title(f"(a) Masked-imputation error (mean ± sd, {N_TRIALS} trials)")
for i, s in enumerate(order): ax[0].text(i, imp_tab.loc[s, "RMSE"] + imp_tab.loc[s, "RMSE_sd"] + 2, f"{imp_tab.loc[s,'RMSE']:.1f}", ha="center", fontsize=8)
d2 = dtab.drop(index="median")
ax[1].errorbar(d2["mean_diff"], range(len(d2)), xerr=[d2["mean_diff"] - d2["ci_lo"], d2["ci_hi"] - d2["mean_diff"]], fmt="o", capsize=4, color=PALETTE[3])
ax[1].axvline(0, color="k", lw=1); ax[1].set_yticks(range(len(d2))); ax[1].set_yticklabels(d2.index)
ax[1].set_xlabel("ΔCV-RMSE vs. median imputation (USD, negative = better)"); ax[1].set_title("(b) Downstream effect with 95 % CI")
fig.tight_layout(); savefig(fig, "fig05_imputation")
''')
code(r'''
best_row = imp_tab.loc[BEST_IMPUTE]; med_row = imp_tab.loc["median"]
sig = [s for s in STRATS if s != "median" and dtab.loc[s, "NB_p"] < 0.05]
print(f"Interpretation: '{BEST_IMPUTE}' recovers hidden bedroom counts with RMSE {best_row.RMSE:.1f} vs {med_row.RMSE:.1f} for median imputation "
      f"({(1-best_row.RMSE/med_row.RMSE)*100:.0f}% lower). Strategies with a significant downstream CV difference vs median (Nadeau-Bengio p<0.05): {sig if sig else 'none'}.")
print(f"Only {R['quality']['missing_pct_total_bedrooms']:.2f}% of rows are affected, so the downstream effect is expected to be small.")
''')

# ───────────────────────────── O2 ─────────────────────────────
md(r'''
---
# Objective 2 – Income as a determinant of house value
**Research question.** How strong, how steep and how *linear* is the relationship between median income and median house value?  
**Relevance.** Affordability analysis and policy rely on the income–price elasticity.  
**Success measures.** Pearson/Spearman correlation with bootstrap 95 % CI; OLS slope and R²; F-test for non-linearity; elasticity from a log–log model; Breusch–Pagan test for heteroscedasticity; sensitivity to censoring.
''')
code(r'''
d = train_df[["median_income", TARGET]].rename(columns={"median_income": "income", TARGET: "value"})
rng = np.random.default_rng(RANDOM_STATE); NB = 200 if QUICK else 1000
pr, sr, kt = stats.pearsonr(d.income, d.value), stats.spearmanr(d.income, d.value), stats.kendalltau(d.income, d.value)
bp_, bs_ = [], []
for _ in range(NB):
    i = rng.integers(0, len(d), len(d)); s_ = d.iloc[i]
    bp_.append(np.corrcoef(s_.income, s_.value)[0, 1]); bs_.append(stats.spearmanr(s_.income, s_.value)[0])
ci_p, ci_s = np.percentile(bp_, [2.5, 97.5]), np.percentile(bs_, [2.5, 97.5])

m1 = smf.ols("value ~ income", d).fit()
m2 = smf.ols("value ~ income + I(income**2)", d).fit()
m3 = smf.ols("np.log(value) ~ np.log(income)", d).fit()
f_cmp = sm.stats.anova_lm(m1, m2)
bp1 = sm.stats.diagnostic.het_breuschpagan(m1.resid, m1.model.exog); bp3 = sm.stats.diagnostic.het_breuschpagan(m3.resid, m3.model.exog)
du = d[d.value < CAP_VALUE]; m1u = smf.ols("value ~ income", du).fit()

o2 = {"pearson": float(pr[0]), "pearson_ci": ci_p.tolist(), "spearman": float(sr[0]), "spearman_ci": ci_s.tolist(), "kendall": float(kt[0]),
      "ols_slope": float(m1.params["income"]), "ols_slope_ci": m1.conf_int().loc["income"].tolist(), "ols_intercept": float(m1.params["Intercept"]),
      "ols_r2": float(m1.rsquared), "quad_r2": float(m2.rsquared), "quad_F": float(f_cmp["F"].iloc[1]), "quad_F_p": float(f_cmp["Pr(>F)"].iloc[1]),
      "quad_b2": float(m2.params["I(income ** 2)"]),
      "loglog_elasticity": float(m3.params["np.log(income)"]), "loglog_elasticity_ci": m3.conf_int().loc["np.log(income)"].tolist(), "loglog_r2": float(m3.rsquared),
      "bp_linear_LM": float(bp1[0]), "bp_linear_p": float(bp1[1]), "bp_loglog_LM": float(bp3[0]), "bp_loglog_p": float(bp3[1]),
      "uncensored_slope": float(m1u.params["income"]), "uncensored_r2": float(m1u.rsquared), "n_uncensored": int(len(du)),
      "bootstrap_resamples": NB}
R["o2"] = o2
display(pd.Series(o2).to_frame("value").T.T.astype(str))
''')
code(r'''
dec = d.assign(decile=pd.qcut(d.income, 10, labels=False) + 1).groupby("decile").agg(
    income_min=("income", "min"), income_max=("income", "max"), n=("value", "size"),
    median_value=("value", "median"), mean_value=("value", "mean"), sd_value=("value", "std"),
    capped_pct=("value", lambda v: (v >= CAP_VALUE).mean() * 100)).round(2)
display(dec); R["o2"]["deciles"] = dec.to_dict("index")
R["o2"]["decile_ratio_top_bottom"] = float(dec.median_value.iloc[-1] / dec.median_value.iloc[0]); dump()
''')
code(r'''
# Figure 6 – shape of the relationship (linear, quadratic, LOWESS, log-log)
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
hb = ax[0].hexbin(d.income, d.value, gridsize=55, bins="log", cmap="Blues", mincnt=1); plt.colorbar(hb, ax=ax[0], label="log10(count)")
xs = np.linspace(d.income.min(), d.income.max(), 200)
ax[0].plot(xs, m1.params["Intercept"] + m1.params["income"] * xs, "r-", lw=2, label=f"OLS linear (R²={m1.rsquared:.2f})")
ax[0].plot(xs, m2.params["Intercept"] + m2.params["income"] * xs + m2.params["I(income ** 2)"] * xs ** 2, "g--", lw=2, label=f"OLS quadratic (R²={m2.rsquared:.2f})")
lw = lowess(d.value, d.income, frac=0.25, it=1, delta=0.02 * (d.income.max() - d.income.min())); ax[0].plot(lw[:, 0], lw[:, 1], "k-", lw=2, label="LOWESS")
ax[0].axhline(CAP_VALUE, color="orange", ls=":", lw=1.5, label="cap $500,001"); ax[0].set_ylim(0, 540000)
ax[0].set_xlabel("median income (×$10,000)"); ax[0].set_ylabel("median house value (USD)"); ax[0].set_title("(a) Income vs. house value"); ax[0].legend(loc="upper left")
ax[1].hexbin(np.log(d.income), np.log(d.value), gridsize=55, bins="log", cmap="Blues", mincnt=1)
xl = np.linspace(np.log(d.income).min(), np.log(d.income).max(), 100); ax[1].plot(xl, m3.params["Intercept"] + m3.params["np.log(income)"] * xl, "r-", lw=2)
ax[1].set_xlabel("ln(median income)"); ax[1].set_ylabel("ln(median house value)"); ax[1].set_title(f"(b) Log–log fit: elasticity = {m3.params['np.log(income)']:.2f}, R² = {m3.rsquared:.2f}")
fig.tight_layout(); savefig(fig, "fig06_income_relationship")
''')
code(r'''
# Figure 7 – heteroscedasticity and income deciles
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.5))
ax[0].hexbin(m1.fittedvalues, m1.resid, gridsize=50, bins="log", cmap="Purples", mincnt=1); ax[0].axhline(0, color="k", lw=1)
ax[0].set_xlabel("fitted value (USD)"); ax[0].set_ylabel("residual (USD)"); ax[0].set_title(f"(a) Linear-model residuals (Breusch-Pagan p = {bp1[1]:.1e})")
ax[1].bar(dec.index, dec.median_value / 1000, color=PALETTE[0], label="median value")
ax[1].errorbar(dec.index, dec.mean_value / 1000, yerr=dec.sd_value / 1000, fmt="o", color="red", capsize=3, label="mean ± 1 sd")
ax[1].set_xlabel("income decile (1 = lowest)"); ax[1].set_ylabel("house value (thousand USD)"); ax[1].set_title("(b) House value by income decile"); ax[1].legend(); ax[1].set_xticks(dec.index)
fig.tight_layout(); savefig(fig, "fig07_income_deciles_residuals")
''')
code(r'''
print(f"Interpretation: Pearson r = {pr[0]:.3f} (95% CI {ci_p[0]:.3f}-{ci_p[1]:.3f}); Spearman ρ = {sr[0]:.3f}. A 1-unit (=$10,000) rise in income is associated with +${m1.params['income']:,.0f} in value; income alone explains {m1.rsquared*100:.1f}% of variance.")
print(f"Non-linearity: adding income² changes R² from {m1.rsquared:.3f} to {m2.rsquared:.3f} (F-test p = {f_cmp['Pr(>F)'].iloc[1]:.2g}).")
print(f"Elasticity: a 1% rise in income ↔ {m3.params['np.log(income)']:.2f}% rise in value. Heteroscedasticity (BP) linear p={bp1[1]:.2g}; log-log p={bp3[1]:.2g}.")
print(f"Censoring sensitivity: slope on uncapped rows = {m1u.params['income']:,.0f} vs {m1.params['income']:,.0f} (all rows).")
''')

# ───────────────────────────── O3 ─────────────────────────────
md(r'''
---
# Objective 3 – Geography: does location explain value, and do spatial features help prediction?
**Research question.** (a) Do house values differ significantly between ocean-proximity classes, and by how much? (b) Are values spatially autocorrelated (neighbouring districts similar)? (c) Do distance-to-major-city features improve out-of-sample prediction?  
**Relevance.** "Location, location, location" is the central valuation principle; quantifying it guides feature design and land-use policy.  
**Success measures.** Kruskal–Wallis H with effect size ε² and Holm-corrected pairwise tests; Moran's *I* with permutation p-value; ΔCV-RMSE (95 % CI, corrected *t*-test) with/without spatial features.
''')
code(r'''
# 3a. Group differences – Kruskal-Wallis (non-parametric: target is skewed and censored)
tr = train_df.copy()
grp = tr.groupby("ocean_proximity")[TARGET]
gtab = grp.agg(n="count", mean="mean", median="median", q1=lambda v: v.quantile(.25), q3=lambda v: v.quantile(.75), sd="std").round(0)
gtab["capped_pct"] = grp.apply(lambda v: (v >= CAP_VALUE).mean() * 100).round(1)
display(gtab.sort_values("median"))
groups = [g.values for _, g in grp]
H, p_kw = stats.kruskal(*groups); k_, n_ = len(groups), len(tr); eps2 = (H - k_ + 1) / (n_ - k_)
print(f"Kruskal-Wallis H={H:.1f}, p={p_kw:.3g}, epsilon^2={eps2:.3f}")
names = sorted(tr.ocean_proximity.unique()); pairs = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = tr.loc[tr.ocean_proximity == names[i], TARGET], tr.loc[tr.ocean_proximity == names[j], TARGET]
        U, p = stats.mannwhitneyu(a, b); pairs.append({"A": names[i], "B": names[j], "n_A": len(a), "n_B": len(b), "median_diff": a.median() - b.median(), "rank_biserial_r": 2 * U / (len(a) * len(b)) - 1, "p_raw": p})
pw = pd.DataFrame(pairs); pw["p_holm"] = multipletests(pw.p_raw, method="holm")[1]; pw["significant_5%"] = pw.p_holm < 0.05
display(pw.round(4))
R["o3"] = {"group_table": gtab.to_dict("index"), "kruskal_H": float(H), "kruskal_p": float(p_kw), "epsilon2": float(eps2), "pairwise": pw.round(6).to_dict("records")}
''')
code(r'''
# 3b. Spatial autocorrelation of the target (Moran's I, k = 8 nearest neighbours, haversine distance)
NP = 99 if QUICK else 499
I_t, EI, zI, pI, zc, lag = morans_i(tr[TARGET].values, tr.latitude.values, tr.longitude.values, k=8, n_perm=NP)
R["o3"]["moran_I"] = {"I": I_t, "expected": EI, "z": zI, "p": pI, "k": 8, "n_perm": NP}
print(f"Moran's I = {I_t:.3f}  (expected under no autocorrelation = {EI:.5f}); permutation z = {zI:.1f}, p = {pI:.4f}")
dump()
''')
md(r'''
### Feature-set ablation (supports O3 *and* O4)
Four feature sets are compared with two very different learners (regularised linear model *Ridge*; gradient boosting), using identical folds:

| Set | Content |
|---|---|
| F0 | raw variables only |
| F1 | F0 + ratio features (rooms/household, bedrooms/room, people/household) |
| F2 | F0 + spatial distance features (5 city hubs + nearest hub) |
| F3 | F0 + ratio + spatial |
''')
code(r'''
FEATSETS = {"F0 raw": (False, False), "F1 +ratios": (True, False), "F2 +spatial": (False, True), "F3 +ratios+spatial": (True, True)}
def make_model(kind):
    if kind == "Ridge": return Ridge(alpha=1.0)      # raw target: a log-target handicaps dollar-scale RMSE (tested, see O5)
    return hgb()
abl_rmse, abl_r2 = {}, {}
t0 = time.time()
for fs, (r_, s_) in FEATSETS.items():
    for kind in ["Ridge", "HGB"]:
        pipe = Pipeline([("prep", build_preprocessor(BEST_IMPUTE, r_, s_)), ("m", make_model(kind))])
        cvres = cross_validate(pipe, X_train, y_train, cv=CV_ABL, scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"}, n_jobs=1)
        abl_rmse[(fs, kind)], abl_r2[(fs, kind)] = -cvres["test_rmse"], cvres["test_r2"]
print(f"ablation done in {time.time()-t0:.0f}s")
ab = pd.DataFrame({f"{fs} | {k}": {"mean_RMSE": v.mean(), "sd_RMSE": v.std(ddof=1), "mean_R2": abl_r2[(fs, k)].mean()} for (fs, k), v in abl_rmse.items()}).T
display(ab.round(4)); R["ablation"] = ab.round(5).to_dict("index")

def contrast(new, old, kind):
    s = paired_summary(abl_rmse[(new, kind)], abl_rmse[(old, kind)])
    s.update({"model": kind, "from": old, "to": new, "pct_change": s["mean_diff"] / abl_rmse[(old, kind)].mean() * 100,
              "delta_R2": float(abl_r2[(new, kind)].mean() - abl_r2[(old, kind)].mean())}); return s
spatial_c = pd.DataFrame([contrast("F2 +spatial", "F0 raw", m) for m in ["Ridge", "HGB"]] + [contrast("F3 +ratios+spatial", "F1 +ratios", m) for m in ["Ridge", "HGB"]])
ratio_c   = pd.DataFrame([contrast("F1 +ratios", "F0 raw", m) for m in ["Ridge", "HGB"]] + [contrast("F3 +ratios+spatial", "F2 +spatial", m) for m in ["Ridge", "HGB"]])
display(spatial_c.round(4)); display(ratio_c.round(4))
R["o3"]["spatial_contrasts"] = spatial_c.round(6).to_dict("records"); R["o4"] = {"ratio_contrasts": ratio_c.round(6).to_dict("records")}
# decision rule: feature set with the lowest mean CV-RMSE for the gradient-boosting learner
BEST_FS = min(FEATSETS, key=lambda f: abl_rmse[(f, "HGB")].mean()); RATIOS, SPATIAL = FEATSETS[BEST_FS]
R["ablation_choice"] = {"best_featureset": BEST_FS, "ratios": RATIOS, "spatial": SPATIAL}; print("Selected feature set:", BEST_FS); dump()
''')
code(r'''
# Figure 8 – value by ocean proximity ; Figure 9 – Moran scatter ; Figure 10 – spatial-feature gain
order = tr.groupby("ocean_proximity")[TARGET].median().sort_values().index
fig, ax = plt.subplots(figsize=(8.5, 4.6))
sns.violinplot(data=tr, x="ocean_proximity", y=TARGET, order=order, inner="quartile", cut=0, palette="deep", ax=ax, density_norm="width")
for i, o in enumerate(order): ax.text(i, 535000, f"n={int((tr.ocean_proximity==o).sum())}\nmed=${tr.loc[tr.ocean_proximity==o, TARGET].median()/1000:.0f}k", ha="center", fontsize=8)
ax.set_ylim(0, 600000); ax.set_xlabel("ocean proximity"); ax.set_ylabel("median house value (USD)")
ax.set_title(f"House value by ocean proximity (Kruskal-Wallis H = {H:.0f}, ε² = {eps2:.2f})"); savefig(fig, "fig08_ocean_proximity")

fig, ax = plt.subplots(figsize=(6, 5.2)); sidx = np.random.default_rng(1).choice(len(zc), min(6000, len(zc)), replace=False)
ax.hexbin(zc[sidx] / 1000, lag[sidx] / 1000, gridsize=45, bins="log", cmap="Greens", mincnt=1)
xx = np.array([zc.min(), zc.max()]) / 1000; ax.plot(xx, I_t * xx, "r-", lw=2, label=f"slope = Moran's I = {I_t:.2f}")
ax.axhline(0, color="k", lw=.8); ax.axvline(0, color="k", lw=.8); ax.set_xlabel("value − mean (thousand USD)"); ax.set_ylabel("average of 8 neighbours (thousand USD)")
ax.set_title("Moran scatter-plot of house value"); ax.legend(); savefig(fig, "fig09_moran_scatter")

fig, ax = plt.subplots(figsize=(7, 4.2)); w = 0.35
for j, (lab, rows) in enumerate([("F0 → F2 (raw + spatial)", spatial_c.iloc[[0, 1]]), ("F1 → F3 (ratios + spatial)", spatial_c.iloc[[2, 3]])]):
    ax.bar(np.arange(2) + j * w, rows.mean_diff, w, yerr=[rows.mean_diff - rows.ci_lo, rows.ci_hi - rows.mean_diff], capsize=4, label=lab, color=PALETTE[j])
ax.axhline(0, color="k", lw=1); ax.set_xticks(np.arange(2) + w / 2); ax.set_xticklabels(["Ridge", "Gradient boosting"])
ax.set_ylabel("ΔCV-RMSE (USD); negative = improvement"); ax.set_title("Effect of adding spatial distance features (95 % CI)"); ax.legend(); savefig(fig, "fig10_spatial_features_gain")
''')
code(r'''
top = gtab["median"].idxmax(); low = gtab["median"].idxmin()
print(f"Interpretation (a): median value ranges from ${gtab.loc[low,'median']:,.0f} ({low}) to ${gtab.loc[top,'median']:,.0f} ({top}); epsilon²={eps2:.2f} -> {'large' if eps2>0.14 else 'moderate' if eps2>0.06 else 'small'} effect.")
print(f"Significant pairs (Holm): {int(pw['significant_5%'].sum())} of {len(pw)}.")
print(f"Interpretation (b): Moran's I = {I_t:.3f} (p={pI:.3f}) -> {'strong positive spatial clustering of values' if I_t>0.3 and pI<0.05 else 'weak/no clustering'}.")
for _, r_ in spatial_c.iterrows():
    print(f"Interpretation (c): {r_['from']} -> {r_['to']} [{r_['model']}]: ΔRMSE = {r_.mean_diff:,.0f} ({r_['pct_change']:+.1f}%), 95% CI [{r_.ci_lo:,.0f}, {r_.ci_hi:,.0f}], NB p = {r_.NB_p:.3g}")
''')

# ───────────────────────────── O4 ─────────────────────────────
md(r'''
---
# Objective 4 – Do engineered ratio features add predictive value?
**Research question.** Raw `total_rooms`, `total_bedrooms`, `population`, `households` are *block-group totals* and therefore mostly measure district **size**. Do per-household ratios (rooms/household, bedrooms/room, people/household), which describe **housing character**, carry more information about value?  
**Relevance.** Good feature engineering is usually the cheapest way to improve a model and it makes models interpretable.  
**Success measures.** Spearman correlation with the target (raw vs. ratio); ΔCV-RMSE and ΔR² with corrected resampled *t*-test (from the shared ablation above).
''')
code(r'''
fe_tr = FeatureEngineer(ratios=True, spatial=False).transform(BedroomImputer(BEST_IMPUTE).fit(X_train).transform(X_train))
cand = ["total_rooms", "total_bedrooms", "population", "households", "rooms_per_household", "bedrooms_per_room", "population_per_household"]
rows = []
for c in cand:
    rho, p = stats.spearmanr(fe_tr[c], y_train); rows.append({"feature": c, "type": "engineered" if "per" in c else "raw count", "spearman_rho": rho, "abs_rho": abs(rho), "p_value": p, "skew": fe_tr[c].skew()})
cor_tab = pd.DataFrame(rows); display(cor_tab.round(4))
R["o4"]["correlations"] = cor_tab.round(5).to_dict("records")
R["o4"]["rooms_per_household_max"] = float(fe_tr.rooms_per_household.max()); R["o4"]["rooms_per_household_p99"] = float(fe_tr.rooms_per_household.quantile(.99))
# multicollinearity relief: correlation between raw counts vs between ratios
raw_cc = fe_tr[["total_rooms", "total_bedrooms", "population", "households"]].corr(method="spearman").values[np.triu_indices(4, 1)]
R["o4"]["raw_counts_mean_abs_rho"] = float(np.abs(raw_cc).mean()); print("mean |ρ| between raw counts:", round(np.abs(raw_cc).mean(), 3)); dump()
''')
code(r'''
# Figure 11 – ratio features: correlation with target and out-of-sample gain
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
ct_ = cor_tab.sort_values("spearman_rho"); ax[0].barh(ct_.feature, ct_.spearman_rho, color=[PALETTE[3] if t == "engineered" else PALETTE[0] for t in ct_.type])
ax[0].axvline(0, color="k", lw=1); ax[0].set_xlabel("Spearman ρ with house value"); ax[0].set_title("(a) Raw counts (blue) vs. engineered ratios (red)")
w = 0.35
for j, (lab, rows) in enumerate([("F0 → F1 (raw + ratios)", ratio_c.iloc[[0, 1]]), ("F2 → F3 (spatial + ratios)", ratio_c.iloc[[2, 3]])]):
    ax[1].bar(np.arange(2) + j * w, rows.mean_diff, w, yerr=[rows.mean_diff - rows.ci_lo, rows.ci_hi - rows.mean_diff], capsize=4, label=lab, color=PALETTE[j + 2])
ax[1].axhline(0, color="k", lw=1); ax[1].set_xticks(np.arange(2) + w / 2); ax[1].set_xticklabels(["Ridge", "Gradient boosting"])
ax[1].set_ylabel("ΔCV-RMSE (USD); negative = improvement"); ax[1].set_title("(b) Out-of-sample effect of ratio features (95 % CI)"); ax[1].set_ylim(top=ax[1].get_ylim()[1] + 500); ax[1].legend(loc="upper left")
fig.tight_layout(); savefig(fig, "fig11_ratio_features")
''')
code(r'''
for _, r_ in ratio_c.iterrows():
    print(f"{r_['from']} -> {r_['to']} [{r_['model']}]: ΔRMSE = {r_.mean_diff:,.0f} ({r_['pct_change']:+.1f}%), ΔR² = {r_.delta_R2:+.4f}, NB p = {r_.NB_p:.3g}")
best_eng = cor_tab[cor_tab.type == "engineered"].sort_values("abs_rho").iloc[-1]; best_raw = cor_tab[cor_tab.type == "raw count"].sort_values("abs_rho").iloc[-1]
print(f"Strongest engineered |ρ| = {best_eng.abs_rho:.3f} ({best_eng.feature}) vs strongest raw count |ρ| = {best_raw.abs_rho:.3f} ({best_raw.feature}).")
''')

# ───────────────────────────── O5 ─────────────────────────────
md(r'''
---
# Objective 5 – Fair comparison of ML models and evidence-based model selection
**Research question.** Which model family predicts district house value best, and is the winner *significantly* better than the runner-up?  
**Relevance.** Automated valuation models must be accurate and defensible; choosing by evidence (not by habit) is the core of applied ML.  
**Protocol (leakage-free).**
1. Same pre-processing pipeline (imputer → feature engineering → log/scale/one-hot) for every model, fitted *inside each CV fold*.
2. Hyper-parameters tuned **only on the training set** with 3-fold CV (grid search or randomised search, scoring = RMSE).
3. Tuned models re-evaluated with 5-fold CV on the training set → **model selection uses CV-RMSE only**.
4. The held-out test set is touched **once** for the final, unbiased estimate; significance via corrected resampled *t*-test (CV) and paired bootstrap (test).

| Model | Why included |
|---|---|
| Dummy (mean) | Naïve baseline – any real model must beat it |
| Ridge | Interpretable linear baseline with L2 regularisation (raw target) |
| Ridge (log-target) | Same, fitted on ln(price) and back-transformed - tests whether the log transform helps in dollar-scale RMSE |
| k-NN | Instance-based; naturally exploits spatial neighbourhoods |
| Random Forest | Bagged trees; strong, low-tuning non-linear model |
| HistGradientBoosting | Fast gradient boosting (scikit-learn) |
| XGBoost | Regularised gradient boosting; state-of-the-art for tabular data |
''')
code(r'''
from sklearn.base import clone
print(f"Selected from O1: imputer = {BEST_IMPUTE} | from ablation: feature set = {BEST_FS} (ratios={RATIOS}, spatial={SPATIAL})")
def pipe_of(est): return Pipeline([("prep", build_preprocessor(BEST_IMPUTE, RATIOS, SPATIAL)), ("model", est)])
CV_TUNE  = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
CV_FINAL = KFold(n_splits=K_SPLITS, shuffle=True, random_state=RANDOM_STATE)
NT = 60 if QUICK else 150
SPECS = {
 "Dummy (mean)":  (DummyRegressor(strategy="mean"), None, 0),
 "Ridge":         (Ridge(), {"model__alpha": [0.01, 0.1, 1, 10, 100]}, 0),
 "Ridge (log-target)": (TransformedTargetRegressor(Ridge(), func=np.log, inverse_func=np.exp), {"model__regressor__alpha": [0.01, 0.1, 1, 10, 100]}, 0),
 "k-NN":          (KNeighborsRegressor(n_jobs=1), {"model__n_neighbors": [3, 5, 10, 15, 25], "model__weights": ["uniform", "distance"]}, 0),
 "Random Forest": (RandomForestRegressor(n_estimators=NT, random_state=RANDOM_STATE, n_jobs=1),
                   {"model__max_depth": [None, 12, 20, 30], "model__min_samples_leaf": [1, 2, 4], "model__max_features": [0.33, 0.5, 0.8]}, 2 if QUICK else 6),
 "HistGB":        (HistGradientBoostingRegressor(random_state=RANDOM_STATE),
                   {"model__learning_rate": [0.03, 0.05, 0.1], "model__max_iter": [300, 500, 800], "model__max_leaf_nodes": [15, 31, 63],
                    "model__min_samples_leaf": [20, 40], "model__l2_regularization": [0, 1, 5]}, 3 if QUICK else 12),
 "XGBoost":       (XGBRegressor(tree_method="hist", n_jobs=1, random_state=RANDOM_STATE),
                   {"model__n_estimators": [400, 800], "model__max_depth": [4, 6, 8], "model__learning_rate": [0.03, 0.05, 0.1],
                    "model__subsample": [0.7, 0.9], "model__colsample_bytree": [0.7, 0.9], "model__min_child_weight": [1, 5]}, 3 if QUICK else 10),
}
tuned, best_params, tune_info = {}, {}, {}
for name, (est, space, n_iter) in SPECS.items():
    t0 = time.time(); pipe = pipe_of(est)
    if space is not None:
        size = int(np.prod([len(v) for v in space.values()]))
        S = (GridSearchCV(pipe, space, cv=CV_TUNE, scoring="neg_root_mean_squared_error", n_jobs=1, refit=False) if n_iter == 0 else
             RandomizedSearchCV(pipe, space, n_iter=min(n_iter, size), cv=CV_TUNE, scoring="neg_root_mean_squared_error", n_jobs=1, refit=False, random_state=RANDOM_STATE))
        S.fit(X_train, y_train); pipe.set_params(**S.best_params_)
        best_params[name] = {k.split("model__")[1]: (v.item() if hasattr(v, "item") else v) for k, v in S.best_params_.items()}
        tune_info[name] = {"search": "grid" if n_iter == 0 else "randomized", "candidates": size if n_iter == 0 else min(n_iter, size), "space_size": size, "best_tuning_RMSE": float(-S.best_score_), "seconds": time.time() - t0}
    else: best_params[name] = {}; tune_info[name] = {"search": "none", "candidates": 0, "space_size": 0, "best_tuning_RMSE": None, "seconds": 0}
    tuned[name] = pipe; print(f"{name:14s} tuned in {time.time()-t0:6.0f}s  params={best_params[name]}")
R["o5"] = {"best_params": best_params, "tuning": tune_info}; dump()
''')
code(r'''
# 5-fold CV of every tuned model on the TRAINING set (basis for model selection)
cvr, fold_rmse = {}, {}
for name, pipe in tuned.items():
    t0 = time.time()
    r_ = cross_validate(pipe, X_train, y_train, cv=CV_FINAL, scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}, n_jobs=1)
    fold_rmse[name] = -r_["test_rmse"]
    cvr[name] = {"CV_RMSE": -r_["test_rmse"].mean(), "CV_RMSE_sd": r_["test_rmse"].std(ddof=1), "CV_MAE": -r_["test_mae"].mean(), "CV_R2": r_["test_r2"].mean(), "fit_s_per_fold": r_["fit_time"].mean()}
    print(f"{name:14s} CV-RMSE {cvr[name]['CV_RMSE']:>10,.0f} ± {cvr[name]['CV_RMSE_sd']:,.0f}   R² {cvr[name]['CV_R2']:.3f}   ({time.time()-t0:.0f}s)")
cv_tab = pd.DataFrame(cvr).T
BEST_NAME = cv_tab.drop(index="Dummy (mean)").CV_RMSE.idxmin(); RANK = cv_tab.CV_RMSE.sort_values().index.tolist(); RUNNER = RANK[1]
print("\nBest model by CV-RMSE:", BEST_NAME, "| runner-up:", RUNNER)
pd.DataFrame(fold_rmse).to_csv(RESULT_DIR / "cv_fold_rmse.csv", index=False)
''')
code(r'''
# Final test-set evaluation (touched ONCE, after selection) + timing
fitted, test_pred, rows = {}, {}, []
for name, pipe in tuned.items():
    t0 = time.time(); m = clone(pipe).fit(X_train, y_train); ft = time.time() - t0
    t0 = time.time(); p = m.predict(X_test); pt = time.time() - t0
    fitted[name], test_pred[name] = m, p
    rows.append({"model": name, **regression_metrics(y_test, p), "train_fit_s": ft, "predict_s_per_4k": pt})
test_tab = pd.DataFrame(rows).set_index("model")
comp = cv_tab.join(test_tab).sort_values("CV_RMSE"); display(comp.round(3))
comp.round(4).to_csv(RESULT_DIR / "model_comparison.csv")
R["o5"].update({"cv_table": cv_tab.round(4).to_dict("index"), "test_table": test_tab.round(4).to_dict("index"), "best_model": BEST_NAME, "runner_up": RUNNER, "rank_by_cv": RANK})

# significance: best vs every other model
sig = []
for name in tuned:
    if name == BEST_NAME: continue
    ps = paired_summary(fold_rmse[BEST_NAME], fold_rmse[name]); d, lo, hi = paired_bootstrap_rmse_diff(y_test.values, test_pred[BEST_NAME], test_pred[name])
    sig.append({"vs": name, "cv_mean_diff": ps["mean_diff"], "NB_t": ps["NB_t"], "NB_p": ps["NB_p"], "test_RMSE_diff": d, "boot_ci_lo": lo, "boot_ci_hi": hi})
sig = pd.DataFrame(sig).set_index("vs"); display(sig.round(4)); R["o5"]["significance"] = sig.round(6).to_dict("index"); dump()
''')
code(r'''
# Figure 12 – CV-fold RMSE and held-out test RMSE for all models
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6)); ordr = comp.index.tolist()
ax[0].boxplot([fold_rmse[n] / 1000 for n in ordr], labels=ordr, patch_artist=True, boxprops=dict(facecolor=PALETTE[0], alpha=.6)); ax[0].tick_params(axis="x", rotation=25)
for i, n in enumerate(ordr): ax[0].scatter(np.full(len(fold_rmse[n]), i + 1), fold_rmse[n] / 1000, color="k", s=12, zorder=3)
ax[0].set_ylabel("RMSE per CV fold (thousand USD)"); ax[0].set_title(f"(a) {K_SPLITS}-fold CV on the training set")
bars = ax[1].bar(ordr, comp.RMSE / 1000, color=[PALETTE[2] if n == BEST_NAME else PALETTE[0] for n in ordr]); ax[1].tick_params(axis="x", rotation=25)
for b, n in zip(bars, ordr): ax[1].text(b.get_x() + b.get_width() / 2, b.get_height() + .8, f"{comp.loc[n,'RMSE']/1000:.1f}k\nR²={comp.loc[n,'R2']:.2f}", ha="center", fontsize=8)
ax[1].set_ylabel("test RMSE (thousand USD)"); ax[1].set_title("(b) Held-out test set"); ax[1].set_ylim(0, comp.RMSE.max() / 1000 * 1.18)
fig.tight_layout(); savefig(fig, "fig12_model_comparison")
''')
code(r'''
# Figure 13 – predicted vs. actual (best model, test set) ; Figure 14 – learning curve
best = fitted[BEST_NAME]; pred = test_pred[BEST_NAME]
fig, ax = plt.subplots(figsize=(6.2, 5.6))
hb = ax.hexbin(y_test / 1000, pred / 1000, gridsize=50, bins="log", cmap="viridis", mincnt=1); plt.colorbar(hb, label="log10(count)")
ax.plot([0, 520], [0, 520], "r--", lw=1.5, label="perfect prediction"); ax.axvline(CAP_VALUE / 1000, color="orange", ls=":", lw=1.5, label="cap $500,001")
ax.set_xlabel("actual median house value (thousand USD)"); ax.set_ylabel("predicted (thousand USD)"); ax.set_title(f"{BEST_NAME}: predicted vs. actual (test set)"); ax.legend(loc="upper left"); savefig(fig, "fig13_pred_vs_actual")

sizes = [0.3, 0.6, 1.0] if QUICK else [0.2, 0.4, 0.6, 0.8, 1.0]
ts, trs, vas = learning_curve(clone(tuned[BEST_NAME]), X_train, y_train, train_sizes=sizes, cv=KFold(3, shuffle=True, random_state=RANDOM_STATE), scoring="neg_root_mean_squared_error", n_jobs=1)
lc = pd.DataFrame({"n": ts, "train_RMSE": -trs.mean(1), "val_RMSE": -vas.mean(1), "val_sd": vas.std(1)}); display(lc.round(1))
R["o5"]["learning_curve"] = lc.round(2).to_dict("records")
fig, ax = plt.subplots(figsize=(6.6, 4.4)); ax.plot(lc.n, lc.train_RMSE / 1000, "o-", label="training RMSE"); ax.plot(lc.n, lc.val_RMSE / 1000, "s-", label="validation RMSE")
ax.fill_between(lc.n, (lc.val_RMSE - lc.val_sd) / 1000, (lc.val_RMSE + lc.val_sd) / 1000, alpha=.2)
ax.set_xlabel("training-set size"); ax.set_ylabel("RMSE (thousand USD)"); ax.set_title(f"Learning curve – {BEST_NAME}"); ax.legend(); savefig(fig, "fig14_learning_curve"); dump()
''')
code(r'''
b_ = comp.loc[BEST_NAME]; d_ = comp.loc["Dummy (mean)"]
print(f"Interpretation: {BEST_NAME} reaches test RMSE ${b_.RMSE:,.0f}, MAE ${b_.MAE:,.0f}, R² {b_.R2:.3f}; the mean-baseline has RMSE ${d_.RMSE:,.0f} -> {(1-b_.RMSE/d_.RMSE)*100:.0f}% lower error.")
r_ = sig.loc[RUNNER]; print(f"vs runner-up {RUNNER}: ΔRMSE(test) = {r_.test_RMSE_diff:,.0f} (95% bootstrap CI {r_.boot_ci_lo:,.0f} to {r_.boot_ci_hi:,.0f}); corrected-t p (CV) = {r_.NB_p:.3g}.")
gap = lc.val_RMSE.iloc[-1] - lc.train_RMSE.iloc[-1]; slope = lc.val_RMSE.iloc[-2] - lc.val_RMSE.iloc[-1]
print(f"Learning curve: train–validation gap = ${gap:,.0f}; last increase in data reduced validation RMSE by ${slope:,.0f}.")
''')

# ───────────────────────────── O6 ─────────────────────────────
md(r'''
---
# Objective 6 – Error analysis, effect of censoring, and explainability
**Research question.** Where does the best model fail, how much of its error is due to the \$500,001 target cap, is the remaining error spatially structured, and which variables drive predictions?  
**Relevance.** A model is only deployable if its weaknesses are known and its behaviour can be explained to stakeholders.  
**Success measures.** Residual diagnostics; segment-wise RMSE/MedAPE; metric change when censored rows are excluded; Moran's *I* of residuals; permutation importance and SHAP (concentration of importance in the top-3 drivers); partial-dependence curves.
''')
code(r'''
res = y_test.values - pred
jb = stats.jarque_bera(res); sp_h = stats.spearmanr(np.abs(res), pred)
R["o6"] = {"bias_mean_residual": float(res.mean()), "residual_sd": float(res.std()), "residual_skew": float(stats.skew(res)), "residual_kurtosis": float(stats.kurtosis(res)),
           "jarque_bera": float(jb[0]), "jarque_bera_p": float(jb[1]), "abs_resid_vs_pred_spearman": float(sp_h[0])}
print({k: round(v, 4) for k, v in R["o6"].items()})
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
ax[0].hexbin(pred / 1000, res / 1000, gridsize=50, bins="log", cmap="Purples", mincnt=1); ax[0].axhline(0, color="k", lw=1)
lw_ = lowess(res, pred, frac=0.3, it=0, delta=0.01 * np.ptp(pred)); ax[0].plot(lw_[:, 0] / 1000, lw_[:, 1] / 1000, "r-", lw=2, label="LOWESS of residuals"); ax[0].legend()
ax[0].set_xlabel("predicted value (thousand USD)"); ax[0].set_ylabel("residual (thousand USD)"); ax[0].set_title("(a) Residuals vs. predictions")
sns.histplot(res / 1000, bins=60, kde=True, ax=ax[1], color=PALETTE[0]); ax[1].set_xlabel("residual (thousand USD)"); ax[1].set_title(f"(b) Residual distribution (skew {stats.skew(res):.2f})")
stats.probplot(res / 1000, dist="norm", plot=ax[2]); ax[2].set_title("(c) Normal Q-Q plot of residuals"); ax[2].get_lines()[0].set_markersize(3)
fig.tight_layout(); savefig(fig, "fig15_residual_diagnostics")
''')
code(r'''
# Segment-wise error
te = X_test.copy(); te["y"] = y_test.values; te["pred"] = pred; te["res"] = res; te["ape"] = np.abs(res) / te.y
edges = train_df.median_income.quantile([0, .2, .4, .6, .8, 1]).values.copy(); edges[0], edges[-1] = -np.inf, np.inf
te["income_quintile"] = pd.cut(te.median_income, edges, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
te["value_band"] = pd.cut(te.y, [0, 150000, 300000, CAP_VALUE - 1, np.inf], labels=["<150k", "150-300k", "300-500k", "capped (500,001)"])
def seg(col):
    g = te.groupby(col, observed=True)
    return pd.DataFrame({"n": g.size(), "RMSE": g.apply(lambda d: np.sqrt((d.res ** 2).mean())), "MAE": g.apply(lambda d: d.res.abs().mean()), "bias": g.res.mean(), "MedAPE_%": g.ape.median() * 100})
seg_o, seg_i, seg_v = seg("ocean_proximity"), seg("income_quintile"), seg("value_band")
for t in (seg_o, seg_i, seg_v): display(t.round(1))
R["o6"]["segments"] = {"ocean_proximity": seg_o.round(3).to_dict("index"), "income_quintile": seg_i.round(3).to_dict("index"), "value_band": seg_v.round(3).to_dict("index")}
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
for a, t, title in zip(ax, (seg_o, seg_i, seg_v), ("(a) By ocean proximity", "(b) By income quintile", "(c) By true-value band")):
    bars = a.bar(t.index.astype(str), t.RMSE / 1000, color=PALETTE[0]); a.set_title(title); a.set_ylabel("RMSE (thousand USD)"); a.tick_params(axis="x", rotation=25)
    for b, (_, r_) in zip(bars, t.iterrows()): a.text(b.get_x() + b.get_width() / 2, b.get_height() + .5, f"MedAPE\n{r_['MedAPE_%']:.0f}%\nn={int(r_.n)}", ha="center", fontsize=7)
    a.set_ylim(0, t.RMSE.max() / 1000 * 1.35)
fig.tight_layout(); savefig(fig, "fig16_segment_errors")
''')
code(r'''
# Effect of target censoring (cap at $500,001)
unc = (y_test < CAP_VALUE).values
m_all, m_unc = regression_metrics(y_test, pred), regression_metrics(y_test[unc], pred[unc])
cap_rows = te[~unc]
pipe_u = clone(tuned[BEST_NAME]).fit(X_train[y_train < CAP_VALUE], y_train[y_train < CAP_VALUE]); pred_u = pipe_u.predict(X_test)
m_u_on_unc = regression_metrics(y_test[unc], pred_u[unc])
R["o6"]["censoring"] = {"n_test_capped": int((~unc).sum()), "pct_test_capped": float((~unc).mean() * 100), "metrics_all": m_all, "metrics_uncapped_rows": m_unc,
    "mean_pred_for_capped": float(cap_rows.pred.mean()), "mean_shortfall_capped": float((CAP_VALUE - cap_rows.pred).mean()), "share_capped_underpredicted": float((cap_rows.pred < CAP_VALUE).mean() * 100),
    "trained_without_capped_on_uncapped_test": m_u_on_unc,
    "rmse_share_from_capped_rows_pct": float((cap_rows.res ** 2).sum() / (te.res ** 2).sum() * 100)}
display(pd.DataFrame({"all test rows": m_all, "uncapped rows only": m_unc, "retrained w/o capped (uncapped rows)": m_u_on_unc}).round(3))
print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in R["o6"]["censoring"].items() if not isinstance(v, dict)})
# worst predictions
worst = te.reindex(te.res.abs().sort_values(ascending=False).index).head(10)[["longitude", "latitude", "median_income", "ocean_proximity", "y", "pred", "res"]]
display(worst.round(2)); R["o6"]["worst10"] = worst.round(2).to_dict("records")
''')
code(r'''
# Spatial structure of residuals (Moran's I) + map – Figure 17
I_r, EIr, zr, pr_, _, _ = morans_i(te.res.values, te.latitude.values, te.longitude.values, k=8, n_perm=99 if QUICK else 499)
R["o6"]["moran_residuals"] = {"I": I_r, "z": zr, "p": pr_}; print(f"Moran's I of test residuals = {I_r:.3f} (z={zr:.1f}, p={pr_:.4f})")
fig, ax = plt.subplots(figsize=(6.8, 6)); s = ax.scatter(te.longitude, te.latitude, c=te.res / 1000, cmap="RdBu", vmin=-150, vmax=150, s=10, alpha=.8)
plt.colorbar(s, label="residual = actual − predicted (thousand USD)"); ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title(f"Spatial map of test residuals (Moran's I = {I_r:.2f})")
savefig(fig, "fig17_residual_map"); dump()
''')
code(r'''
# Permutation importance (test set; drop in RMSE when a feature is shuffled) – single features and feature groups
rs = np.random.default_rng(RANDOM_STATE); base_rmse = np.sqrt(np.mean((y_test.values - best.predict(X_test)) ** 2)); NR = 3 if QUICK else 10
def perm_importance(cols):
    out = []
    for _ in range(NR):
        Xp = X_test.copy(); idx = rs.permutation(len(Xp)); Xp[cols] = X_test[cols].values[idx]
        out.append(np.sqrt(np.mean((y_test.values - best.predict(Xp)) ** 2)) - base_rmse)
    return np.mean(out), np.std(out, ddof=1)
feats = [c for c in X_test.columns]
pi = pd.DataFrame({f: perm_importance([f]) for f in feats}, index=["delta_RMSE", "sd"]).T.sort_values("delta_RMSE", ascending=False)
groups_ = {"location (lon + lat)": ["longitude", "latitude"], "size counts (rooms, bedrooms, pop., households)": ["total_rooms", "total_bedrooms", "population", "households"]}
pg = pd.DataFrame({g: perm_importance(c) for g, c in groups_.items()}, index=["delta_RMSE", "sd"]).T
display(pi.round(0)); display(pg.round(0))
pos = pi.delta_RMSE.clip(lower=0); R["o6"]["perm_importance"] = pi.round(2).to_dict("index"); R["o6"]["perm_groups"] = pg.round(2).to_dict("index")
R["o6"]["top3_share_pct"] = float(pos.head(3).sum() / pos.sum() * 100); R["o6"]["base_test_rmse"] = float(base_rmse)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
p_ = pi.iloc[::-1]; ax[0].barh(p_.index, p_.delta_RMSE / 1000, xerr=p_.sd / 1000, color=PALETTE[0], capsize=3); ax[0].set_xlabel("increase in test RMSE when shuffled (thousand USD)"); ax[0].set_title("(a) Single-feature permutation importance")
g_ = pg.sort_values("delta_RMSE"); ax[1].barh(g_.index, g_.delta_RMSE / 1000, xerr=g_.sd / 1000, color=PALETTE[2], capsize=3); ax[1].set_title("(b) Grouped permutation importance"); ax[1].set_xlabel("increase in test RMSE (thousand USD)")
fig.tight_layout(); savefig(fig, "fig18_permutation_importance"); dump()
''')
code(r'''
# SHAP values (TreeExplainer for tree ensembles; model-agnostic fallback otherwise)
import shap
prep_f, est_f = best.named_steps["prep"], best.named_steps["model"]
Xs = X_test.sample(n=min(1500, len(X_test)), random_state=RANDOM_STATE); Zt = prep_f.transform(Xs); Zt = Zt.toarray() if hasattr(Zt, "toarray") else np.asarray(Zt)
Zdf = pd.DataFrame(Zt, columns=prep_f[-1].get_feature_names_out())
try:
    sv = shap.TreeExplainer(est_f).shap_values(Zdf.values); shap_method = "TreeExplainer"   # array input: model was fitted on an array
except Exception as e:
    bg = shap.sample(Zdf, 100, random_state=0); sv = shap.Explainer(est_f.predict, bg)(Zdf.iloc[:300]).values; Zdf = Zdf.iloc[:300]; shap_method = "model-agnostic PermutationExplainer"
mshap = pd.Series(np.abs(sv).mean(0), index=Zdf.columns).sort_values(ascending=False); display(mshap.head(12).round(0).to_frame("mean |SHAP| (USD)").T)
R["o6"]["shap"] = {"method": shap_method, "n_samples": int(len(Zdf)), "mean_abs": mshap.round(2).to_dict()}
plt.figure(figsize=(8, 6)); shap.summary_plot(sv, Zdf, show=False, max_display=14, plot_size=None); fig = plt.gcf(); fig.axes[0].set_title("SHAP summary (feature value colour; impact on prediction, USD)", fontsize=10)
savefig(fig, "fig19_shap_summary"); dump()
''')
code(r'''
# Partial dependence (average effect of one/two raw features on the prediction)
Xp = X_test.sample(n=min(1000, len(X_test)), random_state=RANDOM_STATE)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
PartialDependenceDisplay.from_estimator(best, Xp, ["median_income"], ax=ax[0], grid_resolution=30, line_kw={"color": PALETTE[0], "lw": 2})
PartialDependenceDisplay.from_estimator(best, Xp, ["housing_median_age"], ax=ax[1], grid_resolution=30, line_kw={"color": PALETTE[2], "lw": 2})
from sklearn.inspection import partial_dependence
from sklearn.neighbors import NearestNeighbors
pd2 = partial_dependence(best, Xp, ["longitude", "latitude"], grid_resolution=30, percentiles=(0.0, 1.0), kind="average")
gx, gy = pd2["grid_values"]; Z2 = np.array(pd2["average"][0], float)                      # shape (len(gx), len(gy))
GX, GY = np.meshgrid(gx, gy, indexing="ij")
dist_deg = NearestNeighbors(n_neighbors=1).fit(X_train[["longitude", "latitude"]].values).kneighbors(np.c_[GX.ravel(), GY.ravel()])[0].reshape(GX.shape)[:, :]
Z2 = np.where(dist_deg > 0.35, np.nan, Z2)                                                  # mask ocean / places with no districts (extrapolation)
im = ax[2].pcolormesh(gx, gy, Z2.T / 1000, cmap="viridis", shading="auto"); plt.colorbar(im, ax=ax[2], label="predicted value (thousand USD)")
ax[2].set_xlabel("longitude"); ax[2].set_ylabel("latitude")
ax[0].set_title("(a) Median income"); ax[1].set_title("(b) Housing median age"); ax[2].set_title("(c) Location (longitude × latitude)")
ax[0].set_ylabel("predicted value (USD)"); fig.tight_layout(); savefig(fig, "fig20_partial_dependence")
''')
code(r'''
print(f"Interpretation: bias (mean residual) = ${res.mean():,.0f}; residual skew = {stats.skew(res):.2f}; |residual| vs prediction Spearman = {sp_h[0]:.2f} (errors grow with price level).")
print(f"Capped rows ({R['o6']['censoring']['pct_test_capped']:.1f}% of test) produce {R['o6']['censoring']['rmse_share_from_capped_rows_pct']:.0f}% of the squared error; mean prediction for them is ${cap_rows.pred.mean():,.0f} (true 500,001).")
print(f"RMSE all rows ${m_all['RMSE']:,.0f} vs uncapped rows ${m_unc['RMSE']:,.0f}; retrained without capped rows -> ${m_u_on_unc['RMSE']:,.0f} on uncapped rows.")
print(f"Residual Moran's I = {I_r:.3f} (p={pr_:.3f}). Top-3 permutation features carry {R['o6']['top3_share_pct']:.0f}% of total importance: {list(pi.index[:3])}.")
''')

# ───────────────────────────── SAVE ARTIFACTS ─────────────────────────────
md("## 4. Save the best model, predictions and a model card")
code(r'''
joblib.dump(best, MODEL_DIR / "best_model.joblib", compress=3)
pd.DataFrame({"actual": y_test.values, "predicted": pred, "residual": res}).to_csv(RESULT_DIR / "test_predictions.csv", index=False)
card = {"model": BEST_NAME, "hyperparameters": best_params[BEST_NAME], "imputer": BEST_IMPUTE, "feature_set": BEST_FS,
        "trained_on_rows": int(len(X_train)), "test_metrics": regression_metrics(y_test, pred), "target": TARGET,
        "input_columns": list(X_train.columns), "environment": R["environment"],
        "load_example": "import joblib, pandas as pd; m = joblib.load('models/best_model.joblib'); m.predict(pd.read_csv('data/processed/test.csv').drop(columns='median_house_value').head())",
        "note": "Pipeline pickles classes from src/ -> run from the project root so `src` is importable."}
json.dump(card, open(MODEL_DIR / "model_card.json", "w"), indent=2, default=float)
# reload test: the saved artefact must reproduce the in-memory predictions exactly
re_ = joblib.load(MODEL_DIR / "best_model.joblib"); assert np.allclose(re_.predict(X_test), pred); print("saved model reproduces test predictions: OK")
R["final"] = card; dump(); print(json.dumps(card["test_metrics"], indent=2))
''')
md(r'''
## 5. Conclusions
Conclusions, limitations and future scope are discussed in detail in the project report (`report/`). All numbers quoted there are generated by this notebook and stored in `results/metrics.json`.
''')

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
from pathlib import Path
out = str(Path(__file__).resolve().parents[1] / "notebooks" / "California_Housing_Predictive_Analysis.ipynb")
nbf.write(nb, out); print("wrote", out, "cells:", len(cells))
