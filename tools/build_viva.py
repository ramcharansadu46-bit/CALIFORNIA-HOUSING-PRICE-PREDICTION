"""Builds docs/VIVA_PREPARATION_GUIDE.docx (likely viva questions with model answers, formula sheet, code walkthrough)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from docx_helpers import Report
import rc
from rc import usd, num, pv, IMP

R = rc.load(); o1, o2, o3, o4, o5, o6, q, e = (R[k] for k in ("o1", "o2", "o3", "o4", "o5", "o6", "quality", "eda"))
best = o5["best_model"]; te = o5["test_table"]; cv = o5["cv_table"]; run = o5["runner_up"]; bt = te[best]
rp = Report(); rp.doc.styles["Heading 1"].paragraph_format.page_break_before = False
rp.p("VIVA PREPARATION GUIDE", align="center", bold=True, size=16, space_after=2)
rp.p("Predictive Analysis CA2 - California Housing Price Prediction", align="center", size=12, space_after=12)
rp.p("How to use this guide: read the answers *aloud* and re-phrase them in your own words. The numbers come from the executed notebook (results/metrics.json). A professor will care more that you understand **why** each step was taken than about the exact digits - but know the headline numbers.")
rp.h(2, "Headline numbers to remember")
rp.table(["Item", "Value"], [
    ["Dataset", f"California Housing (1990 census), {num(q['n_rows'])} districts, 9 predictors + target"],
    ["Split", f"{num(R['split']['n_train'])} train / {num(R['split']['n_test'])} test, stratified on income, seed 42"],
    ["Best model", f"{best}: test RMSE {usd(bt['RMSE'])}, MAE {usd(bt['MAE'])}, R² {bt['R2']:.2f}"],
    ["Baseline (mean)", f"RMSE {usd(te['Dummy (mean)']['RMSE'])}"],
    ["Linear baseline (Ridge)", f"RMSE {usd(te['Ridge']['RMSE'])}, R² {te['Ridge']['R2']:.2f}"],
    ["Runner-up", f"{run}: RMSE {usd(te[run]['RMSE'])}"],
    ["Income vs value", f"Pearson r = {o2['pearson']:.2f}; +$10,000 income ↔ +{usd(o2['ols_slope'])}; elasticity {o2['loglog_elasticity']:.2f}"],
    ["Geography", f"Kruskal-Wallis ε² = {o3['epsilon2']:.2f}; Moran's I = {o3['moran_I']['I']:.2f}"],
    ["Censoring", f"{q['target_capped_pct']:.1f} % of targets capped at 500,001; capped test rows cause {o6['censoring']['rmse_share_from_capped_rows_pct']:.0f} % of squared error"],
], "Headline numbers", widths=[4.2, 10.4], size=10, align=["left", "left"])

QA = []
def qa(q_, a_): QA.append((q_, a_))

# ───────── A. problem & data
A = [
 ("What problem does your project solve and why is it regression?",
  "It predicts the median house value (a continuous dollar amount) of a California census district from its location, housing age, room/bedroom/population/household totals, median income and ocean-proximity class. Because the target is a continuous number, this is a supervised regression problem, evaluated with RMSE, MAE and R²."),
 ("Why did you choose this dataset? How do you know it is genuine?",
  "I downloaded and profiled five candidates and chose California Housing because it has verifiable provenance (1990 US Census, StatLib, Pace and Barry 1997; also on Kaggle and OpenML), a useful size, real data-quality problems (missing values and censoring) and strong spatial structure that supports several objectives. I could not open Kaggle from my build environment, so I cross-checked size and origin against the Kaggle description as republished on a mirror site, OpenML metadata and Keras documentation and stored the file's SHA-256 checksum so that anyone can verify the file against the Kaggle download. I downloaded the file from a public GitHub mirror because Kaggle was not reachable from my build environment - you can compare checksums with the Kaggle file."),
 ("What is a census block group and what is the 'ecological fallacy'?",
  "A block group is the smallest area for which the census publishes sample data (roughly 600-3,000 people). Each row is therefore an *aggregate* of many households. The ecological fallacy is assuming that a relationship found between aggregates (for example income and price across districts) holds for individuals (one house)."),
 ("What is censoring in your data? Why is it a problem?",
  f"Censoring means values above a threshold were recorded as the threshold. Here the target is capped at 500,001 ({q['target_capped_n']} rows, {q['target_capped_pct']:.1f} %), housing age at 52 ({q['age_capped_n']} rows) and income at 15.0001 ({q['income_capped_n']} rows). The true value of a capped district is *at least* the cap, so a model trained on it systematically under-predicts the most expensive districts. I did not delete these rows (that would bias the sample towards cheaper areas); I quantified the effect in Analysis 6."),
 ("How many missing values were there and how did you handle them?",
  f"Only total_bedrooms had missing values: {q['missing']['total_bedrooms']} rows ({q['missing_pct_total_bedrooms']:.1f} %). I tested whether the missingness related to other variables (Mann-Whitney and chi-square: no significant relation), compared five imputation strategies in a masked experiment and chose {IMP[o1['best_imputer']]} imputation, which recovers hidden values with an RMSE of {o1['masked_experiment'][o1['best_imputer']]['RMSE']:.0f} rooms versus {o1['masked_experiment']['median']['RMSE']:.0f} for the median. The imputer sits inside the pipeline so it is fitted on training data only."),
]
# ───────── B. leakage & preprocessing
B = [
 ("What is data leakage and how did you prevent it?",
  "Leakage is when information from the test data (or the target) influences training, giving over-optimistic scores. I prevented it by (1) splitting into train and test *before* any exploration, (2) fitting all imputers, scalers and encoders inside scikit-learn Pipelines so they are re-fitted on each training fold, (3) never using the target to create features (the spatial features use fixed city coordinates), and (4) using the test set only once, after the model was chosen by cross-validation."),
 ("Why did you stratify the split on income?",
  "Median income is the strongest predictor, so I want the same income distribution in the training and test sets. Stratifying on five income bands guarantees that, so the test score is not distorted by an unlucky random split."),
 ("Why log-transform and standardise features? Which models need it?",
  "The count variables are extremely right-skewed (skewness about 4-5 for total_rooms and population, against about 1 for the target). log(1+x) reduces skewness. Standardisation puts features on the same scale, which matters for distance-based models (k-NN) and penalised linear models (Ridge). Tree-based models are invariant to monotone transformations, so it neither helps nor harms them; I used one preprocessing pipeline so that all models are compared fairly."),
 ("Why didn't you remove outliers?",
  "The outliers (for example 100+ rooms per household) are real districts such as resort areas, not recording errors. Removing them would hide real cases the model must handle. I used log transforms and tree models, which are robust to extreme values."),
 ("What is one-hot encoding and why for ocean_proximity?",
  "ocean_proximity is a nominal category with five classes and no natural order, so numbering them 1-5 would impose a false order. One-hot encoding creates one 0/1 column per class. handle_unknown='ignore' makes the pipeline safe if a new class appears at prediction time."),
]
# ───────── C. statistics
C = [
 ("Why did you use Spearman correlation as well as Pearson?",
  "Pearson measures *linear* association and is sensitive to outliers and to the censored band. Spearman works on ranks, so it captures any monotone relationship and is robust. They agree here (Pearson %.2f, Spearman %.2f), which shows the relationship is strong and roughly monotone." % (o2["pearson"], o2["spearman"])),
 ("What is a bootstrap confidence interval?",
  "Resample the data with replacement many times (1,000 here), recompute the statistic each time, and take the 2.5th and 97.5th percentiles. It estimates the sampling uncertainty without assuming a distribution. For the income correlation the 95 %% interval is [%.3f, %.3f]." % tuple(o2["pearson_ci"])),
 ("Why Kruskal-Wallis instead of ANOVA? What is ε² and why Holm?",
  f"ANOVA assumes roughly normal residuals and equal variances; house values are skewed, censored and have unequal spread, so I used the rank-based Kruskal-Wallis test (H = {o3['kruskal_H']:,.0f}, p < 0.001). ε² = (H − k + 1)/(N − k) = {o3['epsilon2']:.2f} is the effect size - it says the difference is large, not just significant. With 5 groups there are 10 pairwise tests, so the chance of at least one false positive rises; Holm's step-down correction controls that family-wise error rate and is more powerful than Bonferroni."),
 ("What is Moran's I and what does your value mean?",
  f"Moran's I measures spatial autocorrelation: whether nearby places have similar values. It ranges roughly from −1 to +1; 0 means spatial randomness. I got {o3['moran_I']['I']:.2f} for house values (8 nearest neighbours), meaning prices are strongly clustered: a district's price is largely predictable from its neighbours. The permutation test (shuffle values over locations) gives p ≈ 0.002, the minimum possible with 499 permutations. After modelling, the residuals' Moran's I fell to {o6['moran_residuals']['I']:.2f}: the model captured most of the spatial structure."),
 ("Why does spatial autocorrelation matter for your validation?",
  "In random cross-validation, neighbouring districts end up in both training and test folds, so the model can 'look up' a similar neighbour; the score is therefore optimistic for predicting in a completely new region. A spatially blocked validation (holding out whole areas) would give a more honest estimate for new regions. I list it as a limitation and as future work."),
 ("What is the Breusch-Pagan test and what did it show?",
  f"It tests whether the residual variance is constant (homoscedasticity). It regresses squared residuals on the regressors; LM = n·R² follows a χ² distribution under constant variance. For the income model p < 0.001, so the variance grows with income. Consequence: classical OLS standard errors are unreliable, which is why I used bootstrap intervals."),
 ("What is multicollinearity and what is VIF?",
  "Multicollinearity means predictors are strongly inter-correlated (rooms, bedrooms, population and households all scale with district size). VIF = 1/(1 − R²_j), where R²_j comes from regressing feature j on the others; above ~10 is serious. It makes linear coefficients unstable but does not hurt tree ensembles' predictions."),
 ("Why did you use the Nadeau-Bengio corrected t-test instead of a normal paired t-test?",
  "Cross-validation training sets overlap, so the fold scores are not independent and the ordinary paired t-test underestimates the variance and finds 'significant' differences too easily. Nadeau and Bengio (2003) inflate the variance by (1/k + n_test/n_train) to correct for this. I also confirmed the main conclusion with a paired bootstrap on the test set."),
]
# ───────── D. models
D = [
 ("Explain Ridge regression and why you included it.",
  f"Ridge minimises squared error plus α·‖w‖², shrinking coefficients to reduce variance when predictors are correlated. It is the interpretable linear baseline. It reached test RMSE {usd(te['Ridge']['RMSE'])} (R² {te['Ridge']['R2']:.2f}); the big gap to tree models shows the problem is strongly non-linear."),
 ("Why did you also fit a log-target Ridge, and what did you learn?",
  f"Many analysts model log(price). I tested it: its RMSE was higher ({usd(te['Ridge (log-target)']['RMSE'])} vs {usd(te['Ridge']['RMSE'])}) even though its median percentage error was lower ({te['Ridge (log-target)']['MedAPE_%']:.1f} % vs {te['Ridge']['MedAPE_%']:.1f} %). The log model treats percentage errors equally, which suits typical districts, but RMSE is dominated by the expensive ones, which back-transformation under-predicts. Lesson: choose the target transformation according to the metric that matters. I originally had only the log version; when I noticed it understated the linear model I added the raw-target version as the main baseline."),
 ("How does a Random Forest work and why does it reduce variance?",
  "It trains many deep decision trees, each on a bootstrap sample and with a random subset of features at each split, then averages them. Individual deep trees have low bias but high variance; averaging de-correlated trees keeps the low bias and reduces the variance."),
 ("What is gradient boosting and how is XGBoost different?",
  f"Boosting builds trees *sequentially*: each new small tree is fitted to the residual errors (the negative gradient of the loss) of the current ensemble, and added with a learning rate ν. XGBoost adds explicit regularisation (γ per leaf, λ on leaf weights), second-order gradient information, subsampling and efficient histogram-based split finding. It was the best model here (test RMSE {usd(bt['RMSE'])})."),
 ("Why is XGBoost better than the runner-up if the difference is small?",
  f"The difference to {run} is {abs(o5['significance'][run]['test_RMSE_diff']):,.0f} dollars of RMSE (95 % bootstrap CI {o5['significance'][run]['boot_ci_lo']:,.0f} to {o5['significance'][run]['boot_ci_hi']:,.0f}, excluding zero; corrected-t p = {pv(o5['significance'][run]['NB_p'])}), so it is statistically reliable but practically small (~{abs(o5['significance'][run]['test_RMSE_diff'])/bt['RMSE']*100:.0f} %). If training speed or simplicity mattered, {run} would be defensible - I say that in the report."),
 ("Why does k-NN need scaled features?",
  "k-NN uses distances; an unscaled variable with a large range (like total_rooms) would dominate the distance. Standardising puts all features on comparable scales."),
 ("How did you tune hyper-parameters without overfitting to the test set?",
  "Tuning used only the training set with 3-fold cross-validation (grid search for small spaces, randomised search for large ones). Model selection used 5-fold CV RMSE. The test set was touched once, at the end, to report an unbiased estimate. A small optimism remains because the same training data were used for tuning and for the selection CV; the test set quantifies it (test RMSE %s vs CV RMSE %s for %s)." % (usd(bt["RMSE"]), usd(cv[best]["CV_RMSE"]), best)),
 ("What does your learning curve tell you? (bias-variance)",
  f"Training RMSE is very low ({usd(o5['learning_curve'][-1]['train_RMSE'])}) while validation RMSE is {usd(o5['learning_curve'][-1]['val_RMSE'])}: a large gap, typical for flexible boosted trees (high variance). Validation error is still decreasing as data grows, so the model is limited by information, not capacity; more data or better features would help more than more tuning."),
]
# ───────── E. metrics & results
E = [
 ("Explain RMSE, MAE and R². Why is RMSE larger than MAE?",
  f"RMSE = √mean(error²) penalises large errors; MAE = mean(|error|) treats all errors equally; R² = 1 − SS_res/SS_tot is the share of variance explained (0 = as good as predicting the mean). RMSE ≥ MAE always, and a big gap means heavy-tailed errors: here RMSE {usd(bt['RMSE'])} vs MAE {usd(bt['MAE'])}, because of a few very large errors, mostly capped districts."),
 ("Is an RMSE of ~$41,000 good?",
  f"The mean house value is about {usd(e['target_mean'])} and the standard deviation about {usd(e['target_std'])}; the mean-prediction baseline has RMSE {usd(te['Dummy (mean)']['RMSE'])}. My model cuts the error by {(1-bt['RMSE']/te['Dummy (mean)']['RMSE'])*100:.0f} %, explains {bt['R2']*100:.0f} % of the variance, and {bt['Within20%']:.0f} % of predictions are within ±20 % of the truth. For aggregate 1990 census data without neighbourhood amenities, that is good; a perfect score is impossible because of censoring and unmeasured local factors."),
 ("What did your error analysis find?",
  f"Errors are unbiased on average (mean residual {usd(o6['bias_mean_residual'])}) but heavy-tailed and growing with price. Relative errors are similar across classes (about 9-11 %). Capped districts are {o6['censoring']['pct_test_capped']:.1f} % of the test set but cause {o6['censoring']['rmse_share_from_capped_rows_pct']:.0f} % of squared error because the model predicts them {usd(500001-o6['censoring']['mean_pred_for_capped'])} too low on average (and the true shortfall is bigger). Residuals keep a small spatial pattern (Moran's I = {o6['moran_residuals']['I']:.2f})."),
 ("What are permutation importance, SHAP and partial dependence? When do they mislead?",
  "Permutation importance: shuffle one feature and see how much the test error rises - model-agnostic but misleading when features are correlated or derived from each other (shuffling total_rooms also breaks rooms_per_household). SHAP: each feature's additive contribution to each individual prediction, based on Shapley values from game theory - local and global but computed on the model, not on causal effects. Partial dependence: average prediction as one feature is varied - can be misleading if features are strongly correlated (it evaluates unrealistic combinations)."),
 ("Which features matter most?",
  "Location (longitude, latitude and the distance features) matters most, then median income, then household density (people per household) and the ocean-proximity class. Grouped permutation shows that shuffling longitude and latitude together hurts more than shuffling all four size counts together: where a district is matters more than how big it is."),
 ("Did ratio and spatial features really help?",
  f"Yes, modestly but consistently. Ratio features lowered CV-RMSE by {abs(o4['ratio_contrasts'][1]['pct_change']):.1f} % (gradient boosting) and {abs(o4['ratio_contrasts'][0]['pct_change']):.1f} % (Ridge) on the raw set; spatial features by {abs(o3['spatial_contrasts'][1]['pct_change']):.1f} % and {abs(o3['spatial_contrasts'][0]['pct_change']):.1f} %. All eight paired comparisons were significant (corrected t-test, p < 0.05) and the direction was the same for both a linear and a tree model."),
]
# ───────── F. critical
F = [
 ("Can you say that higher income *causes* higher house prices?",
  "No. The analysis is observational and at district level: it shows a strong association (r = %.2f), not causation. Income and prices share common drivers (location, amenities), and reverse effects are possible (expensive areas attract higher incomes)." % o2["pearson"]),
 ("Can your model be used to value houses today?",
  "No. The data are from 1990 and describe districts, not individual houses. The project demonstrates a methodology; for current valuation I would retrain on recent data with house-level features."),
 ("What are the main limitations of your work?",
  "(1) 1990 data; (2) censoring of target, age and income; (3) district-level aggregation; (4) random CV is optimistic for new regions because of spatial autocorrelation; (5) the ISLAND class has only five districts; (6) the hyper-parameter search budget was small (6-12 candidates, single CPU); (7) some choices (hub cities, imputer candidates) were mine."),
 ("What would you do next?",
  "Spatially blocked cross-validation; a censoring-aware model (Tobit or two-stage); better location features (coast distance, amenities); prediction intervals via quantile regression or conformal prediction; deployment as an API with drift monitoring."),
 ("What did not go as planned in the project?",
  "Two things worth admitting: the iterative imputer once produced a negative bedroom count, so imputed values are clipped; and my first linear baseline used a log-target that handicapped it in dollar RMSE, so I added a raw-target Ridge and kept the log version as a documented comparison. Both are in the report."),
 ("Walk me through the code structure.",
  "src/features.py holds the custom scikit-learn transformers: BedroomImputer (five strategies), FeatureEngineer (ratio and spatial features) and build_preprocessor, which assembles imputer → feature engineer → ColumnTransformer. src/evaluate.py has the metrics, the Nadeau-Bengio test, the paired bootstrap and Moran's I. The notebook is organised by objective and writes every number to results/metrics.json; the saved model models/best_model.joblib is the whole pipeline, so predict() takes raw columns."),
 ("How can the results be reproduced?",
  "Install requirements.txt, open the notebook and run all cells (fixed seed 42); the notebook regenerates all figures and metrics.json. The README lists the commands. The raw file's SHA-256 checksum is stored so the exact data can be verified."),
]
for title, items in (("A. Problem and data", A), ("B. Leakage and preprocessing", B), ("C. Statistical methods", C), ("D. Models", D), ("E. Metrics, results and explainability", E), ("F. Critical questions and viva traps", F)):
    rp.h(2, title)
    for i, (q_, a_) in enumerate(items, 1):
        rp.p(f"**Q. {q_}**", align="left", space_after=2, keep_next=True)
        rp.p(f"**A.** {a_}", space_after=8)

rp.h(2, "Formula sheet")
for t in [
    "RMSE = √( (1/n) Σ (y_{i} − ŷ_{i})^{2} )        MAE = (1/n) Σ |y_{i} − ŷ_{i}|        R^{2} = 1 − SS_{res}/SS_{tot}",
    "Standardisation: z = (x − μ)/σ        Log transform: x' = ln(1 + x)",
    "Ridge: min_{w} Σ (y_{i} − w·x_{i})^{2} + α‖w‖^{2}",
    "Random forest: ŷ = (1/B) Σ T_{b}(x)        Boosting: F_{m} = F_{m−1} + ν h_{m}",
    "XGBoost objective: Σ l(y_{i}, ŷ_{i}) + Σ_{k} Ω(f_{k}),  Ω(f) = γT + ½λΣ w_{j}^{2}",
    "Kruskal-Wallis: H = [12/(N(N+1))] Σ n_{g} R̄_{g}^{2} − 3(N+1);  ε^{2} = (H − k + 1)/(N − k)",
    "Moran's I = (n/S_{0}) Σ_{i}Σ_{j} w_{ij} z_{i} z_{j} / Σ z_{i}^{2}",
    "Corrected resampled t: t = d̄ / √[(1/k + n_{test}/n_{train}) s_{d}^{2}]",
    "Haversine: d = 2R arcsin √[sin^{2}(Δφ/2) + cos φ_{1} cos φ_{2} sin^{2}(Δλ/2)]",
    "VIF_{j} = 1/(1 − R^{2}_{j})        Permutation importance: ΔRMSE_{j} = RMSE(shuffled j) − RMSE(base)",
]:
    rp.equation(t, number=False)
out = rc.ROOT / "docs" / "VIVA_PREPARATION_GUIDE.docx"; rp.save(out); print("saved", out)
