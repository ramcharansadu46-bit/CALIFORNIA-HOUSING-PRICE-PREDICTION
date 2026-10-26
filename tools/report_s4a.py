"""Section 4.1-4.4 (Objectives 1-4)."""
from rc import FIG, usd, num, pct, pv, pvs, IMP

NAME = {"median": "Median", "group_median": "Group median (by ocean proximity)", "ratio": "Ratio (households × median ratio)", "knn": "k-NN (k = 5)", "iterative": "Iterative (BayesianRidge)"}


def sub(rp, n, title):
    rp.h(2, f"4.{n} Analysis {n}: {title}")


def o1(rp, R):
    o, q = R["o1"], R["quality"]
    sub(rp, 1, "Data Quality, Censoring and Imputation of *total_bedrooms* (Objective 1)")
    rp.h(3, "4.1.1 Introduction")
    rp.p("Real data are rarely complete or error-free. This analysis asks whether the 207 missing values of *total_bedrooms* are ignorable, which imputation strategy recovers the true values best, and how much of the data is censored by the data publisher. The answers determine how the data are prepared for every later analysis and model. Missing-data theory (Little and Rubin, 2019) distinguishes data that are missing completely at random (MCAR), at random given observed variables (MAR) and not at random (MNAR); imputation by borrowing information from other variables is justified when the mechanism does not depend on the missing value itself.")
    rp.h(3, "4.1.2 General Description")
    rp.p("The analysis has four parts. First, the 'missing' and 'observed' groups are compared on every other variable (Mann-Whitney U test for numeric variables, χ² test for ocean proximity) to look for evidence that missingness is related to other characteristics. Second, five imputation strategies are compared in a *masked-imputation experiment*: 5 % of the observed bedroom counts of the training set are hidden at random, imputed, and compared with their true values; the experiment is repeated with five different random masks. Third, the strategies are compared by their effect on the cross-validated RMSE of a gradient-boosting model. Fourth, the extent of censoring in the target, age and income is documented (Table 5 in Section 3).")
    rp.h(3, "4.1.3 Specific Requirements, Functions and Formulas")
    rp.table(["Function / class", "Library", "Purpose"], [
        ["mannwhitneyu, chi2_contingency", "scipy.stats", "Compare missing vs. observed groups"],
        ["BedroomImputer (custom)", "src/features.py", "Median, group-median, ratio, k-NN and iterative imputation, fitted on training data only"],
        ["KNNImputer, IterativeImputer", "scikit-learn", "Neighbour-based and regression-based imputation engines"],
        ["cross_val_score, RepeatedKFold", "scikit-learn", "Paired downstream evaluation (5 folds × 2 repeats)"],
        ["nadeau_bengio_ttest", "src/evaluate.py", "Corrected resampled t-test for CV score differences"],
    ], "Functions used in Analysis 1", widths=[4.8, 3.2, 6.6], align=["left", "left", "left"], size=10)
    rp.p("The following formulas are used. The Mann–Whitney statistic for samples of sizes *n*_{1} and *n*_{2} with rank-sum *R*_{1} is")
    rp.equation("U = n_{1}n_{2} + n_{1}(n_{1}+1)/2 − R_{1}        r_{rb} = 2U /(n_{1}n_{2}) − 1")
    rp.p("where *r*_{rb} is the rank-biserial correlation (effect size, between −1 and 1). The χ² statistic of independence is")
    rp.equation("χ^{2} = Σ ( O_{ij} − E_{ij} )^{2} / E_{ij}")
    rp.p("Imputation accuracy in the masked experiment is measured by")
    rp.equation("RMSE = √( (1/m) Σ ( x̂_{i} − x_{i} )^{2} ),        MAE = (1/m) Σ | x̂_{i} − x_{i} |")
    rp.p("The *ratio* imputer sets total_bedrooms = households × median(total_bedrooms / households), the k-NN imputer averages the value of the five nearest districts in the space of the eight standardised numeric variables, and the iterative imputer models total_bedrooms as a Bayesian ridge regression of the other variables. Because CV training sets overlap, differences between strategies are tested with the Nadeau–Bengio (2003) corrected resampled *t*-statistic")
    rp.equation("t = d̄ / √[ ( 1/k + n_{test}/n_{train} ) s_{d}^{2} ]")
    rp.p("where *d̄* and *s*_{d}^{2} are the mean and variance of the per-fold score differences and *k* is the number of folds.")
    rp.h(3, "4.1.4 Analysis Results")
    rp.p(f"**Is the missingness related to other variables?** In the training set {o['n_missing_train']} districts have missing total_bedrooms. Table {rp.tab_n+1} compares the missing and observed groups.")
    rows = [[t["variable"], num(t["median_missing_group"], 1), num(t["median_observed_group"], 1), pv(t["p_value"]), f"{t['rank_biserial_r']:+.3f}"] for t in o["missing_tests"]]
    rp.table(["Variable", "Median (missing group)", "Median (observed group)", "Mann–Whitney p", "Rank-biserial r"], rows, "Comparison of districts with missing and observed total_bedrooms (training set)", widths=[4.2, 3.2, 3.2, 2.2, 1.8], size=10, align=["left", "center", "center", "center", "center"])
    mb = o["missing_by_ocean"]
    rp.p(f"None of the six variables differs significantly between the groups (smallest p = {min(t['p_value'] for t in o['missing_tests']):.2f}), and all effect sizes are small (|r_{{rb}}| ≤ {max(abs(t['rank_biserial_r']) for t in o['missing_tests']):.2f}). The distribution of the missing values over ocean-proximity classes (<1H OCEAN {mb['<1H OCEAN']}, INLAND {mb['INLAND']}, NEAR OCEAN {mb['NEAR OCEAN']}, NEAR BAY {mb['NEAR BAY']}, ISLAND {mb['ISLAND']}) does not differ from the overall class mix (χ² = {o['chi2']:.2f}, p = {o['chi2_p']:.2f}).")
    me = o["masked_experiment"]; best = o["best_imputer"]
    rp.p(f"**Which imputer recovers the hidden values best?** Table {rp.tab_n+1} reports the masked-imputation experiment (mean over {o['n_trials']} random masks).")
    rows = [[NAME[s], f"{me[s]['RMSE']:.1f} ± {me[s]['RMSE_sd']:.1f}", f"{me[s]['MAE']:.1f}", f"{me[s]['MedAPE_%']:.1f} %"] for s in sorted(me, key=lambda s: me[s]["RMSE"])]
    rp.table(["Imputation strategy", "RMSE (rooms)", "MAE (rooms)", "Median abs. % error"], rows, "Masked-imputation experiment (5 % of observed values hidden)", widths=[6.4, 3.0, 2.6, 2.6], size=10, align=["left", "center", "center", "center"], highlight_row=0)
    dn = o["downstream"]
    rp.p(f"**Does the choice matter for prediction?** Table {rp.tab_n+1} shows the cross-validated RMSE of a gradient-boosting model under each strategy and the paired difference from median imputation.")
    rows = [[NAME[s], usd(dn[s]["mean_CV_RMSE"]), "–" if s == "median" else f"{dn[s]['mean_diff']:+,.0f}", "–" if s == "median" else f"[{dn[s]['ci_lo']:,.0f}, {dn[s]['ci_hi']:,.0f}]", "–" if s == "median" else pv(dn[s]["NB_p"])] for s in dn]
    rp.table(["Strategy", "Mean CV-RMSE", "Δ vs. median (USD)", "95 % CI", "Nadeau–Bengio p"], rows, "Downstream effect of the imputation strategy (5×2 repeated CV, gradient boosting)", widths=[5.4, 2.4, 2.6, 2.6, 1.9], size=10, align=["left", "center", "center", "center", "center"])
    rp.h(3, "4.1.5 Visualization and Interpretation")
    rp.figure(FIG / "fig05_imputation.png", "Imputation strategies: (a) error on masked values, (b) downstream effect with 95 % confidence intervals", 15.5)
    ratio_red = (1 - me[best]["RMSE"] / me["median"]["RMSE"]) * 100
    rp.p(f"**Interpretation.** The missingness shows no detectable relation with any observed variable, which is consistent with data missing completely at random; this cannot be proven from the data, but it supports treating the 207 rows by imputation rather than deletion. Strategies that use information from the other variables recover the hidden counts far better than a single constant: the {NAME[best].split(' (')[0]} imputer reaches an RMSE of {me[best]['RMSE']:.1f} rooms against {me['median']['RMSE']:.1f} for median imputation ({ratio_red:.0f} % lower). This is expected because the number of bedrooms is almost proportional to the number of households and rooms of the district, whereas the group median adds nothing (ocean proximity barely informs bedroom counts). In contrast, Figure {rp.fig_n}(b) shows that the downstream effect on house-value prediction is negligible: all differences are below {max(abs(dn[s]['mean_diff']) for s in dn):,.0f} dollars (≈ {max(abs(dn[s]['mean_diff']) for s in dn)/dn['median']['mean_CV_RMSE']*100:.2f} % of the RMSE) and none is significant. The reason is simply that only {q['missing_pct_total_bedrooms']:.1f} % of the rows are affected.")
    rp.p(f"**Decision.** The {NAME[best]} imputer was selected for all subsequent models because it is the most accurate at no cost in prediction quality. One practical observation from development: the unconstrained iterative imputer produced an impossible negative bedroom count for one district, so all model-based imputed values are clipped to the smallest observed count. **Censoring:** {q['target_capped_n']} districts ({q['target_capped_pct']:.1f} %) have the target capped at $500,001, {q['age_capped_n']} ({q['age_capped_pct']:.1f} %) have housing age capped at 52 years and {q['income_capped_n']} have income capped at 15.0001. These values are retained, and their consequences are examined in Analyses 2 and 6.")


def o2(rp, R):
    o = R["o2"]
    sub(rp, 2, "Income as a Determinant of House Value (Objective 2)")
    rp.h(3, "4.2.1 Introduction")
    rp.p("Median household income is the single most important economic variable in housing-affordability analysis. This analysis quantifies the strength, the steepness and the functional form of the relationship between a district’s median income and its median house value, and tests whether classical regression assumptions (linearity and constant variance) hold. Knowing how much house value changes per unit of income, and whether the effect saturates at high incomes, informs both feature design and policy interpretation.")
    rp.h(3, "4.2.2 General Description")
    rp.p("On the training set the analysis computes three correlation coefficients with bootstrap confidence intervals, fits three regression specifications (linear, quadratic, log–log), tests for non-linearity with an *F*-test, tests for heteroscedasticity with the Breusch–Pagan test, summarises value by income decile, and repeats the linear fit after excluding the censored rows to assess the sensitivity of the results to the $500,001 cap.")
    rp.h(3, "4.2.3 Specific Requirements, Functions and Formulas")
    rp.table(["Function", "Library", "Purpose"], [
        ["pearsonr, spearmanr, kendalltau", "scipy.stats", "Linear and rank correlation"],
        ["OLS (smf.ols)", "statsmodels", "Linear, quadratic and log–log regression with confidence intervals"],
        ["anova_lm", "statsmodels", "F-test comparing nested linear and quadratic models"],
        ["het_breuschpagan", "statsmodels", "Test for non-constant residual variance"],
        ["lowess", "statsmodels", "Locally weighted smoother (Cleveland, 1979) as a model-free trend"],
        ["bootstrap (1,000 resamples)", "NumPy", "Percentile confidence intervals for correlations (Efron, 1979)"],
    ], "Functions used in Analysis 2", widths=[4.6, 3.0, 7.0], align=["left", "left", "left"], size=10)
    rp.equation("r = Σ(x_{i} − x̄)(y_{i} − ȳ) / √[ Σ(x_{i} − x̄)^{2} Σ(y_{i} − ȳ)^{2} ]        ρ_{s} = r( rank(x), rank(y) )")
    rp.equation("Linear: y = β_{0} + β_{1}x + ε        Quadratic: y = β_{0} + β_{1}x + β_{2}x^{2} + ε        Log–log: ln y = α + η ln x + ε")
    rp.p("In the log–log model the slope η is the *elasticity*: a 1 % change in income is associated with an η % change in house value. The *F*-test for the added quadratic term is")
    rp.equation("F = [ (RSS_{lin} − RSS_{quad}) / 1 ] / [ RSS_{quad} / (n − 3) ]")
    rp.p("The Breusch–Pagan statistic is LM = *n*R^{2}_{aux}, where R^{2}_{aux} comes from regressing the squared residuals on the regressors; under constant variance LM follows a χ² distribution (Breusch and Pagan, 1979).")
    rp.h(3, "4.2.4 Analysis Results")
    rp.table(["Measure", "Estimate", "95 % bootstrap CI"], [
        ["Pearson r", f"{o['pearson']:.3f}", f"[{o['pearson_ci'][0]:.3f}, {o['pearson_ci'][1]:.3f}]"],
        ["Spearman ρ", f"{o['spearman']:.3f}", f"[{o['spearman_ci'][0]:.3f}, {o['spearman_ci'][1]:.3f}]"],
        ["Kendall τ", f"{o['kendall']:.3f}", "–"]], "Correlation between median income and median house value (training set)", widths=[5.0, 4.0, 5.0], size=10.5)
    rp.table(["Model", "Key parameter", "R²", "Note"], [
        ["Linear OLS", f"slope = {usd(o['ols_slope'])} per $10,000 (95 % CI {usd(o['ols_slope_ci'][0])}–{usd(o['ols_slope_ci'][1])}); intercept {usd(o['ols_intercept'])}", f"{o['ols_r2']:.4f}", f"Breusch–Pagan LM = {o['bp_linear_LM']:.1f}, {pvs(o['bp_linear_p'])}"],
        ["Quadratic OLS", f"quadratic term = {o['quad_b2']:,.0f}", f"{o['quad_r2']:.4f}", f"F = {o['quad_F']:.1f}, {pvs(o['quad_F_p'])}"],
        ["Log–log OLS", f"elasticity η = {o['loglog_elasticity']:.3f} (95 % CI {o['loglog_elasticity_ci'][0]:.3f}–{o['loglog_elasticity_ci'][1]:.3f})", f"{o['loglog_r2']:.4f}", f"Breusch–Pagan LM = {o['bp_loglog_LM']:.1f}, {pvs(o['bp_loglog_p'])}"],
        ["Linear, uncapped rows only", f"slope = {usd(o['uncensored_slope'])} (n = {num(o['n_uncensored'])})", f"{o['uncensored_r2']:.4f}", "Sensitivity to censoring"],
    ], "Regression results for house value on income", widths=[3.0, 6.2, 1.6, 3.8], size=9.5, align=["left", "left", "center", "left"])
    dec = o["deciles"]
    rows = [[k, f"{v['income_min']:.2f}–{v['income_max']:.2f}", num(v["n"]), usd(v["median_value"]), usd(v["mean_value"]), f"{v['capped_pct']:.1f}"] for k, v in dec.items()]
    rp.table(["Decile", "Income range (×$10k)", "n", "Median value", "Mean value", "% capped"], rows, "House value by income decile (training set)", widths=[1.6, 3.6, 1.6, 2.9, 2.9, 2.0], size=9.5, align=["center"] * 6)
    rp.h(3, "4.2.5 Visualization and Interpretation")
    rp.figure(FIG / "fig06_income_relationship.png", "Relationship between median income and house value: (a) linear, quadratic and LOWESS fits, (b) log–log fit", 15.5)
    rp.figure(FIG / "fig07_income_deciles_residuals.png", "(a) Residuals of the linear model against fitted values; (b) house value by income decile", 15.5)
    rp.p(f"**Interpretation.** Income is strongly and positively related to value (Pearson r = {o['pearson']:.2f}, Spearman ρ = {o['spearman']:.2f}); the narrow bootstrap intervals show the estimate is precise. Each additional $10,000 of district median income is associated with about {usd(o['ols_slope'], 0)} higher median house value, yet income alone explains only {o['ols_r2']*100:.1f} % of the variance, so more than half of the variation must come from other factors — notably location (Analysis 3). The log–log elasticity of {o['loglog_elasticity']:.2f} (below one) means that values rise less than proportionally with income. The median value of the top income decile is {o['decile_ratio_top_bottom']:.1f} times that of the bottom decile (Table {rp.tab_n}).")
    rp.p(f"The quadratic term is negative ({o['quad_b2']:,.0f}), indicating diminishing returns at high incomes, and it is statistically significant (F = {o['quad_F']:.0f}), but it raises R² by only {(o['quad_r2']-o['ols_r2'])*100:.2f} percentage points, so the departure from linearity is practically small in the central range of incomes. The LOWESS curve in Figure {rp.fig_n-1}(a) flattens at the top where the $500,001 cap creates a horizontal band. Residual spread grows with the fitted value (Figure {rp.fig_n}(a)); the Breusch–Pagan test rejects constant variance for both the linear and the log–log model (the log transform does not remove it, a possible contributor being the censored band), so classical standard errors should not be trusted and the bootstrap intervals are preferred. Finally, excluding capped rows lowers the slope from {usd(o['ols_slope'])} to {usd(o['uncensored_slope'])} and R² from {o['ols_r2']:.2f} to {o['uncensored_r2']:.2f}; both the all-rows and the uncapped-rows estimates are biased in known ways (censoring compresses the top, truncation removes it), so the two should be read as a plausible range rather than a single truth.")


def o3(rp, R):
    o = R["o3"]; gt = o["group_table"]; mo = o["moran_I"]
    sub(rp, 3, "Geography: Ocean Proximity, Spatial Autocorrelation and Spatial Features (Objective 3)")
    rp.h(3, "4.3.1 Introduction")
    rp.p("Location is the central principle of real-estate valuation. This analysis tests three questions: (a) do house values differ significantly between the ocean-proximity classes and how large is the difference; (b) are values spatially autocorrelated, i.e. are neighbouring districts more alike than distant ones; and (c) do features describing a district’s distance to the state’s major urban centres improve out-of-sample prediction beyond raw longitude and latitude?")
    rp.h(3, "4.3.2 General Description")
    rp.p("For (a), the non-parametric Kruskal–Wallis test is used because the target is skewed and censored, followed by all pairwise Mann–Whitney tests with Holm’s correction for multiple comparisons. For (b), global Moran’s I is computed with weights defined by each district’s eight nearest neighbours. For (c), a paired ablation compares models with and without five hub-distance features (San Francisco, Los Angeles, San Diego, San Jose, Sacramento) and the distance to the nearest hub, using identical cross-validation folds, a regularised linear model (Ridge) and a gradient-boosting model. The ablation covers all four combinations of ratio and spatial feature sets (Table below) and is shared with Analysis 4.")
    rp.h(3, "4.3.3 Specific Requirements, Functions and Formulas")
    rp.table(["Function", "Library", "Purpose"], [
        ["kruskal, mannwhitneyu", "scipy.stats", "Omnibus and pairwise group comparisons"],
        ["multipletests (Holm)", "statsmodels", "Family-wise error control for 10 pairwise tests"],
        ["morans_i (custom)", "src/evaluate.py", "Global Moran’s I, k-NN weights, haversine metric, permutation p-value"],
        ["FeatureEngineer(spatial=True)", "src/features.py", "Haversine distances to five hubs and to the nearest hub"],
        ["cross_validate, RepeatedKFold", "scikit-learn", "Paired CV-RMSE and R² for each feature set"],
    ], "Functions used in Analysis 3", widths=[4.6, 3.2, 6.8], align=["left", "left", "left"], size=10)
    rp.equation("H = [12 / (N(N+1))] Σ_{g} n_{g} R̄_{g}^{2} − 3(N+1)        ε^{2} = ( H − k + 1 ) / ( N − k )")
    rp.p("where *R̄*_{g} is the mean rank of group *g*, *N* the total sample size and *k* the number of groups; ε² is an effect-size measure between 0 and 1 (Tomczak and Tomczak, 2014). Holm’s method orders the *m* raw p-values and rejects the *i*-th smallest if it is below α/(*m* − *i* + 1). Moran’s I (Moran, 1950) with row-standardised weights *w*_{ij} is")
    rp.equation("I = [ n / S_{0} ] · Σ_{i}Σ_{j} w_{ij} z_{i} z_{j} / Σ_{i} z_{i}^{2},        z_{i} = y_{i} − ȳ,        E[I] = −1/(n−1)")
    rp.p("Values near +1 indicate strong positive spatial clustering. Significance is assessed by randomly permuting the values over locations (499 permutations).")
    rp.h(3, "4.3.4 Analysis Results")
    rows = [[k, num(v["n"]), usd(v["median"]), f"{usd(v['q1'])} – {usd(v['q3'])}", f"{v['capped_pct']:.1f}"] for k, v in sorted(gt.items(), key=lambda kv: kv[1]["median"])]
    rp.table(["Ocean proximity", "n", "Median value", "Interquartile range", "% capped"], rows, "House value by ocean-proximity class (training set)", widths=[3.6, 1.6, 2.9, 5.0, 1.6], size=10, align=["left", "center", "center", "center", "center"])
    rp.p(f"The Kruskal–Wallis test rejects equal distributions (H = {o['kruskal_H']:,.0f}, {pvs(o['kruskal_p'])}) with an effect size of ε² = {o['epsilon2']:.2f}, which is a large effect. Table {rp.tab_n+1} gives the pairwise comparisons.")
    pw = o["pairwise"]
    rows = [[f"{p['A']} vs {p['B']}", f"{p['median_diff']:+,.0f}", f"{p['rank_biserial_r']:+.2f}", pv(p["p_holm"]), "Yes" if p["significant_5%"] else "No"] for p in pw]
    rp.table(["Pair (A vs B)", "Median difference (USD)", "Rank-biserial r", "Holm-adjusted p", "Significant (5 %)"], rows, "Pairwise comparisons of ocean-proximity classes (Mann–Whitney U, Holm correction)", widths=[5.2, 3.2, 2.2, 2.2, 1.9], size=9.5, align=["left", "center", "center", "center", "center"])
    rp.p(f"**Spatial autocorrelation.** Moran’s I of the target is {mo['I']:.3f} (expected under spatial randomness {mo['expected']:.5f}; permutation z = {mo['z']:.0f}, p = {mo['p']:.3f}, the smallest p attainable with {mo['n_perm']} permutations).")
    ab = R["ablation"]
    rows = []
    for fs in ["F0 raw", "F1 +ratios", "F2 +spatial", "F3 +ratios+spatial"]:
        rows.append([fs, usd(ab[f"{fs} | Ridge"]["mean_RMSE"]), f"{ab[f'{fs} | Ridge']['mean_R2']:.3f}", usd(ab[f"{fs} | HGB"]["mean_RMSE"]), f"{ab[f'{fs} | HGB']['mean_R2']:.3f}"])
    rp.p(f"**Feature-set ablation.** Table {rp.tab_n+1} lists the cross-validated results for the four feature sets (F0: raw variables; F1: plus ratio features; F2: plus spatial features; F3: both).")
    rp.table(["Feature set", "Ridge RMSE", "Ridge R²", "Gradient-boosting RMSE", "Gradient-boosting R²"], rows, "Ablation of feature sets (5×2 repeated CV on the training set)", widths=[4.0, 2.7, 2.2, 3.4, 2.7], size=10, align=["left"] + ["center"] * 4)
    sc = o["spatial_contrasts"]
    rows = [[c["model"] if c["model"] == "Ridge" else "Gradient boosting", f"{c['from']} → {c['to']}", f"{c['mean_diff']:+,.0f}", f"[{c['ci_lo']:,.0f}, {c['ci_hi']:,.0f}]", f"{c['pct_change']:+.1f} %", pv(c["NB_p"])] for c in sc]
    rp.table(["Model", "Comparison", "ΔRMSE (USD)", "95 % CI", "Relative change", "NB p"], rows, "Effect of adding spatial distance features", widths=[2.8, 4.8, 2.0, 2.6, 1.7, 1.4], size=9.5, align=["left", "left", "center", "center", "center", "center"])
    rp.h(3, "4.3.5 Visualization and Interpretation")
    rp.figure(FIG / "fig08_ocean_proximity.png", "House value by ocean-proximity class (violin plots with quartiles)", 12.5)
    rp.figure(FIG / "fig09_moran_scatter.png", "Moran scatter-plot: house value of a district against the average value of its eight neighbours", 9.5)
    rp.figure(FIG / "fig10_spatial_features_gain.png", "Change in cross-validated RMSE when spatial distance features are added (95 % CI)", 11.5)
    big = {k: v for k, v in gt.items() if v["n"] >= 30}; top = max(big, key=lambda k: big[k]["median"]); low = min(big, key=lambda k: big[k]["median"]); nsig = sum(p["significant_5%"] for p in pw)
    isl = gt.get("ISLAND", {}).get("n")
    sig_all = all(c["NB_p"] < 0.05 for c in sc)
    rp.p(f"**Interpretation.** Geography matters a great deal. Among classes with a meaningful sample, median value ranges from {usd(gt[low]['median'])} in the {low} class to {usd(gt[top]['median'])} in the {top} class (the ISLAND class, with only {isl} training districts, is excluded from this comparison), and ocean proximity alone explains a large share of the rank variation (ε² = {o['epsilon2']:.2f}). {nsig} of the {len(pw)} pairwise differences are significant after Holm correction; the non-significant ones involve the ISLAND class, which has only {isl} training districts, so the test lacks power rather than demonstrating equality, and the <1H OCEAN versus NEAR OCEAN pair, whose medians are close. The very high Moran’s I of {mo['I']:.2f} shows that a district’s value is largely predictable from its neighbours’ values — prices form smooth spatial surfaces rather than random noise. This has two consequences: spatial information should help prediction, and random cross-validation is optimistic compared with predicting in a completely new region, because neighbouring districts appear in both training and test folds. This limitation is revisited in Section 5.")
    rp.p("Adding hub-distance features " + ("significantly lowers the RMSE for both models in every comparison (Table " + str(rp.tab_n) + ")" if sig_all else "changes the RMSE as shown in Table " + str(rp.tab_n)) + ". " + " ".join(f"For {'Ridge' if c['model']=='Ridge' else 'gradient boosting'} the change is {c['pct_change']:+.1f} % ({c['from']} → {c['to']})." for c in sc[:2]) + " The gain is largest for the linear model, which can only represent location linearly through raw coordinates and therefore benefits most from explicit distance features; the tree ensemble can already carve up the coordinate space and gains less, but its improvement is still statistically significant.")


def o4(rp, R):
    o = R["o4"]; rc_ = o["ratio_contrasts"]; ab = R["ablation"]
    sub(rp, 4, "Engineered Ratio Features (Objective 4)")
    rp.h(3, "4.4.1 Introduction")
    rp.p("The census file reports *totals* per block group — total rooms, total bedrooms, population and households. Such totals mostly measure the *size* of the district and are only indirectly related to how valuable its houses are. Features that divide one total by another describe the *character* of the housing: how spacious the dwellings are, what share of rooms are bedrooms, and how crowded the households are. This analysis tests whether three such ratio features carry information about value beyond the raw counts.")
    rp.h(3, "4.4.2 General Description")
    rp.p("First, the rank correlation of each raw count and each engineered ratio with the target is compared. Second, the out-of-sample effect of adding the ratio features is evaluated with the paired ablation introduced in Analysis 3, using both a linear and a gradient-boosting model and two baselines (with and without the spatial features) so that the conclusion does not depend on one specific configuration.")
    rp.h(3, "4.4.3 Specific Requirements, Functions and Formulas")
    rp.table(["Function", "Library", "Purpose"], [
        ["FeatureEngineer(ratios=True)", "src/features.py", "Creates the three ratio features inside the pipeline"],
        ["spearmanr", "scipy.stats", "Rank correlation of each feature with the target"],
        ["cross_validate (5×2 repeated CV)", "scikit-learn", "Paired CV-RMSE and R² for each feature set"],
        ["nadeau_bengio_ttest", "src/evaluate.py", "Corrected resampled t-test of the RMSE differences"],
    ], "Functions used in Analysis 4", widths=[5.0, 3.2, 6.4], align=["left", "left", "left"], size=10)
    rp.equation("rooms_per_household = total_rooms / households")
    rp.equation("bedrooms_per_room = total_bedrooms / total_rooms")
    rp.equation("population_per_household = population / households")
    rp.equation("ΔRMSE = RMSE_{with features} − RMSE_{without features}    (negative = improvement)")
    rp.h(3, "4.4.4 Analysis Results")
    cor = o["correlations"]
    rows = [[c["feature"], c["type"], f"{c['spearman_rho']:+.3f}", f"{c['skew']:.1f}"] for c in cor]
    rp.table(["Feature", "Type", "Spearman ρ with target", "Skewness"], rows, "Rank correlation of raw counts and engineered ratios with house value (training set)", widths=[5.2, 3.0, 3.8, 2.6], size=10, align=["left", "center", "center", "center"])
    rows = [[c["model"] if c["model"] == "Ridge" else "Gradient boosting", f"{c['from']} → {c['to']}", f"{c['mean_diff']:+,.0f}", f"[{c['ci_lo']:,.0f}, {c['ci_hi']:,.0f}]", f"{c['pct_change']:+.1f} %", f"{c['delta_R2']:+.3f}", pv(c["NB_p"])] for c in rc_]
    rp.table(["Model", "Comparison", "ΔRMSE (USD)", "95 % CI", "Rel. change", "ΔR²", "NB p"], rows, "Effect of adding ratio features (paired, corrected resampled t-test)", widths=[2.3, 3.6, 1.8, 2.5, 1.6, 1.6, 1.2], size=9, align=["left", "left"] + ["center"] * 5)
    rp.h(3, "4.4.5 Visualization and Interpretation")
    rp.figure(FIG / "fig11_ratio_features.png", "Ratio features: (a) correlation with the target, raw vs. engineered; (b) change in CV-RMSE when ratios are added", 15.5)
    raw_best = max((c for c in cor if c["type"] == "raw count"), key=lambda c: abs(c["spearman_rho"]))
    eng_best = max((c for c in cor if c["type"] == "engineered"), key=lambda c: abs(c["spearman_rho"]))
    nsig = sum(c["NB_p"] < 0.05 for c in rc_)
    rp.p(f"**Interpretation.** The strongest raw count, {raw_best['feature']}, has |ρ| = {abs(raw_best['spearman_rho']):.2f}, whereas the strongest engineered ratio, {eng_best['feature']}, has |ρ| = {abs(eng_best['spearman_rho']):.2f}; population alone is essentially uncorrelated with value (ρ = {[c for c in cor if c['feature']=='population'][0]['spearman_rho']:.3f}), yet population per household is clearly negatively related (crowding). The signs are economically sensible: roomier dwellings (more rooms per household) are associated with higher values, whereas a higher share of bedrooms among rooms and more people per household — typical of smaller or more crowded housing — are associated with lower values. The ratios are also less redundant than the raw counts, which are highly inter-correlated (mean |ρ| = {o['raw_counts_mean_abs_rho']:.2f} between pairs), whereas the ratios describe different aspects of the housing stock. They are, however, *more* skewed (Table {rp.tab_n-1}: up to {max(c['skew'] for c in cor):.0f} for population_per_household, caused by a few districts with very few households), which is why they are log-transformed in the pipeline and why tree-based models, which are insensitive to such tails, are a natural fit.")
    rp.p(f"In the ablation, adding the ratio features reduces the RMSE in {nsig} of {len(rc_)} comparisons at the 5 % level (Table {rp.tab_n}). For gradient boosting the improvement is {abs(rc_[1]['pct_change']):.1f} % without and {abs(rc_[3]['pct_change']):.1f} % with the spatial features; for Ridge it is {abs(rc_[0]['pct_change']):.1f} % and {abs(rc_[2]['pct_change']):.1f} %. The gains are modest in absolute size (roughly {min(abs(c['mean_diff']) for c in rc_):,.0f} to {max(abs(c['mean_diff']) for c in rc_):,.0f} dollars of RMSE) but consistent across models and baselines, which supports the conclusion that engineered ratios contain information the raw totals do not. Because the feature set with both ratio and spatial features gave the lowest gradient-boosting error (RMSE {usd(ab['F3 +ratios+spatial | HGB']['mean_RMSE'])}), it was selected as the feature set for the model comparison in Analysis 5.")
