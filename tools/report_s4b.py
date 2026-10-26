"""Section 4.5-4.6 (Objectives 5-6) and Section 5 (Conclusion)."""
from rc import FIG, usd, num, pct, pv, pvs, IMP
from report_s4a import sub

MODEL_INFO = [
    ["Dummy (mean)", "Baseline", "Always predicts the training-set mean; any useful model must beat it", "–"],
    ["Ridge", "Linear", "Least squares with L2 penalty on coefficients; interpretable, fast", "alpha"],
    ["Ridge (log-target)", "Linear", "Ridge fitted to ln(price) and back-transformed with exp; tests whether the log transform helps dollar-scale error", "alpha"],
    ["k-NN", "Instance-based", "Averages the values of the nearest districts in feature space; exploits spatial neighbourhoods", "n_neighbors, weights"],
    ["Random Forest", "Bagged trees", "Average of many de-correlated deep trees grown on bootstrap samples (Breiman, 2001)", "max_depth, min_samples_leaf, max_features"],
    ["HistGB", "Boosted trees", "Gradient boosting on histogram-binned features (scikit-learn); sequentially fits trees to residuals (Friedman, 2001)", "learning_rate, max_iter, max_leaf_nodes, min_samples_leaf, l2_regularization"],
    ["XGBoost", "Boosted trees", "Regularised gradient boosting with second-order optimisation (Chen and Guestrin, 2016)", "n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight"],
]


def o5(rp, R):
    o = R["o5"]; cv = o["cv_table"]; te = o["test_table"]; best, run = o["best_model"], o["runner_up"]; sig = o["significance"]
    imp, fs = R["o1"]["best_imputer"], R["ablation_choice"]["best_featureset"]
    sub(rp, 5, "Model Comparison and Selection (Objective 5)")
    rp.h(3, "4.5.1 Introduction")
    rp.p("Choosing the model that will be deployed is the central decision of a predictive-analysis project. A model that is too simple under-fits and misses real structure; one that is too flexible over-fits and fails on new data. This analysis compares one naïve baseline, a linear model and four non-linear model families under an identical, leakage-free protocol, selects the best on cross-validated evidence alone, and then tests whether the winner is *significantly* better than the runner-up rather than merely first in a ranking.")
    rp.h(3, "4.5.2 General Description")
    rp.p(f"All models share the same preprocessing pipeline (imputer: {IMP[imp]}; feature set: {fs}, i.e. ratio and spatial features included, as selected in Analyses 1–4) and receive exactly the same training data. The protocol has four steps. (1) Hyper-parameters are tuned on the training set only, by grid search for the small search spaces (Ridge, k-NN) and by randomised search for the larger ones, with 3-fold cross-validation and RMSE as the criterion. (2) Each tuned model is re-evaluated with 5-fold cross-validation on the training set; the model with the lowest mean CV-RMSE is selected. (3) Only then are all models refitted on the full training set and evaluated once on the held-out test set. (4) The winner is compared with every other model using the corrected resampled *t*-test on the CV fold scores and a paired bootstrap of the RMSE difference on the test set. Table {rp.tab_n+1} describes the seven models.")
    rp.table(["Model", "Family", "Idea", "Tuned hyper-parameters"], MODEL_INFO, "Models compared", widths=[2.6, 2.2, 6.4, 3.4], size=9, align=["left", "left", "left", "left"])
    rp.h(3, "4.5.3 Specific Requirements, Functions and Formulas")
    rp.table(["Function / class", "Library", "Purpose"], [
        ["Pipeline, ColumnTransformer", "scikit-learn", "Leakage-free preprocessing + model in one object"],
        ["GridSearchCV, RandomizedSearchCV", "scikit-learn", "Hyper-parameter tuning with 3-fold CV"],
        ["cross_validate (KFold, 5 folds)", "scikit-learn", "RMSE, MAE and R² per fold"],
        ["Ridge, KNeighborsRegressor, RandomForestRegressor, HistGradientBoostingRegressor", "scikit-learn", "Model implementations"],
        ["XGBRegressor", "xgboost", "Regularised gradient boosting"],
        ["learning_curve", "scikit-learn", "Training/validation error against training-set size"],
        ["paired_bootstrap_rmse_diff, nadeau_bengio_ttest", "src/evaluate.py", "Significance of the difference between two models"],
    ], "Functions used in Analysis 5", widths=[6.4, 2.8, 5.4], align=["left", "left", "left"], size=9.5)
    rp.p("Evaluation metrics for *n* test districts with true values *y*_{i} and predictions ŷ_{i}:")
    rp.equation("RMSE = √( (1/n) Σ ( y_{i} − ŷ_{i} )^{2} )        MAE = (1/n) Σ | y_{i} − ŷ_{i} |")
    rp.equation("R^{2} = 1 − Σ ( y_{i} − ŷ_{i} )^{2} / Σ ( y_{i} − mean(y) )^{2}")
    rp.equation("MedAPE = median( | y_{i} − ŷ_{i} | / y_{i} ) × 100")
    rp.p("The model families are defined by the equations below, where *T*_{b} is a tree grown on a bootstrap sample with random feature subsets, and *h*_{m} is a small tree fitted to the negative gradient of the loss with learning rate ν:")
    rp.equation("Ridge:  min_{w} Σ ( y_{i} − w·x_{i} )^{2} + α ‖w‖^{2}")
    rp.equation("k-NN:  ŷ(x) = Σ_{i ∈ N_{k}(x)} w_{i} y_{i},   w_{i} ∝ 1/d(x, x_{i})")
    rp.equation("Random forest:  ŷ(x) = (1/B) Σ_{b=1}^{B} T_{b}(x)")
    rp.equation("Gradient boosting:  F_{m}(x) = F_{m−1}(x) + ν h_{m}(x)")
    rp.equation("XGBoost objective:  Σ_{i} l( y_{i}, ŷ_{i} ) + Σ_{k} Ω( f_{k} ),   Ω(f) = γT + ½ λ Σ_{j} w_{j}^{2}")
    rp.p("The paired bootstrap resamples the test districts with replacement 2,000 times and recomputes RMSE(A) − RMSE(B) on each resample; the 2.5th and 97.5th percentiles give a 95 % confidence interval, which excludes zero when the difference is significant.")
    rp.h(3, "4.5.4 Analysis Results")
    tn = o["tuning"]; bp = o["best_params"]
    rows = []
    for m in tn:
        t = tn[m]; params = ", ".join(f"{k.replace('regressor__','')} = {v}" for k, v in bp[m].items()) or "–"
        rows.append([m, "none" if t["search"] == "none" else f"{t['search']} ({t['candidates']} of {t['space_size']})", params, "–" if t["best_tuning_RMSE"] is None else usd(t["best_tuning_RMSE"]), f"{t['seconds']:.0f}"])
    rp.table(["Model", "Search (candidates)", "Selected hyper-parameters", "Tuning RMSE", "Time (s)"], rows, "Hyper-parameter tuning (3-fold CV on the training set)", widths=[2.8, 2.9, 5.6, 2.0, 1.3], size=9, align=["left", "left", "left", "center", "center"])
    order = o["rank_by_cv"]
    rows = [[m, f"{usd(cv[m]['CV_RMSE'])} ± {num(cv[m]['CV_RMSE_sd'])}", usd(cv[m]["CV_MAE"]), f"{cv[m]['CV_R2']:.3f}", f"{cv[m]['fit_s_per_fold']:.1f}"] for m in order]
    rp.table(["Model", "CV-RMSE (mean ± sd)", "CV-MAE", "CV-R²", "Fit time per fold (s)"], rows, "Five-fold cross-validation of the tuned models (training set; basis of selection)", widths=[3.6, 4.2, 2.4, 1.8, 2.6], size=9.5, align=["left", "center", "center", "center", "center"], highlight_row=0)
    rows = [[m, usd(te[m]["RMSE"]), usd(te[m]["MAE"]), f"{te[m]['R2']:.3f}", f"{te[m]['MedAPE_%']:.1f} %", f"{te[m]['Within20%']:.1f} %", f"{te[m]['train_fit_s']:.1f}"] for m in order]
    rp.table(["Model", "Test RMSE", "Test MAE", "Test R²", "MedAPE", "Within ±20 %", "Fit (s)"], rows, "Held-out test-set performance (evaluated once, after selection)", widths=[3.4, 2.2, 2.2, 1.6, 1.8, 2.2, 1.2], size=9.5, align=["left"] + ["center"] * 6, highlight_row=0)
    rows = [[m, f"{sig[m]['cv_mean_diff']:+,.0f}", pv(sig[m]["NB_p"]), f"{sig[m]['test_RMSE_diff']:+,.0f}", f"[{sig[m]['boot_ci_lo']:,.0f}, {sig[m]['boot_ci_hi']:,.0f}]"] for m in order if m != best]
    rp.table(["Compared with", "Mean ΔCV-RMSE", "Corrected-t p", "ΔRMSE (test)", "95 % bootstrap CI"], rows, f"Is {best} significantly better? (negative = {best} better)", widths=[3.8, 2.6, 2.4, 2.6, 3.4], size=9.5, align=["left"] + ["center"] * 4)
    rp.h(3, "4.5.5 Visualization and Interpretation")
    rp.figure(FIG / "fig12_model_comparison.png", "Model comparison: (a) RMSE of each CV fold, (b) held-out test RMSE with R²", 15.5)
    rp.figure(FIG / "fig13_pred_vs_actual.png", f"Predicted against actual median house value for {best} on the test set", 10.5)
    rp.figure(FIG / "fig14_learning_curve.png", f"Learning curve of {best}", 11.0)
    d, b = te["Dummy (mean)"], te[best]; lc = o["learning_curve"]
    gap = lc[-1]["val_RMSE"] - lc[-1]["train_RMSE"]; gain = lc[-2]["val_RMSE"] - lc[-1]["val_RMSE"]
    r1 = sig[run]
    rp.p(f"**Interpretation.** {best} is the best model by cross-validation (CV-RMSE {usd(cv[best]['CV_RMSE'])}) and by the independent test set: RMSE {usd(b['RMSE'])}, MAE {usd(b['MAE'])} and R² = {b['R2']:.3f}. It reduces the error of the mean baseline by {(1-b['RMSE']/d['RMSE'])*100:.0f} %, and about {b['Within20%']:.0f} % of its predictions are within ±20 % of the true value (median absolute percentage error {b['MedAPE_%']:.1f} %). The ranking by CV and by test set is identical (Figure {rp.fig_n-2}), which gives confidence that the selection was not an artefact of one split. The tree ensembles (XGBoost, HistGB, Random Forest) form a clear leading group; k-NN is next, then the linear models, and the baseline is far behind. The large step between the linear models and the tree ensembles shows that the relationship between the district characteristics and value is substantially non-linear and interaction-driven (for example the effect of income depends on location).")
    rp.p(f"**Is the winner really better?** Against the runner-up, {run}, {best} has a lower test RMSE by {abs(r1['test_RMSE_diff']):,.0f} dollars (95 % bootstrap CI {r1['boot_ci_lo']:,.0f} to {r1['boot_ci_hi']:,.0f}, which excludes zero) and the corrected resampled *t*-test on the CV folds gives {pvs(r1['NB_p'])}. The advantage is therefore statistically reliable but small in practical terms (about {abs(r1['test_RMSE_diff'])/b['RMSE']*100:.0f} % of the RMSE); for an application in which training time or simplicity mattered, HistGB, which is several times faster to train ({te[run]['train_fit_s']:.1f} s against {b['train_fit_s']:.1f} s), would be a defensible alternative. Differences to Random Forest, k-NN and the linear models are larger and also significant.")
    rr, rl = te["Ridge"], te["Ridge (log-target)"]
    rp.p(f"**Design lesson on the target transformation.** Fitting the linear model to the logarithm of price and back-transforming (Ridge, log-target) gave a *higher* RMSE ({usd(rl['RMSE'])}) than fitting it to price directly ({usd(rr['RMSE'])}), although its median percentage error is lower ({rl['MedAPE_%']:.1f} % against {rr['MedAPE_%']:.1f} %) and its MAE is similar ({usd(rl['MAE'])} against {usd(rr['MAE'])}). The log model treats percentage errors evenly, so it describes typical districts better, but RMSE is dominated by the few very expensive districts, and back-transformation from the log scale under-predicts them. This is why the raw-target Ridge was used as the principal linear baseline: the choice of target transformation should follow the metric that matters for the application. It is also the reason a log-target handicap was removed from the main comparison after a preliminary check showed it understated what a linear model can achieve.")
    rp.p(f"**Bias–variance diagnosis.** The learning curve of {best} (Figure {rp.fig_n}) shows a large gap between training RMSE ({usd(lc[-1]['train_RMSE'])}) and validation RMSE ({usd(lc[-1]['val_RMSE'])}), the typical signature of a flexible boosted model that fits its training data very closely, yet the validation error is still falling as data are added (the last increase in training size lowered it by {usd(gain)}). Hence the model is not limited by capacity but by the amount of information; more districts, more features or stronger regularisation could further improve it, whereas simply tuning more would yield diminishing returns.")


def o6(rp, R):
    o = R["o6"]; seg = o["segments"]; c = o["censoring"]; sh = o["shap"]; best = R["o5"]["best_model"]; pi = o["perm_importance"]; pg = o["perm_groups"]
    sub(rp, 6, "Error Analysis, Effect of Censoring and Explainability (Objective 6)")
    rp.h(3, "4.6.1 Introduction")
    rp.p(f"An accurate average score is not enough to trust a model. Stakeholders need to know where it fails, whether its errors are systematic, how much of the error is due to limitations of the data (here the cap on the target) and which variables drive its predictions. This analysis examines the residuals of the selected model, {best}, on the held-out test set, quantifies the effect of the $500,001 censoring, tests whether the residuals still contain spatial structure, and explains the model with permutation importance, SHAP values and partial-dependence plots.")
    rp.h(3, "4.6.2 General Description")
    rp.p("Residuals (actual minus predicted) are summarised and tested for normality and constant variance. Errors are then broken down by ocean-proximity class, income quintile and true-value band. The effect of censoring is assessed in three ways: the share of squared error produced by capped districts, the metrics on uncapped districts only, and a sensitivity experiment in which the model is retrained without capped training districts. The spatial structure of the residuals is measured with Moran’s I and mapped. Finally, three complementary explanation methods are applied: permutation importance (the increase in test RMSE when a feature is shuffled), SHAP values (each feature’s additive contribution to each prediction) and partial-dependence plots (the average prediction as a feature is varied).")
    rp.h(3, "4.6.3 Specific Requirements, Functions and Formulas")
    rp.table(["Function", "Library", "Purpose"], [
        ["jarque_bera, skew, probplot", "scipy.stats", "Residual normality diagnostics"],
        ["lowess", "statsmodels", "Trend of residuals against predictions"],
        ["morans_i (custom)", "src/evaluate.py", "Spatial autocorrelation of residuals"],
        ["perm_importance (custom, grouped)", "notebook", "Permutation importance of single features and feature groups, 10 repetitions"],
        ["TreeExplainer, summary_plot", "shap", "Exact SHAP values for tree ensembles (Lundberg and Lee, 2017)"],
        ["partial_dependence", "scikit-learn", "Partial-dependence curves and surfaces"],
    ], "Functions used in Analysis 6", widths=[4.8, 2.8, 7.0], align=["left", "left", "left"], size=10)
    rp.equation("e_{i} = y_{i} − ŷ_{i}        Bias = (1/n) Σ e_{i}        JB = (n/6) [ S^{2} + (K − 3)^{2}/4 ]")
    rp.equation("ΔRMSE_{j} = RMSE( X with column j shuffled ) − RMSE( X )")
    rp.equation("SHAP value:   φ_{j} = Σ_{S ⊆ F∖{j}} [ |S|!(|F|−|S|−1)! / |F|! ] ( f_{S∪{j}}(x) − f_{S}(x) )")
    rp.equation("Partial dependence:   PD_{j}(v) = (1/n) Σ_{i} f( x_{i,−j}, x_{j} = v )")
    rp.p("where *S* and *K* in the Jarque–Bera statistic are the skewness and kurtosis of the residuals, *F* is the set of features in the SHAP formula, and the partial-dependence average runs over the sampled districts.")
    rp.h(3, "4.6.4 Analysis Results")
    rp.table(["Residual statistic", "Value"], [
        ["Mean residual (bias)", usd(o["bias_mean_residual"])], ["Standard deviation", usd(o["residual_sd"])],
        ["Skewness", f"{o['residual_skew']:.2f}"], ["Excess kurtosis", f"{o['residual_kurtosis']:.2f}"],
        ["Jarque–Bera statistic (p)", f"{o['jarque_bera']:,.0f} ({pvs(o['jarque_bera_p'])})"],
        ["Spearman ρ between |residual| and prediction", f"{o['abs_resid_vs_pred_spearman']:.2f}"]], "Summary of test-set residuals", widths=[8.0, 5.0], size=10.5, align=["left", "center"])
    def segtab(name, title, cap):
        rows = [[k, num(v["n"]), usd(v["RMSE"]), usd(v["MAE"]), f"{v['bias']:+,.0f}", f"{v['MedAPE_%']:.1f} %"] for k, v in seg[name].items()]
        rp.table([title, "n", "RMSE", "MAE", "Bias (actual − predicted)", "MedAPE"], rows, cap, widths=[3.6, 1.3, 2.2, 2.2, 3.2, 1.8], size=9.5, align=["left"] + ["center"] * 5)
    segtab("ocean_proximity", "Ocean proximity", "Test error by ocean-proximity class")
    segtab("income_quintile", "Income quintile", "Test error by income quintile")
    segtab("value_band", "True-value band", "Test error by true-value band (selection on the true value, see text)")
    m_all, m_unc, m_re = c["metrics_all"], c["metrics_uncapped_rows"], c["trained_without_capped_on_uncapped_test"]
    rp.table(["Quantity", "Value"], [
        ["Capped districts in the test set", f"{c['n_test_capped']} ({c['pct_test_capped']:.1f} %)"],
        ["Share of total squared error from capped districts", f"{c['rmse_share_from_capped_rows_pct']:.1f} %"],
        ["Mean prediction for capped districts (true value 500,001)", usd(c["mean_pred_for_capped"])],
        ["Capped districts predicted below the cap", f"{c['share_capped_underpredicted']:.0f} %"],
        ["RMSE / R², all test districts", f"{usd(m_all['RMSE'])} / {m_all['R2']:.3f}"],
        ["RMSE / R², uncapped districts only", f"{usd(m_unc['RMSE'])} / {m_unc['R2']:.3f}"],
        ["RMSE / R², retrained without capped training rows, evaluated on uncapped districts", f"{usd(m_re['RMSE'])} / {m_re['R2']:.3f}"]],
        "Effect of target censoring on test performance", widths=[10.0, 4.4], size=10, align=["left", "center"])
    mo = o["moran_residuals"]
    rp.p(f"**Spatial structure of residuals.** Moran’s I of the test residuals is {mo['I']:.3f} (permutation z = {mo['z']:.1f}, {pvs(mo['p'])}), compared with {R['o3']['moran_I']['I']:.2f} for the raw target.")
    w = o["worst10"]
    rows = [[f"{x['longitude']:.2f}", f"{x['latitude']:.2f}", f"{x['median_income']:.2f}", x["ocean_proximity"], num(x["y"]), num(x["pred"]), f"{x['res']:+,.0f}"] for x in w]
    rp.table(["Longitude", "Latitude", "Income", "Ocean proximity", "Actual", "Predicted", "Residual"], rows, "The ten largest test errors", widths=[1.9, 1.8, 1.5, 2.8, 2.0, 2.2, 2.2], size=9, align=["center"] * 7)
    pi_sorted = sorted(pi.items(), key=lambda kv: -kv[1]["delta_RMSE"])
    rows = [[k, usd(v["delta_RMSE"]), usd(v["sd"])] for k, v in pi_sorted]
    rp.table(["Feature shuffled", "Increase in test RMSE", "sd over repetitions"], rows, "Permutation importance of the original input variables", widths=[5.4, 4.6, 4.0], size=10, align=["left", "center", "center"])
    rows = [[k, usd(v["delta_RMSE"]), usd(v["sd"])] for k, v in sorted(pg.items(), key=lambda kv: -kv[1]["delta_RMSE"])]
    rp.table(["Feature group shuffled", "Increase in test RMSE", "sd"], rows, "Grouped permutation importance", widths=[7.4, 3.8, 2.8], size=10, align=["left", "center", "center"])
    ms = list(sh["mean_abs"].items())[:10]
    rp.table(["Model feature", "Mean |SHAP| (USD)"], [[k, num(v)] for k, v in ms], f"Ten largest mean absolute SHAP values ({sh['method']}, {sh['n_samples']:,} test districts)", widths=[7.0, 4.0], size=10, align=["left", "center"])
    rp.h(3, "4.6.5 Visualization and Interpretation")
    rp.figure(FIG / "fig15_residual_diagnostics.png", "Residual diagnostics: (a) residuals against predictions, (b) distribution, (c) normal Q–Q plot", 15.5)
    rp.p(f"**Residuals.** The model is essentially unbiased (mean residual {usd(o['bias_mean_residual'])}), but its errors are heavy-tailed (excess kurtosis {o['residual_kurtosis']:.1f}) and right-skewed (skewness {o['residual_skew']:.2f}), so normality is rejected (Jarque–Bera {pvs(o['jarque_bera_p'])}) and prediction intervals based on a normal assumption would be unreliable. Errors grow with the predicted value (ρ = {o['abs_resid_vs_pred_spearman']:.2f}): expensive districts are predicted with larger absolute errors. The diagonal band visible in Figure {rp.fig_n}(a) is the signature of the censored target: every district whose true value is 500,001 produces a residual that decreases linearly with its prediction.")
    rp.figure(FIG / "fig16_segment_errors.png", "Test RMSE by (a) ocean proximity, (b) income quintile, (c) true-value band; labels give the median absolute percentage error and n", 15.5)
    oc = seg["ocean_proximity"]; iq = seg["income_quintile"]; vb = seg["value_band"]
    reliable = {k: v for k, v in oc.items() if v["n"] >= 30}
    hi = max(reliable, key=lambda k: reliable[k]["RMSE"]); lo = min(reliable, key=lambda k: reliable[k]["RMSE"])
    rp.p(f"**Segments.** Among classes with a meaningful sample, the absolute error is lowest for {lo} districts (RMSE {usd(oc[lo]['RMSE'])}) and highest for {hi} districts (RMSE {usd(oc[hi]['RMSE'])}), while the *relative* error is similar everywhere (median absolute percentage error {min(v['MedAPE_%'] for v in reliable.values()):.1f}–{max(v['MedAPE_%'] for v in reliable.values()):.1f} %). The ISLAND class contains only {oc['ISLAND']['n']} test districts, so its large error is anecdotal. Across income quintiles the RMSE rises from {usd(iq['Q1 (lowest)']['RMSE'])} to {usd(iq['Q5 (highest)']['RMSE'])} while the relative error falls from {iq['Q1 (lowest)']['MedAPE_%']:.1f} % to {iq['Q5 (highest)']['MedAPE_%']:.1f} %: affluent districts have larger dollar errors simply because prices are larger. The value-band breakdown shows the classic regression-to-the-mean pattern (note that grouping by the true value guarantees such a pattern to some extent): cheap districts are over-predicted by {usd(abs(vb['<150k']['bias']))} on average, whereas districts worth 300,000–500,000 are under-predicted by {usd(vb['300-500k']['bias'])} and capped districts by {usd(vb['capped (500,001)']['bias'])}.")
    rp.p(f"**Effect of censoring.** The {c['n_test_capped']} capped districts are only {c['pct_test_capped']:.1f} % of the test set but generate {c['rmse_share_from_capped_rows_pct']:.0f} % of the squared error, and {c['share_capped_underpredicted']:.0f} % of them are predicted below the cap, on average by {usd(500001 - c['mean_pred_for_capped'])}. Their true value is at least 500,001 and probably much higher, so the real shortfall is larger than measured. Excluding them lowers the RMSE from {usd(m_all['RMSE'])} to {usd(m_unc['RMSE'])}. Retraining without capped training districts improves the score on uncapped districts only slightly (RMSE {usd(m_re['RMSE'])}), and this variant cannot be deployed because it would be blind to the most expensive districts. The practical conclusion is that the cap limits the information in the data about the top of the market, and no modelling technique can fully recover it without censoring-aware methods (Section 6).")
    rp.figure(FIG / "fig17_residual_map.png", "Map of test residuals (actual minus predicted); blue = under-predicted, red = over-predicted", 11.0)
    rp.p(f"**Spatial residuals.** The residuals retain a small but statistically significant positive spatial autocorrelation (I = {mo['I']:.2f}, {pvs(mo['p'])}), far below the {R['o3']['moran_I']['I']:.2f} of the raw target. The model, helped by the coordinate and hub-distance features, has therefore absorbed most of the spatial structure, but neighbouring districts still share some unexplained local effects, visible in Figure {rp.fig_n} as occasional patches of same-coloured points, mainly in the densely populated Bay area and Los Angeles basin. Local features (neighbourhood amenities, school quality, zoning) could capture them. The largest individual errors (Table {rp.tab_n - 3}) are mostly capped districts predicted far below the cap, together with a few districts with implausibly low values (for example 17,500 and 55,000) in Los Angeles that the model predicts at over 200,000; such cases may reflect atypical block groups and illustrate irreducible noise in aggregate census data.")
    rp.figure(FIG / "fig18_permutation_importance.png", "Permutation importance on the test set: (a) single input variables, (b) feature groups", 15.5)
    top3 = [k for k, _ in pi_sorted[:3]]
    rp.p(f"**Permutation importance.** Shuffling longitude, {top3[1]} or latitude damages the predictions most, and the top three variables account for {o['top3_share_pct']:.0f} % of the total importance; shuffling the pair longitude and latitude jointly raises the test RMSE by {usd(pg['location (lon + lat)']['delta_RMSE'])}, which is more than the {usd(pg['size counts (rooms, bedrooms, pop., households)']['delta_RMSE'])} of the four size counts together, so *where* a district is matters more than *how big* it is. Median income ranks {[k for k,_ in pi_sorted].index('median_income')+1}th ({usd(pi['median_income']['delta_RMSE'])}); the high rank of total_rooms is largely an artefact of the pipeline, explained below. Two cautions apply. First, a raw column such as total_rooms enters the model also through the derived ratio features, so shuffling it disrupts several model inputs at once and its importance partly reflects those ratios. Second, strongly correlated variables share importance, so individual values should be read as ranks, not as additive shares.")
    rp.figure(FIG / "fig19_shap_summary.png", "SHAP summary plot: each dot is a district; colour shows the feature value, horizontal position its effect on the prediction", 13.0)
    ma = sh["mean_abs"]; ms_ = list(ma.items())
    loc = sum(v for k, v in ma.items() if k in ("longitude", "latitude") or k.startswith("dist_") or k.startswith("ocean_proximity"))
    rp.p(f"**SHAP.** Exact Tree SHAP values for {sh['n_samples']:,} test districts give a ranking that differs from permutation importance in an instructive way. Median income has the largest single mean absolute SHAP value ({usd(ms_[0][1])}), followed by {ms_[1][0]} ({usd(ms_[1][1])}) and {ms_[2][0]} ({usd(ms_[2][1])}); longitude and latitude (about {usd(ma['longitude'])} and {usd(ma['latitude'])}) come next, then the distances to the hubs. The reason is that location information is *spread over many correlated features* — the coordinates, five hub distances and the ocean-proximity indicators — and SHAP divides the credit among them, whereas permuting a raw coordinate destroys the whole location signal at once. Adding up the mean absolute SHAP values of all location-related features gives {usd(loc)}, against {usd(ms_[0][1])} for income, so location is the dominant driver overall, in agreement with the permutation analysis, with income as the leading single feature.")
    rp.p("The direction of the effects is economically coherent (Figure " + str(rp.fig_n) + "). High median income pushes the predicted value up, and the spread of the income dots to the right shows that high income can add well over $100,000 to a single prediction. High people per household (crowding) and an INLAND location push the prediction down. Districts in the south and west (low latitude, more negative longitude) receive positive contributions, consistent with the coastal premium seen in Analysis 3, and being close to a major hub (small distance) raises the prediction.")
    rp.figure(FIG / "fig20_partial_dependence.png", "Partial dependence of the predicted value on (a) median income, (b) housing median age and (c) location", 15.5)
    rp.p("**Partial dependence.** Panel (a) shows an almost monotonic increase in the predicted value with income across the central 90 % of districts, with a steeper slope above an income of about 4.5 (roughly $45,000 in 1990 dollars): the model has learned a convex, accelerating effect rather than the straight line assumed by the linear model. Panel (b) shows that housing age has only a small average effect, within a range of roughly $12,000, with the lowest predictions for districts with houses of 15–17 years median age and higher predictions for very old housing stock. Panel (c) maps the average predicted value over the geographic coordinates, with cells far from any district masked: the highest predicted values lie along the coast around the San Francisco Bay area and the Los Angeles coast, and the lowest in the far north and the south-eastern interior, reproducing the coastal gradient of Analysis 3.")


def conclusion(rp, R):
    rp.h(1, "5. Conclusion")
    o1, o2, o3, o4, o5, o6 = (R[k] for k in ("o1", "o2", "o3", "o4", "o5", "o6")); q = R["quality"]
    best = o5["best_model"]; te = o5["test_table"][best]; sc = o3["spatial_contrasts"]; rc_ = o4["ratio_contrasts"]; gt = o3["group_table"]
    rp.h(2, "5.1 Summary of Findings")
    rp.p("The project built a complete, leakage-free predictive-analysis pipeline for the California housing data and answered six research objectives with measured evidence. Table " + str(rp.tab_n + 1) + " summarises the answers.")
    me = o1["masked_experiment"]
    rows = [
        ["O1", f"Missingness in total_bedrooms ({q['missing']['total_bedrooms']} rows) shows no relation to other variables; {IMP[o1['best_imputer']]} imputation recovers hidden values with RMSE {me[o1['best_imputer']]['RMSE']:.0f} rooms against {me['median']['RMSE']:.0f} for the median; downstream effect negligible. {q['target_capped_pct']:.1f} % of targets, {q['age_capped_pct']:.1f} % of ages are censored."],
        ["O2", f"Income is strongly related to value (Pearson r = {o2['pearson']:.2f}); +$10,000 income ↔ +{usd(o2['ols_slope'])}; elasticity {o2['loglog_elasticity']:.2f}; mild concavity; heteroscedastic."],
        ["O3", f"Ocean proximity has a large effect (ε² = {o3['epsilon2']:.2f}); strong spatial autocorrelation (Moran’s I = {o3['moran_I']['I']:.2f}); spatial features significantly improve prediction (−{abs(sc[0]['pct_change']):.1f} % RMSE for Ridge, −{abs(sc[1]['pct_change']):.1f} % for gradient boosting)."],
        ["O4", f"Ratio features are better correlated with value than raw counts and reduce RMSE by {abs(rc_[1]['pct_change']):.1f} % (gradient boosting) and {abs(rc_[0]['pct_change']):.1f} % (Ridge) on the raw feature set."],
        ["O5", f"{best} is best: test RMSE {usd(te['RMSE'])}, MAE {usd(te['MAE'])}, R² {te['R2']:.2f}; significantly better than the runner-up {o5['runner_up']} ({pvs(o5['significance'][o5['runner_up']]['NB_p'])})."],
        ["O6", f"Errors are unbiased but heavy-tailed; capped districts ({o6['censoring']['pct_test_capped']:.1f} % of test) cause {o6['censoring']['rmse_share_from_capped_rows_pct']:.0f} % of the squared error; location, income and household density drive predictions; residuals retain a small spatial pattern (I = {o6['moran_residuals']['I']:.2f})."],
    ]
    rp.table(["Obj.", "Main finding"], rows, "Summary of findings by objective", widths=[1.2, 13.4], size=10, align=["center", "left"])
    rp.h(2, "5.2 Recommendations")
    rp.bullets([
        "**Use a gradient-boosted tree model with ratio and spatial features** for district-level valuation on this kind of data; the evidence (Analyses 3–5) shows consistent, significant gains from both feature groups.",
        "**Treat censored targets explicitly.** Report errors separately for capped districts, and consider censoring-aware methods before using the model for the top of the market.",
        "**Validate spatially.** Because values are strongly clustered in space, evaluate any model intended for new regions with geographically blocked validation, not only random folds.",
        "**Report uncertainty.** Because residuals are heavy-tailed and heteroscedastic, use prediction intervals that do not assume normality instead of single-number predictions.",
        "**Prioritise better location data** (coast distance, amenities, schools) over additional tuning, since the learning curve indicates the model is information-limited and residuals retain spatial structure.",
    ])
    rp.h(2, "5.3 Limitations")
    rp.bullets([
        "**Age of the data.** Prices and incomes are from 1990; the model illustrates a method and must not be used to value present-day property.",
        "**Censoring.** The target, housing age and income are capped; the true values of the capped districts are unknown.",
        "**Aggregation.** Each row is a block-group summary; relationships found at this level do not necessarily hold for individual houses (ecological fallacy).",
        "**Spatial dependence and optimistic validation.** Random cross-validation lets neighbouring districts appear in both training and test sets, so the reported accuracy applies to interpolation within the observed region, not extrapolation to new regions.",
        "**Small classes.** The ISLAND class has only five districts in the entire file, so no conclusion about it is reliable.",
        "**Limited tuning budget.** Randomised searches evaluated 6–12 candidate settings on a single CPU core; a larger search might narrow the gaps between the top three models.",
        "**Researcher choices.** The five hub cities, the imputation candidates and the feature sets were chosen by the author; other reasonable choices might change effect sizes, although the qualitative conclusions were stable across the models and baselines tested.",
    ])
