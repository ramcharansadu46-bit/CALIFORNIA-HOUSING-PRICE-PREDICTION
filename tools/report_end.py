"""Future scope and references."""

REFS = [
 "Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.",
 "Breusch, T. S., and Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. *Econometrica*, 47(5), 1287–1294.",
 "Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794).",
 "Cleveland, W. S. (1979). Robust locally weighted regression and smoothing scatterplots. *Journal of the American Statistical Association*, 74(368), 829–836.",
 "Cover, T., and Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27.",
 "Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *The Annals of Statistics*, 7(1), 1–26.",
 "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232.",
 "Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow* (2nd ed.). O’Reilly Media. Dataset file: https://github.com/ageron/handson-ml2/tree/master/datasets/housing",
 "Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362.",
 "Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.",
 "Hoerl, A. E., and Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.",
 "Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.",
 "Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science and Engineering*, 9(3), 90–95.",
 "Kruskal, W. H., and Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *Journal of the American Statistical Association*, 47(260), 583–621.",
 "Little, R. J. A., and Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed.). Wiley.",
 "Lundberg, S. M., and Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30* (pp. 4765–4774).",
 "Mann, H. B., and Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *The Annals of Mathematical Statistics*, 18(1), 50–60.",
 "McKinney, W. (2010). Data structures for statistical computing in Python. In *Proceedings of the 9th Python in Science Conference* (pp. 56–61).",
 "Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1/2), 17–23.",
 "Nadeau, C., and Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239–281.",
 "Nugent, C. (n.d.). *California Housing Prices* [Data set]. Kaggle. https://www.kaggle.com/datasets/camnugent/california-housing-prices",
 "OpenML. (n.d.). California housing dataset, ID 43705. https://api.openml.org/d/43705/json",
 "Pace, R. K., and Barry, R. (1997). Sparse spatial autoregressions. *Statistics and Probability Letters*, 33(3), 291–297.",
 "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.",
 "Seabold, S., and Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. In *Proceedings of the 9th Python in Science Conference* (pp. 92–96).",
 "Tomczak, M., and Tomczak, E. (2014). The need to report effect size estimates revisited: An overview of some recommended measures of effect size. *Trends in Sport Sciences*, 21(1), 19–25.",
 "Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.",
 "Virtanen, P., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272.",
 "Waskom, M. L. (2021). seaborn: statistical data visualization. *Journal of Open Source Software*, 6(60), 3021.",
]


def future_scope(rp, R):
    rp.h(1, "6. Future Scope")
    rp.p("The project can be extended in several directions that address the limitations identified above.")
    rp.bullets([
        "**Spatially blocked validation.** Because values are strongly autocorrelated in space (Analysis 3), random cross-validation is optimistic. Grouping districts into geographic blocks (for example by k-means clusters of coordinates or by county) and holding out entire blocks would estimate how well the model generalises to unseen regions.",
        "**Censoring-aware models.** The $500,001 cap could be handled with a Tobit-style likelihood, survival-analysis methods that treat capped values as right-censored, or two-stage models that first classify capped districts and then regress on the rest.",
        "**Richer location features.** Distance to the actual coastline, elevation, school quality, crime and transport accessibility, or geohash/embedding representations of location, could be added from open geographic sources.",
        "**More recent data.** Retraining on contemporary American Community Survey or local listing data would make the model usable for present-day questions and would allow temporal validation.",
        "**Uncertainty quantification.** Quantile regression, conformal prediction or gradient-boosting prediction intervals would give a range around each prediction, which is essential for decision-making.",
        "**Deployment.** The saved pipeline could be wrapped in a REST API or a small web application, with monitoring for data drift.",
        "**Model stacking and neural networks.** Stacked ensembles or tabular neural networks with spatial embeddings could be benchmarked against the tuned tree ensembles under the same protocol.",
    ])


def references(rp, R):
    rp.h(1, "7. References")
    for i, r in enumerate(REFS, 1):
        par = rp.p(f"[{i}] " + r, align="left")
        par.paragraph_format.left_indent = __import__("docx").shared.Cm(1.0); par.paragraph_format.first_line_indent = __import__("docx").shared.Cm(-1.0)
        par.paragraph_format.line_spacing = 1.5
