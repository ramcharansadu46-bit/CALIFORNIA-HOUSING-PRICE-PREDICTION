"""Leakage-safe preprocessing components.

Every transformer follows the scikit-learn fit/transform contract, so ALL statistics
(medians, neighbour structure, scaling constants) are learned from the training fold only
when the objects are used inside a Pipeline / cross-validation.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (required to import IterativeImputer)
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import CAT_COLS, HUBS, NUM_BASE

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between points given in decimal degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


class BedroomImputer(BaseEstimator, TransformerMixin):
    """Impute the missing ``total_bedrooms`` values with a selectable strategy.

    strategy: 'median' | 'group_median' (by ocean_proximity) | 'ratio' (households x median
    bedrooms-per-household) | 'knn' | 'iterative'.  All parameters are learned in ``fit``.
    """

    def __init__(self, strategy="median", n_neighbors=5):
        self.strategy = strategy
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        s = self.strategy
        self.global_median_ = X["total_bedrooms"].median()
        self.min_observed_ = X["total_bedrooms"].min()   # imputed counts are clipped to this floor
        if s == "group_median":
            self.group_median_ = X.groupby("ocean_proximity")["total_bedrooms"].median()
        elif s == "ratio":
            self.ratio_ = (X["total_bedrooms"] / X["households"]).median()
        elif s in ("knn", "iterative"):
            self.scaler_ = StandardScaler().fit(X[NUM_BASE])
            Z = self.scaler_.transform(X[NUM_BASE])
            if s == "knn":
                self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors).fit(Z)
            else:
                self.imputer_ = IterativeImputer(estimator=BayesianRidge(), max_iter=10,
                                                 random_state=0).fit(Z)
        elif s != "median":
            raise ValueError(f"unknown strategy {s!r}")
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        miss = X["total_bedrooms"].isna()
        if not miss.any():
            return X
        s = self.strategy
        if s == "median":
            X.loc[miss, "total_bedrooms"] = self.global_median_
        elif s == "group_median":
            fill = X.loc[miss, "ocean_proximity"].map(self.group_median_).fillna(self.global_median_)
            X.loc[miss, "total_bedrooms"] = fill
        elif s == "ratio":
            X.loc[miss, "total_bedrooms"] = X.loc[miss, "households"] * self.ratio_
        else:  # knn / iterative
            Z = self.scaler_.transform(X[NUM_BASE])
            Zi = self.imputer_.transform(Z)
            col = NUM_BASE.index("total_bedrooms")
            vals = Zi[:, col] * self.scaler_.scale_[col] + self.scaler_.mean_[col]
            # model-based imputers can extrapolate to impossible (<=0) counts -> clip to observed floor
            X.loc[miss, "total_bedrooms"] = np.maximum(vals[miss.values], self.min_observed_)
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Optionally add (a) per-household ratio features and (b) spatial distance features."""

    def __init__(self, ratios=False, spatial=False):
        self.ratios = ratios
        self.spatial = spatial

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if self.ratios:
            X["rooms_per_household"] = X["total_rooms"] / X["households"]
            X["bedrooms_per_room"] = X["total_bedrooms"] / X["total_rooms"]
            X["population_per_household"] = X["population"] / X["households"]
        if self.spatial:
            cols = []
            for name, (la, lo) in HUBS.items():
                c = f"dist_{name}_km"
                X[c] = haversine_km(X["latitude"], X["longitude"], la, lo)
                cols.append(c)
            X["dist_nearest_hub_km"] = X[cols].min(axis=1)
        return X


def build_preprocessor(impute="median", ratios=False, spatial=False):
    """Imputer -> feature engineering -> (log1p+scale | scale | one-hot) column transformer."""
    log_cols = ["total_rooms", "total_bedrooms", "population", "households"]
    scale_cols = ["longitude", "latitude", "housing_median_age", "median_income"]
    if ratios:
        log_cols += ["rooms_per_household", "population_per_household"]
        scale_cols += ["bedrooms_per_room"]
    if spatial:
        scale_cols += [f"dist_{n}_km" for n in HUBS] + ["dist_nearest_hub_km"]
    ct = ColumnTransformer(
        [("log", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                           ("scale", StandardScaler())]), log_cols),
         ("num", StandardScaler(), scale_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)],
        remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("impute", BedroomImputer(strategy=impute)),
                     ("engineer", FeatureEngineer(ratios=ratios, spatial=spatial)),
                     ("columns", ct)])
