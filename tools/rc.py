"""Report context: results loader and number formatters (every number in the report comes from metrics.json)."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
import os
def load(): return json.load(open(os.environ.get("RESULTS_JSON", ROOT / "results" / "metrics.json")))
def usd(x, d=0): return f"−${abs(x):,.{d}f}" if x < 0 else f"${x:,.{d}f}"
def num(x, d=0): return f"{x:,.{d}f}"
def pct(x, d=1): return f"{x:.{d}f}%"
def pv(p):
    if p is None: return "n/a"
    return "< 0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.1 else f"{p:.2f}")
TITLE_MAIN = "CALIFORNIA HOUSING PRICE PREDICTION"
TITLE_SUB = "Predicting Median House Values of California Census Districts Using Statistical Analysis and Machine Learning"
TITLE = f"{TITLE_MAIN} — {TITLE_SUB}"
STUDENT = "Sadu Ram Charan"

def pvs(p):
    """p-value phrase with operator, e.g. 'p < 0.001' or 'p = 0.002'."""
    s = pv(p)
    return f"p {s}" if s.startswith("<") else f"p = {s}"
IMP = {"knn": "k-NN", "median": "median", "group_median": "group-median", "ratio": "ratio", "iterative": "iterative"}
