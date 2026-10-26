"""Project-wide constants: paths, random seed, column names, reference coordinates."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "housing.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

RANDOM_STATE = 42          # single seed used everywhere for reproducibility
TARGET = "median_house_value"
CAP_VALUE = 500001         # the target is right-censored (capped) at this value in the 1990 census release

NUM_BASE = ["longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income"]
CAT_COLS = ["ocean_proximity"]

# Major urban hubs of California (latitude, longitude) used for the spatial features.
# These are public geographic facts, not derived from the target -> no leakage.
HUBS = {
    "SanFrancisco": (37.7749, -122.4194),
    "LosAngeles":   (34.0522, -118.2437),
    "SanDiego":     (32.7157, -117.1611),
    "SanJose":      (37.3382, -121.8863),
    "Sacramento":   (38.5816, -121.4944),
}
