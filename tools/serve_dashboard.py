"""Lightweight, robust Python HTTP server for California Housing Predictive Analysis Dashboard.
Serves interactive prediction API, benchmark data, static assets, and figures.
"""
import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from src.config import HUBS, MODEL_DIR, RESULT_DIR, TARGET

PORT = 5050
MODEL = None
TEST_DATA = None


def load_resources():
    global MODEL, TEST_DATA
    try:
        model_path = MODEL_DIR / "best_model.joblib"
        if model_path.exists():
            MODEL = joblib.load(model_path)
            print(f"[Dashboard] Loaded production model from {model_path}")
        else:
            print(f"[Dashboard] Warning: Model not found at {model_path}")
    except Exception as e:
        print(f"[Dashboard] Error loading model: {e}")

    try:
        test_path = ROOT / "data" / "processed" / "test.csv"
        if test_path.exists():
            TEST_DATA = pd.read_csv(test_path)
            print(f"[Dashboard] Loaded {len(TEST_DATA)} test records from {test_path}")
    except Exception as e:
        print(f"[Dashboard] Error loading test data: {e}")


class DashboardHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self.send_json({
                "status": "healthy",
                "model_loaded": MODEL is not None,
                "model_name": "XGBoost Regressor (KNN + Ratios + Spatial)",
                "port": PORT,
                "project": "California Housing Predictive Analysis (CA2)"
            })
            return

        elif path == "/api/metrics":
            metrics_path = RESULT_DIR / "metrics.json"
            model_comp_path = RESULT_DIR / "model_comparison.csv"
            card_path = MODEL_DIR / "model_card.json"

            comp_records = []
            if model_comp_path.exists():
                try:
                    df = pd.read_csv(model_comp_path)
                    if df.columns[0] == "Unnamed: 0" or df.columns[0] == "":
                        df.rename(columns={df.columns[0]: "Model"}, inplace=True)
                    comp_records = df.to_dict(orient="records")
                except Exception as e:
                    print("Error reading model_comparison.csv:", e)

            card_data = {}
            if card_path.exists():
                try:
                    with open(card_path, "r", encoding="utf-8") as f:
                        card_data = json.load(f)
                except Exception as e:
                    print("Error reading model_card.json:", e)

            self.send_json({
                "model_comparison": comp_records,
                "model_card": card_data,
                "urban_hubs": HUBS
            })
            return

        elif path == "/api/sample":
            if TEST_DATA is not None:
                sample_rows = TEST_DATA.sample(min(10, len(TEST_DATA)), random_state=None)
                self.send_json(sample_rows.to_dict(orient="records"))
            else:
                self.send_json([], status=404)
            return

        elif path.startswith("/figures/"):
            rel_file = path[len("/figures/"):]
            fig_path = ROOT / "figures" / rel_file
            if fig_path.exists() and fig_path.is_file():
                self.send_file(fig_path, "image/png")
                return

        elif path.startswith("/report/"):
            rel_file = path[len("/report/"):]
            rep_path = ROOT / "report" / rel_file
            if rep_path.exists() and rep_path.is_file():
                content_type = "application/pdf" if rel_file.endswith(".pdf") else "application/octet-stream"
                self.send_file(rep_path, content_type)
                return

        # Serve static web assets from web/
        if path == "/" or path == "/index.html":
            target = ROOT / "web" / "index.html"
            self.send_file(target, "text/html; charset=utf-8")
            return
        elif path == "/style.css":
            target = ROOT / "web" / "style.css"
            self.send_file(target, "text/css; charset=utf-8")
            return
        elif path == "/app.js":
            target = ROOT / "web" / "app.js"
            self.send_file(target, "application/javascript; charset=utf-8")
            return
        else:
            # Fallback to web directory
            asset_path = ROOT / "web" / path.lstrip("/")
            if asset_path.exists() and asset_path.is_file():
                content_type = "text/plain"
                if asset_path.suffix == ".html":
                    content_type = "text/html"
                elif asset_path.suffix == ".css":
                    content_type = "text/css"
                elif asset_path.suffix == ".js":
                    content_type = "application/javascript"
                elif asset_path.suffix in [".png", ".jpg", ".jpeg"]:
                    content_type = f"image/{asset_path.suffix[1:]}"
                self.send_file(asset_path, content_type)
                return

        self.send_error(404, f"File not found: {path}")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            if MODEL is None:
                self.send_json({"error": "Model not loaded on server"}, status=500)
                return

            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                data = json.loads(body)

                df = pd.DataFrame([{
                    "longitude": float(data.get("longitude", -122.23)),
                    "latitude": float(data.get("latitude", 37.88)),
                    "housing_median_age": float(data.get("housing_median_age", 30.0)),
                    "total_rooms": float(data.get("total_rooms", 2000.0)),
                    "total_bedrooms": float(data.get("total_bedrooms", 400.0)),
                    "population": float(data.get("population", 1200.0)),
                    "households": float(data.get("households", 400.0)),
                    "median_income": float(data.get("median_income", 4.0)),
                    "ocean_proximity": str(data.get("ocean_proximity", "<1H OCEAN"))
                }])

                pred_raw = float(MODEL.predict(df)[0])
                pred_capped = min(max(pred_raw, 14999.0), 500001.0)

                # Derive diagnostic metrics
                rooms = df["total_rooms"].iloc[0]
                bedrooms = df["total_bedrooms"].iloc[0]
                pop = df["population"].iloc[0]
                hh = max(df["households"].iloc[0], 1.0)

                rooms_per_hh = round(rooms / hh, 2)
                bedrooms_ratio = round(bedrooms / max(rooms, 1.0), 3)
                pop_per_hh = round(pop / hh, 2)

                # Find closest hub
                lat, lon = df["latitude"].iloc[0], df["longitude"].iloc[0]
                closest_hub = None
                min_dist_deg = 99999.0
                for hub_name, (hlat, hlon) in HUBS.items():
                    d = ((lat - hlat)**2 + (lon - hlon)**2)**0.5
                    if d < min_dist_deg:
                        min_dist_deg = d
                        closest_hub = hub_name
                # Approx km (1 deg ~ 111 km)
                closest_km = round(min_dist_deg * 111.0, 1)

                # Error band from test MAE ($26,216)
                ci_low = max(pred_raw - 26216.0, 14999.0)
                ci_high = min(pred_raw + 26216.0, 500001.0)

                response = {
                    "predicted_value": round(pred_raw, 2),
                    "predicted_formatted": f"${pred_raw:,.0f}",
                    "confidence_band": {
                        "lower": round(ci_low, 2),
                        "upper": round(ci_high, 2),
                        "lower_formatted": f"${ci_low:,.0f}",
                        "upper_formatted": f"${ci_high:,.0f}",
                        "mae_benchmark": 26216.0
                    },
                    "features_derived": {
                        "rooms_per_household": rooms_per_hh,
                        "bedrooms_ratio": bedrooms_ratio,
                        "population_per_household": pop_per_hh,
                        "closest_urban_hub": closest_hub,
                        "distance_to_hub_km": closest_km
                    },
                    "model_used": "XGBoost Regressor (Test RMSE: $40,591 | R²: 0.874)"
                }
                self.send_json(response)
            except Exception as e:
                self.send_json({"error": f"Inference error: {str(e)}"}, status=400)
            return

        self.send_error(404, f"API endpoint not found: {self.path}")

    def send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, filepath, content_type):
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")


def run_server():
    load_resources()
    server_address = ("127.0.0.1", PORT)
    httpd = HTTPServer(server_address, DashboardHandler)
    print(f"\n===========================================================")
    print(f" [Dashboard] California Housing Predictive Analysis Web App")
    print(f" [Dashboard] URL: http://127.0.0.1:{PORT}")
    print(f" [Dashboard] API: http://127.0.0.1:{PORT}/api/predict")
    print(f"===========================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[Dashboard] Server stopped.")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
