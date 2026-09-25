import json
import os
from pathlib import Path

import numpy as np
import xgboost as xgb
from flask import Flask, jsonify, request, send_file

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

with (BASE_DIR / "model_config.json").open(encoding="utf-8") as config_file:
    model_config = json.load(config_file)

model = xgb.Booster()
model.load_model(str(BASE_DIR / "final_fraud_detection_model.json"))

numeric_features = model_config["numeric_features"]
numeric_min = np.asarray(model_config["numeric_min"], dtype=np.float64)
numeric_scale = np.asarray(model_config["numeric_scale"], dtype=np.float64)
merchant_categories = model_config["merchant_categories"]
device_categories = model_config["device_categories"]


def build_features(data):
    numeric = np.asarray([float(data[name]) for name in numeric_features], dtype=np.float64)
    numeric = (numeric - numeric_min) * numeric_scale
    merchant = [float(data["Merchant_Category"] == category) for category in merchant_categories]
    device = [float(data["Device_Type"] == category) for category in device_categories]
    return np.concatenate((numeric, np.asarray(merchant + device, dtype=np.float64))).astype(np.float32).reshape(1, -1)


@app.route("/")
def home():
    return send_file(BASE_DIR / "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        features = build_features(data)
    except (KeyError, TypeError, ValueError):
        return jsonify({"success": False, "error": "Invalid or incomplete transaction data."}), 400

    probability = float(model.predict(xgb.DMatrix(features))[0])
    prediction = int(probability >= 0.5)
    return jsonify({
        "success": True,
        "prediction": prediction,
        "probability": probability,
        "label": "FRAUD" if prediction == 1 else "GENUINE"
    })


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
