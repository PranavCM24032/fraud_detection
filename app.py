from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "final_fraud_detection_model.pkl")

@app.route("/")
def home():
    return send_file(BASE_DIR / "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json(silent=True) or {}
        X = pd.DataFrame([{
            "Amount": float(d["Amount"]),
            "Merchant_Category": d["Merchant_Category"],
            "Distance_from_Home": float(d["Distance_from_Home"]),
            "Device_Type": d["Device_Type"],
            "IP_Risk_Score": float(d["IP_Risk_Score"]),
            "Avg_Spending_Habit": float(d["Avg_Spending_Habit"]),
            "Is_Weekend": int(d["Is_Weekend"]),
            "Is_Night_Transaction": int(d["Is_Night_Transaction"]),
        }])
    except (KeyError, TypeError, ValueError):
        return jsonify({"success": False, "error": "Invalid or incomplete transaction data."}), 400

    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])
    return jsonify({
        "success": True,
        "prediction": pred,
        "probability": prob,
        "label": "FRAUD" if pred == 1 else "GENUINE"
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)