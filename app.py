from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import json
import numpy as np
import pandas as pd
import math # math library ko import karein

app = Flask(__name__)
CORS(app)

# Load artifacts
print("Loading saved artifacts...")
clf = joblib.load("model_xgb.pkl")
iso = joblib.load("model_iso.pkl")
scaler = joblib.load("scaler.pkl")
with open("config.json") as f:
    cfg = json.load(f)
print("Artifacts loaded successfully.")

V_FEATURES_MEANS = np.zeros(28)
V_FEATURES_STDS = np.ones(28) * 0.5

def featurize(tx):
    amount = float(tx.get("amount", 0))
    scaled_amount = scaler.transform([[amount]])[0][0]
    is_simulated_fraud = amount > 10000
    if is_simulated_fraud:
        v_features = np.random.normal(V_FEATURES_MEANS - 0.2, V_FEATURES_STDS, 28)
    else:
        v_features = np.random.normal(V_FEATURES_MEANS, V_FEATURES_STDS, 28)
    features_list = list(v_features) + [scaled_amount]
    return np.array(features_list).reshape(1, -1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    tx = request.json
    amount = float(tx.get("amount", 0))

    # ==================== NEW SANITY CHECK BLOCK ====================
    # Agar amount unbelievable hai ya infinity hai, to model ko bypass karke direct hold karo.
    # Ek practical limit set karte hain, jaise 1 Crore (10,000,000).
    PLAUSIBLE_AMOUNT_LIMIT = 10000000 
    
    if amount > PLAUSIBLE_AMOUNT_LIMIT or math.isinf(amount):
        return jsonify({
            "decision": "HOLD & REVIEW",
            "risk_score": 1.0,
            "probability_of_fraud": 1.0,
            "anomaly_score": 1.0,
            "reasons": ["Transaction Amount Exceeds Plausible Limit"]
        })
    # ==================== END OF SANITY CHECK BLOCK ====================

    # Baaki ka code waisa hi rahega
    x = featurize(tx)
    
    prob = float(clf.predict_proba(x)[:, 1])
    iso_score = float((-iso.decision_function(x))[0])
    
    risk = 0.8 * prob + 0.2 * iso_score
    
    if risk < cfg["T1"]:
        decision = "APPROVE"
    elif risk < cfg["T2"]:
        decision = "STEP-UP (OTP/KYC)"
    else:
        decision = "HOLD & REVIEW"
        
    top_reasons = []
    if prob > 0.5:
        top_reasons.append(f"High Fraud Probability: {prob:.2f}")
    if iso_score > 0.1:
        top_reasons.append(f"Anomalous Transaction Pattern Detected")
    if amount > 5000:
        top_reasons.append("High Transaction Amount")
    
    if not top_reasons:
        top_reasons.append("Low Risk Profile")

    return jsonify({
        "decision": decision,
        "risk_score": round(risk, 3),
        "probability_of_fraud": round(prob, 3),
        "anomaly_score": round(iso_score, 3),
        "reasons": top_reasons
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)