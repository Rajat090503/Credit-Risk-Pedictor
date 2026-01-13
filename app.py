from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

risk_model = joblib.load(os.path.join(BASE_DIR, "rf_risk_model.pkl"))
repayment_model = joblib.load(os.path.join(BASE_DIR, "repayment_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
scaler_reg = joblib.load(os.path.join(BASE_DIR, "scaler_reg.pkl"))


@app.route("/", methods=["GET", "POST"])
def index():
    risk = None
    months = None
    probability = None

    if request.method == "POST":
        age = int(request.form["age"])
        income = int(request.form["income"])
        employment = request.form["employment"]
        credit_limit = int(request.form["credit_limit"])
        spend = int(request.form["spend"])

        
        employment_encoded = 1 if employment == "Self-Employed" else 0

        
        credit_util = spend / credit_limit

        
        X = np.array([[age, income, credit_limit,
                       spend, credit_util, employment_encoded]])

        
        X_scaled = scaler.transform(X)
        risk_pred = risk_model.predict(X_scaled)[0]
        risk_prob = risk_model.predict_proba(X_scaled)[0][risk_pred]

        risk = "High" if risk_pred == 1 else "Low"
        probability = round(risk_prob * 100, 2)

        X_reg_scaled = scaler_reg.transform(X)
        months = round(repayment_model.predict(X_reg_scaled)[0])

    return render_template(
        "index.html",
        risk=risk,
        months=months,
        probability=probability
    )


@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
