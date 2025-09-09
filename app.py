# app.py

from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

# ---------------------------
# 1. Load Model & Preprocessors
# ---------------------------
model = joblib.load("house_price_model.joblib")
scaler = joblib.load("scaler.joblib")
encoder = joblib.load("encoder.joblib")

app = Flask(__name__)

# ---------------------------
# 2. Home Route
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")  # we’ll create index.html for input form

# ---------------------------
# 3. Prediction Route
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get form data
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        stories = int(request.form["stories"])
        mainroad = request.form["mainroad"]
        guestroom = request.form["guestroom"]
        basement = request.form["basement"]
        hotwaterheating = request.form["hotwaterheating"]
        airconditioning = request.form["airconditioning"]
        parking = int(request.form["parking"])
        prefarea = request.form["prefarea"]
        furnishingstatus = request.form["furnishingstatus"]

        # Create DataFrame with same structure as training
        input_df = pd.DataFrame([{
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "parking": parking,
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus,
            # ✅ New feature
            "price_per_area": 0  # placeholder, calculated below
        }])

        # Add feature engineering step
        input_df["price_per_area"] = input_df["area"]  # dummy, since price not known

        # Apply same transformations
        X_encoded = encoder.transform(input_df)
        X_scaled = scaler.transform(X_encoded)

        # Prediction
        prediction = model.predict(X_scaled)[0]

        return render_template("result.html", prediction_text=f"🏠 Estimated House Price: ₹{round(prediction, 2)}")

# ---------------------------
# 4. Run App
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
