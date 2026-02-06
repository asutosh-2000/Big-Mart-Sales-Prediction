from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["item_weight"]),
            float(request.form["item_fat_content"]),
            float(request.form["item_visibility"]),
            float(request.form["item_type"]),
            float(request.form["item_mrp"]),
            float(request.form["outlet_establishment_year"]),
            float(request.form["outlet_size"]),
            float(request.form["outlet_location_type"]),
            float(request.form["outlet_type"]),
            float(request.form["extra_feature_1"]),
            float(request.form["extra_feature_2"]),
        ]

        X = np.array([features])

        scaler = joblib.load(os.path.join("models", "sc.sav"))
        model = joblib.load(os.path.join("models", "lr.sav"))

        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)

        # 🔥 BULLETPROOF FIX
        prediction_value = prediction.ravel()[0]

        return render_template(
            "home.html",
            prediction=round(float(prediction_value), 2)
        )

    except Exception as e:
        print("Error occurred:", e)
        return render_template("home.html", prediction=None)


if __name__ == "__main__":
    app.run(debug=True, port=9457)
