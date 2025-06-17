from flask import Flask, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def result():
    try:
        item_weight = float(request.form['item_weight'])
        item_fat_content = float(request.form['item_fat_content'])
        item_visibility = float(request.form['item_visibility'])
        item_type = float(request.form['item_type'])
        item_mrp = float(request.form['item_mrp'])
        outlet_establishment_year = float(request.form['outlet_establishment_year'])
        outlet_size = float(request.form['outlet_size'])
        outlet_location_type = float(request.form['outlet_location_type'])
        outlet_type = float(request.form['outlet_type'])

        # Extra Features (added in your form)
        extra_feature_1 = float(request.form['extra_feature_1'])
        extra_feature_2 = float(request.form['extra_feature_2'])

        # Prepare the input array with new features
        X = np.array([[item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                    outlet_establishment_year, outlet_size, outlet_location_type, outlet_type,
                    extra_feature_1, extra_feature_2]])

        # Load scaler
        scaler_path = os.path.join('models', 'sc.sav')
        sc = joblib.load(scaler_path)
        X_std = sc.transform(X)

        # Load model
        model_path = os.path.join('models', 'lr.sav')
        model = joblib.load(model_path)
        Y_pred = model.predict(X_std)

        # Render prediction in the same HTML
        return render_template("home.html", prediction=round(float(Y_pred[0]), 2))

    except Exception as e:
        print(f"Error occurred: {e}")
        return render_template("home.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True, port=9457)



