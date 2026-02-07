from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load files
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")


@app.route("/")
def home():
    return render_template("index.html", features=features)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form
    input_data = [float(data[f]) for f in features]

    arr = np.array(input_data).reshape(1, -1)
    arr_scaled = scaler.transform(arr)

    pred = model.predict(arr_scaled)[0]
    prob = model.predict_proba(arr_scaled)[0][1]

    result = "Fraud Transaction 🚨" if pred == 1 else "Normal Transaction ✅"

    return render_template(
        "index.html",
        features=features,
        prediction=result,
        probability=round(prob, 4)
    )


if __name__ == "__main__":
    app.run(debug=True)
