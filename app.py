from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

risk_map = {0: "Low", 1: "Medium", 2: "High"}

@app.route("/")
def home():
    return jsonify({"message": "Smart Health Risk API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    age = data["age"]
    bmi = data["bmi"]
    bp = data["bp"]
    glucose = data["glucose"]
    smoking = data["smoking"]

    features = np.array([[age, bmi, bp, glucose, smoking]])
    features = scaler.transform(features)

    prediction = model.predict(features)[0]

    return jsonify({
        "risk_level": risk_map[prediction]
    })

if __name__ == "__main__":
    app.run(debug=True)