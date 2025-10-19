from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("../Shahul_Hameed_Project/models/flaky_model.pkl")

@app.route('/')
def home():
    return "👋 Welcome to Shahul's Flaky Test Predictor API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Required keys
    required_keys = ['TestName', 'FailureRate', 'DurationVariance', 'EnvVolatility', 'TimeOfDay']
    missing_keys = []
    for key in required_keys:
        if key not in data:
            missing_keys.append(key)

    if missing_keys:
        return jsonify({"error": f"Missing keys: {missing_keys}"}), 400

    # Encode categorical features manually (must match training logic)
    test_name = data['TestName']
    time_of_day = data['TimeOfDay']

    # Dummy encoders (replace with saved LabelEncoders if needed)
    test_name_encoded = hash(test_name) % 1000  # crude encoding
    time_of_day_map = {'Night': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3}
    time_of_day_encoded = time_of_day_map.get(time_of_day, -1)

    if time_of_day_encoded == -1:
        return jsonify({"error": f"Invalid TimeOfDay: {time_of_day}"}), 400

    # Build feature vector
    features = np.array([[
        test_name_encoded,
        data['FailureRate'],
        data['DurationVariance'],
        data['EnvVolatility'],
        time_of_day_encoded
    ]])

    # Predict
    prediction = model.predict(features)[0]
    label = "Flaky" if prediction == 1 else "Stable"

    return jsonify({
        "input": data,
        "prediction": label
    })

if __name__ == "__main__":
    app.run(debug=True)