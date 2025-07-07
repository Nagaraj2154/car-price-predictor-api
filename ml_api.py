from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load("models/best_car_price_model.pkl")

# Mappings
map_fuel = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
map_seller = {'Dealer': 0, 'Trustmark Dealer': 1, 'Individual': 2}
map_transmission = {'Manual': 0, 'Automatic': 1}
map_owner = {
    'First Owner': 0,
    'Second Owner': 1,
    'Third Owner': 2,
    'Fourth Owner': 2,
    'Fourth & Above Owner': 2
}
map_yes_no = {'No': 0, 'Yes': 1}
map_service = {'Incomplete': 0, 'Partial': 1, 'Complete': 2}

@app.route('/')
def home():
    return "✅ Flask ML API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("🔵 /predict hit")

        data = request.get_json(force=True)
        print("🟢 Raw incoming data:", data)

        required_keys = [
            'year', 'km_driven', 'present_price', 'fuel_type',
            'seller_type', 'transmission', 'owner',
            'service_cost', 'modifications', 'accidents',
            'insurance_valid', 'service_history'
        ]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Missing key: {key}")

        input_data = pd.DataFrame([{
            'year': data['year'],
            'km_driven': data['km_driven'],
            'present_price': data['present_price'],
            'fuel_type': map_fuel.get(data['fuel_type'], 0),
            'seller_type': map_seller.get(data['seller_type'], 2),
            'transmission': map_transmission.get(data['transmission'], 0),
            'owner': map_owner.get(data['owner'], 0),
            'service_cost': data['service_cost'],
            'modifications': map_yes_no.get(data['modifications'], 0),
            'accidents': map_yes_no.get(data['accidents'], 0),
            'insurance_valid': map_yes_no.get(data['insurance_valid'], 0),
            'service_history': map_service.get(data['service_history'], 2)
        }])

        print("✅ Input DataFrame:\n", input_data)

        # ✅ New model predicts depreciation ratio directly
        predicted_ratio = model.predict(input_data)[0]

        # ✅ Final resale price = present price × ratio
        present_price = data['present_price']
        predicted_price = present_price * predicted_ratio

        return jsonify({
            'predicted_price_lakhs': float(round(predicted_price, 2)),
            'depreciation_ratio': float(round(predicted_ratio, 3))
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)
