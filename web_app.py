from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
model_kwh = joblib.load('xgboost_model_kwh.pkl')  # Load your pre-trained model 
model_thermal = joblib.load('xgboost_model_therm.pkl') 

@app.route('/')
def index():
    # Shared features between KWH and Thermal
    # Shared features between KWH and Thermal
    shared_features = ['Household Size','Adults In Household','Commute Type','Annual Household Income',	'Housing Type',	'Home Square Footage',	'Building Age (Years)',	'Number Of Stories	Heating', 'Community']
    # Unique features for KWH and Thermal
    unique_features_kwh = []  # Add any unique features for KWH here
    unique_features_thermal = []  # Add any unique features for Thermal here
    # Combine shared and unique features for KWH and Thermal
    features_kwh = shared_features + unique_features_kwh
    features_thermal = shared_features + unique_features_thermal
    # Remove duplicates and pass to the template
    all_features = list(dict.fromkeys(features_kwh + features_thermal))
    return render_template('index.html', features=all_features)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    numeric_data = {}
    for key in data.keys():
        try:
            numeric_data[key] = float(data[key])
        except ValueError:
            return jsonify({'error': f'Invalid input for {key}, please enter numeric values.'})

    df = pd.DataFrame([numeric_data])

    # Predict KWH and Thermal
    prediction_kwh = model_kwh.predict(df)[0]
    prediction_thermal = model_thermal.predict(df)[0]

    # Return predictions as JSON
    return jsonify({'prediction_kwh': prediction_kwh, 'prediction_thermal': prediction_thermal})

if __name__ == '__main__':
    app.run(debug=True)
