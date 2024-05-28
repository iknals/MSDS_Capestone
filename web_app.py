from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
model_kwh = joblib.load('xgboost_model_kwh.pkl')  # Load your pre-trained model 
model_thermal = joblib.load('xgboost_model_therm.pkl') 

# Load your dataset here
dataset = pd.read_csv('acs5yr2010_cook(in).csv')

@app.route('/')
def index():
    # Pass the feature lists to the template
    features_kwh = ['Percent!!SEX AND AGE!!21 years and over', 'Percent!!RACE!!Native Hawaiian and Other Pacific Islander']  # Add all your features here
    features_thermal = ['Percent!!SEX AND AGE!!15 to 19 years', 'Estimate!!SEX AND AGE!!Median age (years)']  # Add all your features here
    return render_template('index.html', features_kwh=features_kwh, features_thermal=features_thermal)

@app.route('/predict1', methods=['POST'])
def predict_kwh():
    data = request.form.to_dict()
    numeric_data = {}
    for key in data.keys():
        try:
            numeric_data[key] = float(data[key])
        except ValueError:
            return jsonify({'error': f'Invalid input for {key}, please enter numeric values.'})

    df = pd.DataFrame([numeric_data])

    # Fill missing features with mean values from the dataset
    for column in dataset.columns:
        if column not in df.columns:
            df[column] = dataset[column].mean()

    prediction = model_kwh.predict(df)
    return jsonify({'prediction': prediction[0]})

@app.route('/predict2', methods=['POST'])
def predict_therm():
    data = request.form.to_dict()
    numeric_data = {}
    for key in data.keys():
        try:
            numeric_data[key] = float(data[key])
        except ValueError:
            return jsonify({'error': f'Invalid input for {key}, please enter numeric values.'})

    df = pd.DataFrame([numeric_data])

    # Fill missing features with mean values from the dataset
    for column in dataset.columns:
        if column not in df.columns:
            df[column] = dataset[column].mean()

    prediction = model_thermal.predict(df)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
