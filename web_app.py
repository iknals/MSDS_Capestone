from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import folium

app = Flask(__name__)
model = joblib.load('best_xgb_model.pkl')  # Load your pre-trained model

@app.route('/')
def index():
    return render_template('index.html')  # The form for data input will be here

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form, assume form data format
    data = request.form.to_dict()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    
    # Generate map with color-coded markers based on prediction
    map = folium.Map(location=[45.5236, -122.6750], zoom_start=13)  # Use appropriate location
    folium.CircleMarker(
        location=[45.5236, -122.6750],  # Use actual location data
        radius=10,
        color='blue' if prediction[0] > threshold else 'red',  # Set threshold as needed
        fill=True,
        fill_color='blue' if prediction[0] > threshold else 'red'
    ).add_to(map)
    
    # Return map as HTML
    return map._repr_html_()

if __name__ == '__main__':
    app.run(debug=True)
