from flask import Flask, render_template, jsonify, request
import pandas as pd

app = Flask(__name__)

# Load the energy data from a CSV file
df = pd.read_csv('output.csv')

@app.route('/')
def index():
    # Serve the main page with the energy profile visualization
    return render_template('index2.html')

@app.route('/api/energy_profiles')
def energy_profiles():
    # Convert the DataFrame to a JSON-friendly format
    data = df.to_dict(orient='records')
    return jsonify(data)

@app.route('/api/sustainability_index', methods=['POST'])
def sustainability_index():
    try:
        # Extract the features submitted by the user
        user_data = request.get_json()

        # Use the actual data to create the sustainability score formula
        # This is a placeholder formula and should be replaced with your actual logic
        total_kwh = df['TOTAL KWH'].mean()
        total_therms = df['TOTAL THERMS'].mean()

        # Example: Calculate a sustainability score based on user input and actual data
        user_kwh = user_data.get('TOTAL KWH', 0)
        user_therms = user_data.get('TOTAL THERMS', 0)

        # Simple example formula
        sustainability_score = max(0, 100 - ((user_kwh / total_kwh) + (user_therms / total_therms)) * 50)

        # Return the sustainability score as part of the response
        return jsonify({'sustainability_score': sustainability_score}), 200

    except Exception as e:
        # Handle any errors during the calculation
        return jsonify({'error': str(e)}), 400


@app.route('/submit_energy_data', methods=['POST'])
def submit_energy_data():
    try:
        # Extract the JSON data submitted by the user
        data = request.get_json()

        # Extract the gross rent range from the data
        gross_rent_range = data['gross-rent-range']

        # Map the gross rent range to specific values
        rent_values = {
            '200-299': 250,  # Midpoint of the $200 to $299 range
            '1500-more': 1500  # Minimum of the $1,500 or more range
        }
        gross_rent_value = rent_values.get(gross_rent_range, 0)

        # Perform calculations based on the provided features
        # Calculate an "Affordability Index" based on the gross rent value
        # This is a placeholder formula and should be replaced with your actual logic
        median_income = df['INCOME AND BENEFITS (IN 2010 INFLATION-ADJUSTED DOLLARS)__Median household income (dollars)'].median()
        affordability_index = median_income / gross_rent_value

        # Compile the response data
        response_data = {
            'Gross Rent Range': gross_rent_range,
            'Affordability Index': affordability_index
            # Include other calculated values as needed
        }

        # Send back the calculated data
        return jsonify(response_data), 200

    except Exception as e:
        # Handle any errors during data processing
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
