from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model_path = "KNN_model.pkl"  # Ensure the path is correct
model = joblib.load(model_path)

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Convert JSON data to a pandas DataFrame
        df = pd.DataFrame(data)

        # Make predictions
        predictions = model.predict(df)

        # Return predictions as JSON
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)