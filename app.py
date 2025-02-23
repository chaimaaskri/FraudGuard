from flask import Flask, jsonify, request
from preprocessing import load_data, clean_data
from feature_engineering import feature_engineering
from fraud_detection import train_model
import pandas as pd

app = Flask(__name__)

# Load and clean data
data = load_data('data/transaction_data.csv')
cleaned_data = clean_data(data)

# Train fraud detection model
model = train_model(cleaned_data)

@app.route('/predict-fraud', methods=['POST'])
def predict_fraud():
    """
    Predict fraud for a given transaction based on the trained model.
    """
    # Get transaction data from the request body (JSON format)
    transaction = request.json  # Assuming input as JSON
    
    # Convert the input data to a DataFramefrom flask import Flask, jsonify, request
from preprocessing import load_data, clean_data
from feature_engineering import feature_engineering
from fraud_detection import train_model
import pandas as pd

app = Flask(__name__)

# Load, clean, and process data
data = load_data('data/transaction_data.csv')
cleaned_data = clean_data(data)
processed_data = feature_engineering(cleaned_data)  # APPLY FEATURE ENGINEERING HERE

# Train fraud detection model
model = train_model(processed_data)  # Use the processed data

@app.route('/predict-fraud', methods=['POST'])
def predict_fraud():
    """
    Predict fraud for a given transaction based on the trained model.
    """
    transaction = request.json
    transaction_df = pd.DataFrame([transaction])

    # Feature engineering on incoming data
    transaction_df = feature_engineering(transaction_df)

    prediction = model.predict(transaction_df[['CostPerItem', 'NumberOfItemsPurchased', 'DayOfWeek', 'Hour']])

    return jsonify({'fraud_prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
