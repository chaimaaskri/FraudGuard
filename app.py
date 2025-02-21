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
    
    # Convert the input data to a DataFrame
    transaction_df = pd.DataFrame([transaction])

    # Feature engineering (convert the transaction data into features usable by the model)
    transaction_df = feature_engineering(transaction_df)

    # Make prediction
    prediction = model.predict(transaction_df[['CostPerItem', 'NumberOfItemsPurchased', 'DayOfWeek', 'Hour']])

    # Return the prediction (fraud prediction)
    return jsonify({'fraud_prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
