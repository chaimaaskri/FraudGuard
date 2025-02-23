import pandas as pd
from datetime import datetime

# Function to load and clean the data
def load_and_clean_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    
    # Convert 'TransactionTime' to datetime format
    data['TransactionTime'] = pd.to_datetime(data['TransactionTime'], errors='coerce')
    
    # Extract day of the week (0=Monday, 1=Tuesday, ..., 6=Sunday)
    data['DayOfWeek'] = data['TransactionTime'].dt.dayofweek
    
    # Extract the hour from the transaction time (0-23)
    data['Hour'] = data['TransactionTime'].dt.hour
    
    # Drop rows where 'TransactionTime' could not be parsed
    data = data.dropna(subset=['TransactionTime'])
    
    return data

# Function to train the model
def train_model(data):
    # Clean the data using the cleaning function
    cleaned_data = load_and_clean_data(data)

    # Use the cleaned data to create features and labels for training
    X = cleaned_data[['CostPerItem', 'NumberOfItemsPurchased', 'DayOfWeek', 'Hour']]  # Features
    y = cleaned_data['Country']  # This can be replaced with a fraud label if you have one
    
    # Continue with model training (e.g., Random Forest, etc.)
    # Add your model code here...

    return model

