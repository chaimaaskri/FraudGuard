import pandas as pd

def process_features(data):
    # Remove unrecognized timezone (IST, UTC, etc.)
    data['TransactionTime'] = data['TransactionTime'].str.replace(r' [A-Z]{3,4} ', ' ', regex=True)
    
    # Convert to datetime
    data['TransactionTime'] = pd.to_datetime(data['TransactionTime'], errors='coerce')

    # Extract features
    data['DayOfWeek'] = data['TransactionTime'].dt.dayofweek
    data['Hour'] = data['TransactionTime'].dt.hour

    # Drop rows where 'TransactionTime' is NaT
    data = data.dropna(subset=['TransactionTime'])

    return data

