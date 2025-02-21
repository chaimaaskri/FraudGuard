import pandas as pd

def feature_engineering(data):
    """
    Generate additional features for the model.
    """
    # Convert 'TransactionTime' to datetime object
    data['TransactionTime'] = pd.to_datetime(data['TransactionTime'])

    # Extract day of week and hour from the TransactionTime
    data['DayOfWeek'] = data['TransactionTime'].dt.dayofweek
    data['Hour'] = data['TransactionTime'].dt.hour

    # You can add more features if necessary
    return data
