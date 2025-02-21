import pandas as pd

def load_data(file_path='data/transaction_data.csv'):
    """
    Load the transaction data from CSV.
    """
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """
    Clean the data by removing any rows with missing values.
    """
    data = data.dropna()
    return data
