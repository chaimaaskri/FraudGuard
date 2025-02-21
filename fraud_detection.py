from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def train_model(data):
    """
    Train a random forest model for fraud detection.
    """
    # Assuming 'IsFraud' column exists in the data for fraud labels (1 for fraud, 0 for non-fraud)
    # Replace it with your actual fraud detection logic
    X = data[['CostPerItem', 'NumberOfItemsPurchased', 'DayOfWeek', 'Hour']]  # Features
    y = data['IsFraud']  # Target label

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f'Model accuracy: {accuracy * 100:.2f}%')

    return model
