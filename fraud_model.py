import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

# Load the dataset
data = pd.read_csv('data/transaction_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Preprocessing: Handling missing values (fill with median or drop rows/columns)
data.fillna(data.median(), inplace=True)

# Feature selection: Assuming 'label' is the target column and others are features
X = data.drop(columns=['label'])  # Replace 'label' with actual target column name
y = data['label']  # Replace with the actual target column name

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (if necessary)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
joblib.dump(model, 'fraud_model.pkl')
print("Model saved as 'fraud_model.pkl'")
