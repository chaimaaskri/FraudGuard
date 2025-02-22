import unittest
from fraud_detection import train_model
from preprocessing import load_data

class TestFraudDetection(unittest.TestCase):

    def setUp(self):
        """
        Load and preprocess the data before each test.
        """
        self.data = load_data('data/transaction_data.csv')
        self.model = train_model(self.data)

    def test_model_accuracy(self):
        """
        Test the model's accuracy to ensure it's above a reasonable threshold.
        """
        X = self.data[['CostPerItem', 'NumberOfItemsPurchased', 'DayOfWeek', 'Hour']]
        y = self.data['IsFraud']
        accuracy = self.model.score(X, y)
        self.assertGreater(accuracy, 0.7)  # Adjust threshold as necessary

if __name__ == '__main__':
    unittest.main()
