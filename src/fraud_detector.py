"""
Financial Fraud Detection Model
Simple fraud detection using machine learning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class FraudDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def generate_data(self, n_samples=1000):
        """Generate sample transaction data"""
        np.random.seed(42)
        
        # Legitimate transactions (95%)
        n_legitimate = int(n_samples * 0.95)
        legitimate = {
            'amount': np.random.normal(100, 30, n_legitimate),
            'hour': np.random.randint(8, 20, n_legitimate),
            'distance': np.random.normal(5, 2, n_legitimate),
            'online': np.random.choice([0, 1], n_legitimate, p=[0.7, 0.3])
        }
        
        # Fraudulent transactions (5%)
        n_fraud = n_samples - n_legitimate
        fraudulent = {
            'amount': np.random.normal(500, 150, n_fraud),
            'hour': np.random.choice([2, 3, 4, 22, 23], n_fraud),
            'distance': np.random.normal(50, 20, n_fraud),
            'online': np.random.choice([0, 1], n_fraud, p=[0.2, 0.8])
        }
        
        # Combine data
        data = []
        for i in range(n_legitimate):
            data.append([legitimate['amount'][i], legitimate['hour'][i], 
                        legitimate['distance'][i], legitimate['online'][i], 0])
        
        for i in range(n_fraud):
            data.append([fraudulent['amount'][i], fraudulent['hour'][i], 
                        fraudulent['distance'][i], fraudulent['online'][i], 1])
        
        df = pd.DataFrame(data, columns=['amount', 'hour', 'distance', 'online', 'fraud'])
        return df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    def train(self, data):
        """Train the fraud detection model"""
        X = data[['amount', 'hour', 'distance', 'online']]
        y = data['fraud']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        
        self.is_trained = True
        return accuracy
    
    def predict(self, transaction):
        """Predict if a transaction is fraudulent"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        features = np.array([[
            transaction['amount'],
            transaction['hour'],
            transaction['distance'],
            transaction['online']
        ]])
        
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0][1]
        
        return prediction, probability

def main():
    print("Starting Fraud Detection Model...")
    
    # Initialize detector
    detector = FraudDetector()
    
    # Generate sample data
    print("Generating transaction data...")
    data = detector.generate_data(1000)
    print(f"Dataset: {len(data)} transactions, {data['fraud'].sum()} fraudulent")
    
    # Train model
    print("Training model...")
    accuracy = detector.train(data)
    
    # Test predictions
    print("Testing predictions...")
    test_transaction = {
        'amount': 600,
        'hour': 3,
        'distance': 45,
        'online': 1
    }
    
    prediction, probability = detector.predict(test_transaction)
    result = "FRAUD" if prediction == 1 else "LEGITIMATE"
    print(f"Test Transaction: {result} (Probability: {probability:.2%})")
    
    print("Model training complete!")

if __name__ == "__main__":
    main()
