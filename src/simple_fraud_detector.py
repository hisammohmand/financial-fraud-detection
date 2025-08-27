"""
Simple Financial Fraud Detection
Basic fraud detection demonstration
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generate_data(n_samples=1000):
    """Generate sample transaction data"""
    np.random.seed(42)
    
    # Create legitimate transactions (95%)
    n_legitimate = int(n_samples * 0.95)
    legitimate_data = []
    
    for i in range(n_legitimate):
        legitimate_data.append([
            np.random.normal(100, 30),  # amount
            np.random.randint(8, 20),   # hour
            np.random.normal(5, 2),     # distance
            np.random.choice([0, 1], p=[0.7, 0.3]),  # online
            0  # fraud label
        ])
    
    # Create fraudulent transactions (5%)
    n_fraud = n_samples - n_legitimate
    fraud_data = []
    
    for i in range(n_fraud):
        fraud_data.append([
            np.random.normal(500, 150),  # amount
            np.random.choice([2, 3, 4, 22, 23]),  # hour
            np.random.normal(50, 20),    # distance
            np.random.choice([0, 1], p=[0.2, 0.8]),  # online
            1  # fraud label
        ])
    
    # Combine and shuffle
    all_data = legitimate_data + fraud_data
    np.random.shuffle(all_data)
    
    df = pd.DataFrame(all_data, columns=['amount', 'hour', 'distance', 'online', 'fraud'])
    return df

def train_model(data):
    """Train fraud detection model"""
    X = data[['amount', 'hour', 'distance', 'online']]
    y = data['fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Model Accuracy: {:.4f}".format(accuracy))
    return model

def predict_fraud(model, transaction):
    """Predict if transaction is fraudulent"""
    features = np.array([[
        transaction['amount'],
        transaction['hour'],
        transaction['distance'],
        transaction['online']
    ]])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return prediction, probability

def main():
    print("Financial Fraud Detection Demo")
    print("=" * 40)
    
    # Generate data
    print("Generating transaction data...")
    data = generate_data(1000)
    print("Dataset: {} transactions, {} fraudulent".format(len(data), data['fraud'].sum()))
    
    # Train model
    print("Training fraud detection model...")
    model = train_model(data)
    
    # Test predictions
    print("Testing fraud detection...")
    
    test_cases = [
        {'amount': 50, 'hour': 14, 'distance': 5, 'online': 0},
        {'amount': 600, 'hour': 3, 'distance': 45, 'online': 1},
        {'amount': 1200, 'hour': 23, 'distance': 100, 'online': 1}
    ]
    
    for i, transaction in enumerate(test_cases, 1):
        prediction, probability = predict_fraud(model, transaction)
        result = "FRAUD" if prediction == 1 else "LEGITIMATE"
        
        print("Test {}: Amount=${}, Hour={}, Distance={}, Online={} -> {} ({:.1%})".format(
            i, transaction['amount'], transaction['hour'], 
            transaction['distance'], transaction['online'], result, probability
        ))
    
    print("Demo complete!")

if __name__ == "__main__":
    main()
