import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def main(train_file, model_file):
    # Load training data
    df = pd.read_csv(train_file)

    # Assume "price" is the target
    X = df.drop(columns=["price"])
    y = df["price"]

    # Convert categorical features to numeric
    X = pd.get_dummies(X, drop_first=True)

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Save model
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    train_file = sys.argv[1]   # e.g. data/train.csv
    model_file = sys.argv[2]   # e.g. models/model.pkl
    main(train_file, model_file)
