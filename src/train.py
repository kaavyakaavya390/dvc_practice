import sys
import pandas as pd
import pickle

def main(train_file, model_file):
    df = pd.read_csv(train_file)

    # "Fake training": just compute average of first numeric column
    numeric_col = df.select_dtypes(include="number").columns[0]
    avg_value = df[numeric_col].mean()

    # Save as a "model"
    with open(model_file, "wb") as f:
        pickle.dump({"avg_"+numeric_col: avg_value}, f)

if __name__ == "__main__":
    train_file = sys.argv[1]  # e.g. data/data2.csv
    model_file = sys.argv[2]  # e.g. models/model.pkl
    main(train_file, model_file)
