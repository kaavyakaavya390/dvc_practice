import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def main(input_file, train_file, test_file):
    # Read the raw data
    df = pd.read_csv(input_file)
    #print(df)
    # Simple train/test split (80/20)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save outputs
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

if __name__ == "__main__":
    input_file = sys.argv[1]   # e.g. data/diamonds.csv
    train_file = sys.argv[2]   # e.g. data/train.csv
    test_file = sys.argv[3]    # e.g. data/test.csv
    main(input_file, train_file, test_file)
