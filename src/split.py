import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def main(input_file, output_file):
    # Load CSV
    df = pd.read_csv(input_file)

    # Simple train/test split
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # Save train set (DVC will track this output)
    train.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = sys.argv[1]   # e.g. data/diamonds.csv
    output_file = sys.argv[2]  # e.g. data/data2.csv
    main(input_file, output_file)
