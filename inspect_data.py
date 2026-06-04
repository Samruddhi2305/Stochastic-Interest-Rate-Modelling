import pandas as pd

def inspect_file(filepath):
    print(f"\n--- Inspecting {filepath} ---")
    df = pd.read_csv(filepath)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Head:\n", df.head())
    print("Tail:\n", df.tail())
    print("Null values:\n", df.isnull().sum())
    print("Types:\n", df.dtypes)

if __name__ == '__main__':
    inspect_file('data/train_data.csv')
    inspect_file('data/test_data.csv')
    inspect_file('data/test_data_3M.csv')
