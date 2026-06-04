import pandas as pd
import numpy as np

def explore():
    train = pd.read_csv('data/train_data.csv')
    test = pd.read_csv('data/test_data.csv')
    test_3M = pd.read_csv('data/test_data_3M.csv')
    
    # Strip whitespace from column names
    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()
    test_3M.columns = test_3M.columns.str.strip()
    
    print("--- Test Data Missing Values ---")
    print("test_data:", test.isnull().sum().to_dict())
    print("test_data_3M:", test_3M.isnull().sum().to_dict())
    
    # Check for negative values
    print("\n--- Negative Values ---")
    for df_name, df in [('train', train), ('test', test), ('test_3M', test_3M)]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        neg_counts = (df[numeric_cols] < 0).sum().to_dict()
        print(f"{df_name}: {neg_counts}")
        
    # Check for sudden daily changes (outliers)
    print("\n--- Daily Volatility Spikes (Diff > 3 * Std of Diff) ---")
    for df_name, df in [('train', train), ('test', test)]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        diffs = df[numeric_cols].diff().abs()
        std_diffs = diffs.std()
        mean_diffs = diffs.mean()
        spikes = (diffs > (mean_diffs + 4 * std_diffs)).sum().to_dict()
        print(f"{df_name} spikes count (> mean + 4*std): {spikes}")

    # Check date continuity
    train['Date'] = pd.to_datetime(train['Date'])
    test['Date'] = pd.to_datetime(test['Date'])
    
    print("\n--- Date Range and Spacing ---")
    print(f"Train Date Range: {train['Date'].min()} to {train['Date'].max()}")
    print(f"Test Date Range: {test['Date'].min()} to {test['Date'].max()}")
    
    train_diffs = train['Date'].diff().dt.days
    test_diffs = test['Date'].diff().dt.days
    
    print(f"Train date gap stats (days):")
    print(train_diffs.describe())
    print(f"Test date gap stats (days):")
    print(test_diffs.describe())
    
    # Gaps larger than 4 days (e.g. holidays or data gaps)
    print(f"Train gaps > 4 days: {train_diffs[train_diffs > 4].tolist()}")
    print(f"Test gaps > 4 days: {test_diffs[test_diffs > 4].tolist()}")

if __name__ == '__main__':
    explore()
