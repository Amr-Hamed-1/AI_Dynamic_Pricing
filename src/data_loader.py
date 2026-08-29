import pandas as pd
import os

def load_data(file_path):
    """
    Function to load data from a CSV file.
    Args:
        file_path (str): The relative path to the CSV file.
    Returns:
        pd.DataFrame: The loaded dataframe, or None if failed.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: The file at '{file_path}' was not found.")
        print("   -> Make sure you put the CSV file in 'data/raw/' folder and named it 'retail_price.csv'.")
        return None

    try:
        df = pd.read_csv(file_path)
        print(f"✅ Success! Data loaded. Shape: {df.shape} (Rows, Columns)")
        return df
    
    except Exception as e:
        
        print(f"❌ Unexpected Error: {e}")
        return None

if __name__ == "__main__":
    path = "data/raw/retail_price.csv"
    
    df = load_data(path)
    
    if df is not None:
        print("\n--- First 5 Rows ---")
        print(df.head())