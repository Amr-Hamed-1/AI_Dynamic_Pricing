import pandas as pd    
import numpy as np
import sys
import os



# Load the dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import load_data


class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = load_data(file_path)

    def preprocess(self):
        df = self.df.copy()

        # Drop unnecessary columns
        drop_columns = ['product_category_name','month_year']
        df.drop(columns=drop_columns, inplace=True, errors='ignore')


        # Handle Missing Entries
        comp_cols = ['comp_1', 'comp_2', 'comp_3']
        for col in comp_cols:
            if col in self.df.columns:
                df[col] =df[col].fillna(df['unit_price'])

        df.dropna(inplace=True)
        self.df = df
        return df
    

    def feature_engineering(self):
        df = self.df.copy()

        # Create New Features
        df['price_diff_1'] = df['unit_price'] - df['comp_1']
        df['price_diff_2'] = df['unit_price'] - df['comp_2']
        df['price_diff_3'] = df['unit_price'] - df['comp_3']

        df['revenue'] = df['unit_price'] * df['qty']
        self.df = df
        return df
    
    def save_processed(self, output_path):
        
        self.df.to_csv(output_path, index=False)
        print(f"✅ Processed data saved to {output_path}")  

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(root, "data", "raw", "retail_price.csv")
    output_path = os.path.join(root, "data", "processed", "processed_retail_price.csv")

    preprocessor = DataPreprocessor(input_path)
    
    if preprocessor.df is not None:
        preprocessor.preprocess()
        preprocessor.feature_engineering()
        preprocessor.save_processed(output_path)




        print("\n--- First 5 Rows of Processed Data ---")
        print(preprocessor.df.head())
        print(f"\n✅ Final Processed Data Shape: {preprocessor.df.shape} (Rows, Columns)")
        

        
    
