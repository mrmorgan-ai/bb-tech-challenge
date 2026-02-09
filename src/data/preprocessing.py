import json
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import (
    TARGET_COL, ID_COL, DATE_COLS, BINARY_COLS, 
    CATEGORICAL_COLS, TEST_SIZE, VAL_SIZE, 
    RANDOM_STATE
)


class MLDataPreProcessor:
    def _init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder()
        
    def load_raw_data(self, path: str)->pd.DataFrame:
        """
        Load raw data from xlsx path
        """
        if not path:
            raise FileNotFoundError(f"Can no upload non-existing file")
        
        df = pd.read_excel(path, sheet_name='data', header=0, engine='openpyxl')
        print(f"Loaded {len(df)} rows and {df.shape[1]} columns")
        return df
    
    def clean_data(self, df: pd.DataFrame)->pd.DataFrame:
        """
        - Drop rows with missing IDs
        - Parse dates
        - Remove identifier column
        """
        df_clean = df.copy()
        
        null_id_count = df_clean.loc[df_clean[ID_COL].isnull(),:].shape

        print(f"Ammount of nulls is Id column: {null_id_count} rows")
        # Drop rows where custid is null
        df_clean = df_clean.dropna(subset=[ID_COL]).reset_index(drop=True)
        
        
        # Parse date columns
        for col in DATE_COLS:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            
        print(f"Sahpe after cleaninf df: {df_clean.shape[0]} rows")
        return df_clean
    
    