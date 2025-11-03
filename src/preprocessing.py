"""
Data Preprocessing Pipeline
Handles data cleaning, missing values, and type conversions
Following Kaggle Grandmaster best practices
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import (CATEGORICAL_FEATURES, NUMERICAL_FEATURES, 
                        TARGET, PROCESSED_DATA_DIR)

class ChurnPreprocessor:
    """
    Preprocessing pipeline for churn data
    Handles missing values, outliers, and data type conversions
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def clean_total_charges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean TotalCharges column (has some whitespace issues in raw data)
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        df = df.copy()
        
        # Convert TotalCharges to numeric, coercing errors
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Fill missing TotalCharges with MonthlyCharges * tenure
        # (logical imputation for customers with 0 tenure)
        mask = df['TotalCharges'].isnull()
        if mask.sum() > 0:
            df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges'] * df.loc[mask, 'tenure']
            print(f"✅ Imputed {mask.sum()} missing TotalCharges values")
        
        return df
    
    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode target variable to binary (0/1)
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with encoded target
        """
        df = df.copy()
        
        if df[TARGET].dtype == 'object':
            df[TARGET] = (df[TARGET] == 'Yes').astype(int)
            print(f"✅ Target encoded: No=0, Yes=1")
            print(f"   Churn rate: {df[TARGET].mean():.2%}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle any remaining missing values
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with no missing values
        """
        df = df.copy()
        
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"⚠️ Missing values found:")
            print(missing_counts[missing_counts > 0])
            
            # Fill numerical with median
            for col in NUMERICAL_FEATURES:
                if col in df.columns and df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
            
            # Fill categorical with mode
            for col in CATEGORICAL_FEATURES:
                if col in df.columns and df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        print(f"✅ No missing values: {df.isnull().sum().sum() == 0}")
        return df
    
    def remove_customer_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove customerID column (not useful for prediction)
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame without customerID
        """
        df = df.copy()
        
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
            print("✅ Removed customerID column")
        
        return df
    
    def fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure correct data types for all columns
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with correct types
        """
        df = df.copy()
        
        # Ensure SeniorCitizen is categorical (currently 0/1)
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
        
        # Ensure numerical features are numeric
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure categorical features are strings
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        print("✅ Data types fixed")
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessor and transform data (for training set)
        
        Args:
            df: Training dataframe
            
        Returns:
            Preprocessed dataframe
        """
        print("\n🔧 Fitting and transforming training data...")
        
        df = self.remove_customer_id(df)
        df = self.clean_total_charges(df)
        df = self.encode_target(df)
        df = self.fix_data_types(df)
        df = self.handle_missing_values(df)
        
        print("✅ Training data preprocessing complete!")
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor (for test set)
        
        Args:
            df: Test dataframe
            
        Returns:
            Preprocessed dataframe
        """
        print("\n🔧 Transforming test data...")
        
        df = self.remove_customer_id(df)
        df = self.clean_total_charges(df)
        if TARGET in df.columns:
            df = self.encode_target(df)
        df = self.fix_data_types(df)
        df = self.handle_missing_values(df)
        
        print("✅ Test data preprocessing complete!")
        return df

def preprocess_data(train_df: pd.DataFrame, 
                   test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess both train and test data
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        
    Returns:
        Preprocessed train and test dataframes
    """
    preprocessor = ChurnPreprocessor()
    
    train_processed = preprocessor.fit_transform(train_df)
    test_processed = preprocessor.transform(test_df)
    
    return train_processed, test_processed

# Main execution
if __name__ == "__main__":
    print("🚀 Starting preprocessing pipeline...")
    
    # Load data
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    
    print(f"📊 Original train shape: {train_df.shape}")
    print(f"📊 Original test shape: {test_df.shape}")
    
    # Preprocess
    train_processed, test_processed = preprocess_data(train_df, test_df)
    
    # Save preprocessed data
    train_processed.to_csv(PROCESSED_DATA_DIR / "train_preprocessed.csv", index=False)
    test_processed.to_csv(PROCESSED_DATA_DIR / "test_preprocessed.csv", index=False)
    
    print(f"\n✅ Preprocessed data saved!")
    print(f"📊 Final train shape: {train_processed.shape}")
    print(f"📊 Final test shape: {test_processed.shape}")   