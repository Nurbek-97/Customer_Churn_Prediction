"""
Data Loader for Telco Customer Churn Dataset
Downloads and loads the IBM Telco dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import sys
from typing import Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import RAW_DATA_DIR, DATASET_URL, DATASET_NAME, TARGET

def download_dataset(force_download: bool = False) -> None:
    """
    Download Telco Customer Churn dataset from IBM GitHub
    
    Args:
        force_download: If True, re-download even if file exists
    """
    file_path = RAW_DATA_DIR / DATASET_NAME
    
    if file_path.exists() and not force_download:
        print(f"✅ Dataset already exists at: {file_path}")
        return
    
    print(f"📥 Downloading dataset from: {DATASET_URL}")
    try:
        urllib.request.urlretrieve(DATASET_URL, file_path)
        print(f"✅ Dataset downloaded successfully to: {file_path}")
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        raise

def load_raw_data() -> pd.DataFrame:
    """
    Load raw Telco Customer Churn dataset
    
    Returns:
        DataFrame with raw data
    """
    file_path = RAW_DATA_DIR / DATASET_NAME
    
    if not file_path.exists():
        print("⚠️ Dataset not found. Downloading...")
        download_dataset()
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully!")
        print(f"📊 Shape: {df.shape}")
        print(f"📋 Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def get_train_test_split(df: pd.DataFrame, 
                         test_size: float = 0.2,
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets (simulating Kaggle competition)
    
    Args:
        df: Input dataframe
        test_size: Proportion of data for test set
        random_state: Random seed
        
    Returns:
        train_df, test_df
    """
    from sklearn.model_selection import train_test_split
    
    # Ensure target is binary
    if df[TARGET].dtype == 'object':
        df[TARGET] = (df[TARGET] == 'Yes').astype(int)
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df[TARGET]
    )
    
    print(f"✅ Data split complete!")
    print(f"📊 Train shape: {train_df.shape}")
    print(f"📊 Test shape: {test_df.shape}")
    print(f"🎯 Train churn rate: {train_df[TARGET].mean():.2%}")
    print(f"🎯 Test churn rate: {test_df[TARGET].mean():.2%}")
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def save_data_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Save train/test splits to processed folder
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
    """
    from src.config import PROCESSED_DATA_DIR
    
    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✅ Train data saved to: {train_path}")
    print(f"✅ Test data saved to: {test_path}")

def quick_data_overview(df: pd.DataFrame) -> None:
    """
    Print quick overview of the dataset
    
    Args:
        df: Input dataframe
    """
    print("\n" + "="*60)
    print("📊 DATASET OVERVIEW")
    print("="*60)
    print(f"Shape: {df.shape}")
    print(f"\nTarget Distribution:")
    print(df[TARGET].value_counts(normalize=True))
    print(f"\nMissing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print(f"\nData Types:")
    print(df.dtypes.value_counts())
    print("="*60 + "\n")

# Main execution
if __name__ == "__main__":
    print("🚀 Starting data loading pipeline...")
    
    # Download dataset
    download_dataset()
    
    # Load data
    df = load_raw_data()
    
    # Quick overview
    quick_data_overview(df)
    
    # Create train/test split
    train_df, test_df = get_train_test_split(df)
    
    # Save splits
    save_data_splits(train_df, test_df)
    
    print("✅ Data loading complete!")