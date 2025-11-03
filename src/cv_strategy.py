"""
Cross-Validation Strategy
Implements time-aware stratified K-Fold for churn prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import List, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import N_SPLITS, SEED, TARGET, FOLDS_DIR

class ChurnCVStrategy:
    """
    Cross-validation strategy for churn prediction
    """
    
    def __init__(self, n_splits: int = N_SPLITS, random_state: int = SEED):
        self.n_splits = n_splits
        self.random_state = random_state
        self.folds = []
        
    def create_folds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create stratified K-fold splits
        
        Args:
            df: Training dataframe with target
            
        Returns:
            DataFrame with 'fold' column added
        """
        df = df.copy()
        df['fold'] = -1
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=self.n_splits, 
                              shuffle=True, 
                              random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[TARGET])):
            df.loc[val_idx, 'fold'] = fold
        
        print(f"✅ Created {self.n_splits} stratified folds")
        
        # Print fold statistics
        print("\n📊 Fold Statistics:")
        for fold in range(self.n_splits):
            fold_data = df[df['fold'] == fold]
            churn_rate = fold_data[TARGET].mean()
            print(f"  Fold {fold}: {len(fold_data)} samples, "
                  f"Churn rate: {churn_rate:.2%}")
        
        return df
    
    def get_fold_indices(self, df: pd.DataFrame, 
                        fold: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get train and validation indices for a specific fold
        
        Args:
            df: DataFrame with 'fold' column
            fold: Fold number
            
        Returns:
            train_indices, val_indices
        """
        train_idx = df[df['fold'] != fold].index.values
        val_idx = df[df['fold'] == fold].index.values
        
        return train_idx, val_idx
    
    def save_folds(self, df: pd.DataFrame, filename: str = "train_folds.csv"):
        """
        Save fold information
        
        Args:
            df: DataFrame with 'fold' column
            filename: Output filename
        """
        fold_path = FOLDS_DIR / filename
        df[['fold']].to_csv(fold_path, index=True)
        print(f"✅ Folds saved to: {fold_path}")

def create_cv_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cross-validation splits for the dataset
    
    Args:
        df: Training dataframe
        
    Returns:
        DataFrame with fold assignments
    """
    cv_strategy = ChurnCVStrategy()
    df_with_folds = cv_strategy.create_folds(df)
    cv_strategy.save_folds(df_with_folds)
    
    return df_with_folds

# Main execution
if __name__ == "__main__":
    from src.config import PROCESSED_DATA_DIR
    
    print("🚀 Creating CV splits...")
    
    # Load data
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_featured.csv")
    
    # Create folds
    train_with_folds = create_cv_splits(train_df)
    
    # Save with fold column
    train_with_folds.to_csv(PROCESSED_DATA_DIR / "train_with_folds.csv", index=False)
    
    print("✅ CV splits created and saved!")    