"""
Feature Engineering Pipeline
Creates advanced features for churn prediction
Following Kaggle Grandmaster best practices
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from typing import Tuple, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import (CATEGORICAL_FEATURES, NUMERICAL_FEATURES, 
                        TARGET, SEED)

class ChurnFeatureEngineer:
    """
    Advanced feature engineering for churn prediction
    """
    
    def __init__(self):
        self.target_encoders = {}
        self.label_encoders = {}
        self.feature_names = []
        
    def create_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tenure-based features
        """
        df = df.copy()
        
        # Tenure categories
        df['tenure_group'] = pd.cut(df['tenure'], 
                                     bins=[0, 12, 24, 48, 72], 
                                     labels=['0-1yr', '1-2yr', '2-4yr', '4yr+'])
        
        # Tenure bins
        df['tenure_bin'] = pd.qcut(df['tenure'], q=4, labels=False, duplicates='drop')
        
        # Is new customer (tenure < 6 months)
        df['is_new_customer'] = (df['tenure'] < 6).astype(int)
        
        # Is long-term customer (tenure > 3 years)
        df['is_longterm_customer'] = (df['tenure'] > 36).astype(int)
        
        print("✅ Tenure features created")
        return df
    
    def create_charge_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create charge-based features
        """
        df = df.copy()
        
        # Average monthly charge per tenure month
        df['avg_monthly_charge'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        # Charge increase rate
        df['charge_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
        
        # Total to monthly ratio
        df['total_to_monthly_ratio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)
        
        # Monthly charges bins
        df['monthly_charges_bin'] = pd.qcut(df['MonthlyCharges'], 
                                             q=5, labels=False, duplicates='drop')
        
        # High/Low monthly charge flags
        df['is_high_monthly_charge'] = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(int)
        df['is_low_monthly_charge'] = (df['MonthlyCharges'] < df['MonthlyCharges'].quantile(0.25)).astype(int)
        
        print("✅ Charge features created")
        return df
    
    def create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create service-related features
        """
        df = df.copy()
        
        # Count of services
        service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # Number of active services
        df['num_services'] = 0
        for col in service_cols:
            if col in df.columns:
                df['num_services'] += (df[col] == 'Yes').astype(int)
        
        # Has internet services
        if 'InternetService' in df.columns:
            df['has_internet'] = (df['InternetService'] != 'No').astype(int)
            df['has_fiber'] = (df['InternetService'] == 'Fiber optic').astype(int)
            df['has_dsl'] = (df['InternetService'] == 'DSL').astype(int)
        
        # Has phone services
        if 'PhoneService' in df.columns:
            df['has_phone'] = (df['PhoneService'] == 'Yes').astype(int)
        
        # Has streaming services
        streaming_cols = ['StreamingTV', 'StreamingMovies']
        df['num_streaming'] = 0
        for col in streaming_cols:
            if col in df.columns:
                df['num_streaming'] += (df[col] == 'Yes').astype(int)
        
        # Has security services
        security_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        df['num_security'] = 0
        for col in security_cols:
            if col in df.columns:
                df['num_security'] += (df[col] == 'Yes').astype(int)
        
        # Service to charge ratio
        df['services_per_charge'] = df['num_services'] / (df['MonthlyCharges'] + 1)
        
        print("✅ Service features created")
        return df
    
    def create_contract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create contract-related features
        """
        df = df.copy()
        
        if 'Contract' in df.columns:
            # Is month-to-month
            df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
            
            # Is long contract
            df['is_long_contract'] = (df['Contract'].isin(['One year', 'Two year'])).astype(int)
        
        if 'PaperlessBilling' in df.columns:
            df['has_paperless'] = (df['PaperlessBilling'] == 'Yes').astype(int)
        
        print("✅ Contract features created")
        return df
    
    def create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create demographic features
        """
        df = df.copy()
        
        # Family indicator
        if 'Partner' in df.columns and 'Dependents' in df.columns:
            df['has_family'] = ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')).astype(int)
            df['family_size'] = (df['Partner'] == 'Yes').astype(int) + (df['Dependents'] == 'Yes').astype(int)
        
        # Senior citizen flag
        if 'SeniorCitizen' in df.columns:
            df['is_senior'] = df['SeniorCitizen'].astype(int)
        
        print("✅ Demographic features created")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features
        """
        df = df.copy()
        
        # Tenure x Monthly Charges
        df['tenure_monthly_interaction'] = df['tenure'] * df['MonthlyCharges']
        
        # Services x Tenure
        if 'num_services' in df.columns:
            df['services_tenure_interaction'] = df['num_services'] * df['tenure']
        
        # Contract x Charges
        if 'is_month_to_month' in df.columns:
            df['contract_charge_interaction'] = df['is_month_to_month'] * df['MonthlyCharges']
        
        print("✅ Interaction features created")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                    fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding
        
        Args:
            df: Input dataframe
            fit: If True, fit encoders (for training). If False, use fitted encoders (for test)
        """
        df = df.copy()
        
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        df[f'{col}_encoded'] = df[col].astype(str).map(
                            lambda x: self.label_encoders[col].transform([x])[0] 
                            if x in self.label_encoders[col].classes_ 
                            else -1
                        )
        
        # Add tenure_group encoding if it exists
        if 'tenure_group' in df.columns:
            if fit:
                self.label_encoders['tenure_group'] = LabelEncoder()
                df['tenure_group_encoded'] = self.label_encoders['tenure_group'].fit_transform(
                    df['tenure_group'].astype(str)
                )
            else:
                if 'tenure_group' in self.label_encoders:
                    df['tenure_group_encoded'] = df['tenure_group'].astype(str).map(
                        lambda x: self.label_encoders['tenure_group'].transform([x])[0]
                        if x in self.label_encoders['tenure_group'].classes_
                        else -1
                    )
        
        print("✅ Categorical features encoded")
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform training data
        """
        print("\n🔧 Engineering features for training data...")
        
        df = self.create_tenure_features(df)
        df = self.create_charge_features(df)
        df = self.create_service_features(df)
        df = self.create_contract_features(df)
        df = self.create_demographic_features(df)
        df = self.create_interaction_features(df)
        df = self.encode_categorical_features(df, fit=True)
        
        print(f"✅ Feature engineering complete! Final shape: {df.shape}")
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test data using fitted encoders
        """
        print("\n🔧 Engineering features for test data...")
        
        df = self.create_tenure_features(df)
        df = self.create_charge_features(df)
        df = self.create_service_features(df)
        df = self.create_contract_features(df)
        df = self.create_demographic_features(df)
        df = self.create_interaction_features(df)
        df = self.encode_categorical_features(df, fit=False)
        
        print(f"✅ Feature engineering complete! Final shape: {df.shape}")
        return df

def engineer_features(train_df: pd.DataFrame, 
                     test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Engineer features for both train and test data
    """
    engineer = ChurnFeatureEngineer()
    
    train_featured = engineer.fit_transform(train_df)
    test_featured = engineer.transform(test_df)
    
    return train_featured, test_featured

# Main execution
if __name__ == "__main__":
    from src.config import PROCESSED_DATA_DIR
    
    print("🚀 Starting feature engineering pipeline...")
    
    # Load preprocessed data
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_preprocessed.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_preprocessed.csv")
    
    print(f"📊 Original train shape: {train_df.shape}")
    print(f"📊 Original test shape: {test_df.shape}")
    
    # Engineer features
    train_featured, test_featured = engineer_features(train_df, test_df)
    
    # Save featured data
    train_featured.to_csv(PROCESSED_DATA_DIR / "train_featured.csv", index=False)
    test_featured.to_csv(PROCESSED_DATA_DIR / "test_featured.csv", index=False)
    
    print(f"\n✅ Featured data saved!")
    print(f"📊 Final train shape: {train_featured.shape}")
    print(f"📊 Final test shape: {test_featured.shape}")
    print(f"📋 New features: {train_featured.shape[1] - train_df.shape[1]} features added!")