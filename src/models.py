"""
Model Training Pipeline
Implements LightGBM, CatBoost, and XGBoost with OOF predictions
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from typing import Dict, Tuple, List
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import (LGBM_PARAMS, CATBOOST_PARAMS, XGB_PARAMS, 
                        TARGET, SEED, N_SPLITS, MODELS_DIR,
                        CATEGORICAL_FEATURES)

class ChurnModelTrainer:
    """
    Train multiple models with cross-validation
    """
    
    def __init__(self):
        self.models = {}
        self.oof_predictions = {}
        self.feature_importance = {}
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for modeling
        
        Returns:
            X, y
        """
        # Remove non-feature columns
        exclude_cols = [TARGET, 'fold'] + CATEGORICAL_FEATURES + ['tenure_group']
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[TARGET].copy() if TARGET in df.columns else None
        
        # Handle any remaining NaN or inf
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        print(f"✅ Prepared {len(feature_cols)} features")
        
        return X, y
    
    def train_lightgbm(self, train_df: pd.DataFrame, 
                       n_estimators: int = 1000) -> Dict:
        """
        Train LightGBM with cross-validation
        """
        print("\n🚀 Training LightGBM...")
        
        X, y = self.prepare_features(train_df)
        
        oof_preds = np.zeros(len(train_df))
        feature_importance_list = []
        models = []
        
        for fold in range(N_SPLITS):
            print(f"\n📊 Fold {fold + 1}/{N_SPLITS}")
            
            train_idx = train_df[train_df['fold'] != fold].index
            val_idx = train_df[train_df['fold'] == fold].index
            
            X_train, X_val = X.loc[train_idx], X.loc[val_idx]
            y_train, y_val = y.loc[train_idx], y.loc[val_idx]
            
            # Train model
            model = lgb.LGBMClassifier(**LGBM_PARAMS, n_estimators=n_estimators)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
            )
            
            # Predict OOF
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            
            # Store
            models.append(model)
            feature_importance_list.append(model.feature_importances_)
            
            # Fold metrics
            fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
            print(f"   Fold {fold} AUC: {fold_auc:.4f}")
        
        # Overall metrics
        oof_auc = roc_auc_score(y, oof_preds)
        print(f"\n✅ LightGBM OOF AUC: {oof_auc:.4f}")
        
        # Average feature importance
        avg_importance = np.mean(feature_importance_list, axis=0)
        
        return {
            'models': models,
            'oof_predictions': oof_preds,
            'oof_auc': oof_auc,
            'feature_importance': dict(zip(X.columns, avg_importance)),
            'feature_names': X.columns.tolist()
        }
    
    def train_catboost(self, train_df: pd.DataFrame, 
                       n_estimators: int = 1000) -> Dict:
        """
        Train CatBoost with cross-validation
        """
        print("\n🚀 Training CatBoost...")
        
        X, y = self.prepare_features(train_df)
        
        oof_preds = np.zeros(len(train_df))
        feature_importance_list = []
        models = []
        
        for fold in range(N_SPLITS):
            print(f"\n📊 Fold {fold + 1}/{N_SPLITS}")
            
            train_idx = train_df[train_df['fold'] != fold].index
            val_idx = train_df[train_df['fold'] == fold].index
            
            X_train, X_val = X.loc[train_idx], X.loc[val_idx]
            y_train, y_val = y.loc[train_idx], y.loc[val_idx]
            
            # Train model
            model = cb.CatBoostClassifier(**CATBOOST_PARAMS, iterations=n_estimators)
            
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=100
            )
            
            # Predict OOF
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            
            # Store
            models.append(model)
            feature_importance_list.append(model.feature_importances_)
            
            # Fold metrics
            fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
            print(f"   Fold {fold} AUC: {fold_auc:.4f}")
        
        # Overall metrics
        oof_auc = roc_auc_score(y, oof_preds)
        print(f"\n✅ CatBoost OOF AUC: {oof_auc:.4f}")
        
        # Average feature importance
        avg_importance = np.mean(feature_importance_list, axis=0)
        
        return {
            'models': models,
            'oof_predictions': oof_preds,
            'oof_auc': oof_auc,
            'feature_importance': dict(zip(X.columns, avg_importance)),
            'feature_names': X.columns.tolist()
        }
    
    def train_xgboost(self, train_df: pd.DataFrame, 
                      n_estimators: int = 1000) -> Dict:
        """
        Train XGBoost with cross-validation
        """
        print("\n🚀 Training XGBoost...")
        
        X, y = self.prepare_features(train_df)
        
        oof_preds = np.zeros(len(train_df))
        feature_importance_list = []
        models = []
        
        for fold in range(N_SPLITS):
            print(f"\n📊 Fold {fold + 1}/{N_SPLITS}")
            
            train_idx = train_df[train_df['fold'] != fold].index
            val_idx = train_df[train_df['fold'] == fold].index
            
            X_train, X_val = X.loc[train_idx], X.loc[val_idx]
            y_train, y_val = y.loc[train_idx], y.loc[val_idx]
            
            # Train model
            model = xgb.XGBClassifier(**XGB_PARAMS, n_estimators=n_estimators)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=100
            )
            
            # Predict OOF
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            
            # Store
            models.append(model)
            feature_importance_list.append(model.feature_importances_)
            
            # Fold metrics
            fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
            print(f"   Fold {fold} AUC: {fold_auc:.4f}")
        
        # Overall metrics
        oof_auc = roc_auc_score(y, oof_preds)
        print(f"\n✅ XGBoost OOF AUC: {oof_auc:.4f}")
        
        # Average feature importance
        avg_importance = np.mean(feature_importance_list, axis=0)
        
        return {
            'models': models,
            'oof_predictions': oof_preds,
            'oof_auc': oof_auc,
            'feature_importance': dict(zip(X.columns, avg_importance)),
            'feature_names': X.columns.tolist()
        }
    
    def predict_test(self, test_df: pd.DataFrame, 
                    model_results: Dict) -> np.ndarray:
        """
        Make predictions on test set
        """
        X_test, _ = self.prepare_features(test_df)
        
        predictions = []
        for model in model_results['models']:
            pred = model.predict_proba(X_test)[:, 1]
            predictions.append(pred)
        
        # Average predictions from all folds
        avg_predictions = np.mean(predictions, axis=0)
        
        return avg_predictions
    
    def save_models(self, model_name: str, model_results: Dict):
        """
        Save trained models
        """
        for fold, model in enumerate(model_results['models']):
            model_path = MODELS_DIR / f"{model_name}_fold{fold}.pkl"
            joblib.dump(model, model_path)
        
        print(f"✅ {model_name} models saved to {MODELS_DIR}")

# Main execution
if __name__ == "__main__":
    from src.config import PROCESSED_DATA_DIR
    
    print("🚀 Starting model training pipeline...")
    
    # Load data
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_with_folds.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_featured.csv")
    
    print(f"📊 Train shape: {train_df.shape}")
    print(f"📊 Test shape: {test_df.shape}")
    
    # Initialize trainer
    trainer = ChurnModelTrainer()
    
    # Train models
    lgbm_results = trainer.train_lightgbm(train_df)
    catboost_results = trainer.train_catboost(train_df)
    xgb_results = trainer.train_xgboost(train_df)
    
    # Save models
    trainer.save_models('lightgbm', lgbm_results)
    trainer.save_models('catboost', catboost_results)
    trainer.save_models('xgboost', xgb_results)
    
    # Predict test
    lgbm_test_pred = trainer.predict_test(test_df, lgbm_results)
    catboost_test_pred = trainer.predict_test(test_df, catboost_results)
    xgb_test_pred = trainer.predict_test(test_df, xgb_results)
    
    # Save OOF predictions
    oof_df = pd.DataFrame({
        'lgbm_oof': lgbm_results['oof_predictions'],
        'catboost_oof': catboost_results['oof_predictions'],
        'xgb_oof': xgb_results['oof_predictions'],
        'target': train_df[TARGET]
    })
    oof_df.to_csv(PROCESSED_DATA_DIR / "oof_predictions.csv", index=False)
    
    # Save test predictions
    test_pred_df = pd.DataFrame({
        'lgbm_pred': lgbm_test_pred,
        'catboost_pred': catboost_test_pred,
        'xgb_pred': xgb_test_pred
    })
    test_pred_df.to_csv(PROCESSED_DATA_DIR / "test_predictions.csv", index=False)
    
    print("\n✅ Training complete!")
    print(f"📊 LightGBM OOF AUC: {lgbm_results['oof_auc']:.4f}")
    print(f"📊 CatBoost OOF AUC: {catboost_results['oof_auc']:.4f}")
    print(f"📊 XGBoost OOF AUC: {xgb_results['oof_auc']:.4f}")