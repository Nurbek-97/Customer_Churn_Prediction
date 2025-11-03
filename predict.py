"""
Prediction Script
Make predictions on new data
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from src.config import MODELS_DIR, PROCESSED_DATA_DIR, N_SPLITS

def load_models(model_name='lightgbm'):
    """
    Load all fold models
    """
    models = []
    for fold in range(N_SPLITS):
        model_path = MODELS_DIR / f"{model_name}_fold{fold}.pkl"
        model = joblib.load(model_path)
        models.append(model)
    
    print(f"✅ Loaded {len(models)} {model_name} models")
    return models

def prepare_features(df):
    """
    Prepare features for prediction
    """
    from src.config import TARGET, CATEGORICAL_FEATURES
    
    # Remove non-feature columns
    exclude_cols = [TARGET, 'fold'] + CATEGORICAL_FEATURES + ['tenure_group']
    if TARGET in df.columns:
        exclude_cols.append(TARGET)
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    
    # Handle NaN and inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    return X

def ensemble_predict(test_df, weights=None):
    """
    Make ensemble predictions
    """
    if weights is None:
        weights = {'lightgbm': 0.4, 'catboost': 0.35, 'xgboost': 0.25}
    
    predictions = {}
    
    for model_name, weight in weights.items():
        models = load_models(model_name)
        X_test = prepare_features(test_df)
        
        preds = []
        for model in models:
            pred = model.predict_proba(X_test)[:, 1]
            preds.append(pred)
        
        # Average predictions from all folds
        predictions[model_name] = np.mean(preds, axis=0)
    
    # Weighted ensemble
    ensemble = sum(predictions[name] * weight for name, weight in weights.items())
    
    return ensemble, predictions

def main():
    """
    Main prediction pipeline
    """
    print("🚀 Making predictions on test set...")
    
    # Load test data
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_featured.csv")
    print(f"📊 Test shape: {test_df.shape}")
    
    # Make predictions
    ensemble_pred, individual_preds = ensemble_predict(test_df)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'churn_probability': ensemble_pred,
        'churn_prediction': (ensemble_pred >= 0.5).astype(int),
        'lgbm_pred': individual_preds['lightgbm'],
        'catboost_pred': individual_preds['catboost'],
        'xgboost_pred': individual_preds['xgboost']
    })
    
    # Save
    submission_path = Path('submissions/final_predictions.csv')
    submission.to_csv(submission_path, index=False)
    
    print(f"\n✅ Predictions saved to: {submission_path}")
    print(f"📊 Predicted churn rate: {submission['churn_prediction'].mean():.2%}")
    print(f"📊 Average churn probability: {submission['churn_probability'].mean():.3f}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("📊 PREDICTION SUMMARY")
    print("="*60)
    print(f"Total predictions: {len(submission):,}")
    print(f"Predicted churners: {submission['churn_prediction'].sum():,}")
    print(f"Predicted non-churners: {(1-submission['churn_prediction']).sum():,}")
    print(f"\nProbability distribution:")
    print(f"  Min:  {submission['churn_probability'].min():.3f}")
    print(f"  25%:  {submission['churn_probability'].quantile(0.25):.3f}")
    print(f"  50%:  {submission['churn_probability'].median():.3f}")
    print(f"  75%:  {submission['churn_probability'].quantile(0.75):.3f}")
    print(f"  Max:  {submission['churn_probability'].max():.3f}")
    print("="*60)

if __name__ == "__main__":
    main()