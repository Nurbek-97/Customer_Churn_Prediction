"""
Configuration file for Customer Churn Prediction
Following Kaggle Grandmaster best practices
"""

import os
from pathlib import Path

# ======================= PATHS =======================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FOLDS_DIR = DATA_DIR / "folds"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FOLDS_DIR, 
                  MODELS_DIR, EXPERIMENTS_DIR, SUBMISSIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ======================= RANDOM SEEDS =======================
SEED = 42
SEEDS = [42, 2024, 777, 1234, 9999]  # Multiple seeds for stability

# ======================= CV STRATEGY =======================
N_SPLITS = 5
SHUFFLE = True
STRATIFY = True

# ======================= MODEL PARAMETERS =======================

# LightGBM
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': SEED,
    'n_jobs': -1,
    'verbose': -1
}

# CatBoost
CATBOOST_PARAMS = {
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_seed': SEED,
    'verbose': False,
    'task_type': 'CPU',
    'loss_function': 'Logloss',
    'eval_metric': 'AUC'
}

# XGBoost
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'seed': SEED,
    'n_jobs': -1
}

# ======================= FEATURE ENGINEERING =======================
CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

NUMERICAL_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges'
]

TARGET = 'Churn'

# ======================= DATASET INFO =======================
DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATASET_NAME = "telco_churn.csv"

# ======================= EXPERIMENT TRACKING =======================
EXPERIMENT_LOG_FILE = EXPERIMENTS_DIR / "experiment_log.csv"

# ======================= CALIBRATION =======================
CALIBRATION_METHOD = 'isotonic'  # 'isotonic' or 'sigmoid'

# ======================= ENSEMBLE =======================
ENSEMBLE_WEIGHTS = {
    'lgbm': 0.4,
    'catboost': 0.35,
    'xgb': 0.25
}

# ======================= PROFIT OPTIMIZATION =======================
# Business metrics (adjust based on your business case)
CLV = 1000  # Customer Lifetime Value
INTERVENTION_COST = 50  # Cost to retain a customer
FALSE_POSITIVE_COST = 10  # Cost of unnecessary intervention

print(f"✅ Configuration loaded successfully!")
print(f"📁 Project Root: {PROJECT_ROOT}")
print(f"🎲 Random Seed: {SEED}")
print(f"📊 CV Strategy: {N_SPLITS}-Fold Stratified")        