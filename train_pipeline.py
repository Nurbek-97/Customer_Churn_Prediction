"""
Complete Training Pipeline
Runs the entire churn prediction workflow
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """
    Run a command and print status
    """
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"❌ Error in: {description}")
        sys.exit(1)
    
    print(f"✅ {description} - COMPLETE!")

def main():
    """
    Execute complete pipeline
    """
    print("\n" + "="*60)
    print("🏆 CUSTOMER CHURN PREDICTION - FULL PIPELINE")
    print("="*60)
    
    steps = [
        ("python src/data_loader.py", "Step 1: Data Loading & Splitting"),
        ("python src/preprocessing.py", "Step 2: Data Preprocessing"),
        ("python src/feature_engineering.py", "Step 3: Feature Engineering"),
        ("python src/cv_strategy.py", "Step 4: Cross-Validation Setup"),
        ("python src/models.py", "Step 5: Model Training (LGBM, CatBoost, XGBoost)"),
        ("python src/calibration.py", "Step 6: Probability Calibration"),
        ("python src/evaluation.py", "Step 7: Model Evaluation"),
        ("python src/profit_optimizer.py", "Step 8: Profit Optimization"),
    ]
    
    for command, description in steps:
        run_command(command, description)
    
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETE!")
    print("="*60)
    print("\n📊 Check the following:")
    print("  - experiments/roc_curves.png")
    print("  - experiments/confusion_matrix.png")
    print("  - experiments/profit_curve.png")
    print("  - data/processed/oof_predictions_final.csv")
    print("\n✅ Ready for submission!")

if __name__ == "__main__":
    main()