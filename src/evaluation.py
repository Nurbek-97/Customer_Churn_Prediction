"""
Model Evaluation Module
Comprehensive evaluation metrics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, 
                             classification_report, f1_score, precision_score,
                             recall_score, average_precision_score)
from sklearn.calibration import calibration_curve
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import TARGET, PROCESSED_DATA_DIR

class ChurnEvaluator:
    """
    Evaluate churn prediction models
    """
    
    def __init__(self):
        self.metrics = {}
        
    def calculate_metrics(self, y_true: np.ndarray, 
                         y_pred_proba: np.ndarray,
                         threshold: float = 0.5) -> dict:
        """
        Calculate comprehensive metrics
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'accuracy': np.mean(y_true == y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'avg_precision': average_precision_score(y_true, y_pred_proba)
        }
        
        return metrics
    
    def plot_roc_curve(self, y_true: np.ndarray, 
                      predictions_dict: dict,
                      save_path: str = None):
        """
        Plot ROC curves for multiple models
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, y_pred in predictions_dict.items():
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Customer Churn Prediction', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ ROC curve saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray,
                             y_pred_proba: np.ndarray,
                             threshold: float = 0.5,
                             save_path: str = None):
        """
        Plot confusion matrix
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'],
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title(f'Confusion Matrix (Threshold = {threshold})', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_calibration_curve(self, y_true: np.ndarray,
                              predictions_dict: dict,
                              n_bins: int = 10,
                              save_path: str = None):
        """
        Plot calibration curves
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, y_pred in predictions_dict.items():
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred, n_bins=n_bins, strategy='uniform'
            )
            plt.plot(mean_predicted_value, fraction_of_positives, 
                    marker='o', label=model_name, linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Calibration curve saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def print_classification_report(self, y_true: np.ndarray,
                                   y_pred_proba: np.ndarray,
                                   threshold: float = 0.5):
        """
        Print detailed classification report
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        print("\n" + "="*60)
        print("📊 CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_true, y_pred, 
                                   target_names=['No Churn', 'Churn']))
        print("="*60 + "\n")
    
    def create_ensemble_predictions(self, predictions_dict: dict,
                                   weights: dict = None) -> np.ndarray:
        """
        Create weighted ensemble predictions
        """
        if weights is None:
            # Equal weights
            weights = {k: 1.0/len(predictions_dict) for k in predictions_dict.keys()}
        
        ensemble_pred = np.zeros_like(list(predictions_dict.values())[0])
        
        for model_name, pred in predictions_dict.items():
            ensemble_pred += weights[model_name] * pred
        
        return ensemble_pred

# Main execution
if __name__ == "__main__":
    print("🚀 Starting model evaluation...")
    
    # Load OOF predictions
    oof_df = pd.read_csv(PROCESSED_DATA_DIR / "oof_predictions.csv")
    
    # Initialize evaluator
    evaluator = ChurnEvaluator()
    
    # Prepare predictions dictionary
    predictions = {
        'LightGBM': oof_df['lgbm_oof'].values,
        'CatBoost': oof_df['catboost_oof'].values,
        'XGBoost': oof_df['xgb_oof'].values
    }
    
    # Create ensemble
    ensemble_pred = evaluator.create_ensemble_predictions(predictions)
    predictions['Ensemble'] = ensemble_pred
    
    y_true = oof_df['target'].values
    
    # Calculate metrics for all models
    print("\n" + "="*60)
    print("📊 MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    for model_name, pred in predictions.items():
        metrics = evaluator.calculate_metrics(y_true, pred)
        print(f"\n{model_name}:")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    print("\n" + "="*60)
    
    # Plot ROC curves
    evaluator.plot_roc_curve(y_true, predictions, 
                            save_path='experiments/roc_curves.png')
    
    # Plot confusion matrix for ensemble
    evaluator.plot_confusion_matrix(y_true, ensemble_pred,
                                   save_path='experiments/confusion_matrix.png')
    
    # Plot calibration curves
    evaluator.plot_calibration_curve(y_true, predictions,
                                    save_path='experiments/calibration_curves.png')
    
    # Print detailed report for ensemble
    evaluator.print_classification_report(y_true, ensemble_pred)
    
    print("✅ Evaluation complete!")