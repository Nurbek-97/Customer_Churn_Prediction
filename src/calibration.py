"""
Model Calibration Module
Calibrate probability predictions using isotonic regression
"""

import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import PROCESSED_DATA_DIR

class ProbabilityCalibrator:
    """
    Calibrate model predictions
    """
    
    def __init__(self, method='isotonic'):
        """
        Args:
            method: 'isotonic' or 'sigmoid'
        """
        self.method = method
        self.calibrator = None
        
    def fit_calibrator(self, y_true: np.ndarray, 
                      y_pred_proba: np.ndarray):
        """
        Fit calibration model
        """
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_pred_proba, y_true)
        
        print(f"✅ {self.method.capitalize()} calibrator fitted")
    
    def calibrate(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Calibrate predictions
        """
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted!")
        
        calibrated = self.calibrator.predict(y_pred_proba)
        return np.clip(calibrated, 0, 1)
    
    def evaluate_calibration(self, y_true: np.ndarray,
                            y_pred_proba: np.ndarray,
                            y_pred_calibrated: np.ndarray):
        """
        Evaluate calibration quality
        """
        brier_before = brier_score_loss(y_true, y_pred_proba)
        brier_after = brier_score_loss(y_true, y_pred_calibrated)
        
        logloss_before = log_loss(y_true, y_pred_proba)
        logloss_after = log_loss(y_true, y_pred_calibrated)
        
        print("\n" + "="*60)
        print("📊 CALIBRATION EVALUATION")
        print("="*60)
        print(f"Brier Score Before:  {brier_before:.4f}")
        print(f"Brier Score After:   {brier_after:.4f} ({'↓' if brier_after < brier_before else '↑'})")
        print(f"\nLog Loss Before:     {logloss_before:.4f}")
        print(f"Log Loss After:      {logloss_after:.4f} ({'↓' if logloss_after < logloss_before else '↑'})")
        print("="*60 + "\n")
    
    def plot_calibration(self, y_true: np.ndarray,
                        predictions_dict: dict,
                        n_bins: int = 10,
                        save_path: str = None):
        """
        Plot calibration reliability diagram
        """
        from sklearn.calibration import calibration_curve
        
        plt.figure(figsize=(10, 8))
        
        for name, y_pred in predictions_dict.items():
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred, n_bins=n_bins, strategy='uniform'
            )
            plt.plot(mean_predicted_value, fraction_of_positives,
                    marker='o', linewidth=2, label=name)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Plot (Reliability Diagram)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Calibration plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    print("🚀 Starting probability calibration...")
    
    # Load OOF predictions
    oof_df = pd.read_csv(PROCESSED_DATA_DIR / "oof_predictions.csv")
    
    y_true = oof_df['target'].values
    
    # Create ensemble
    ensemble_pred = (oof_df['lgbm_oof'] * 0.4 + 
                    oof_df['catboost_oof'] * 0.35 + 
                    oof_df['xgb_oof'] * 0.25)
    
    # Initialize calibrator
    calibrator = ProbabilityCalibrator(method='isotonic')
    
    # Fit calibrator
    calibrator.fit_calibrator(y_true, ensemble_pred)
    
    # Calibrate predictions
    ensemble_calibrated = calibrator.calibrate(ensemble_pred)
    
    # Evaluate
    calibrator.evaluate_calibration(y_true, ensemble_pred, ensemble_calibrated)
    
    # Plot
    predictions_dict = {
        'Original Ensemble': ensemble_pred,
        'Calibrated Ensemble': ensemble_calibrated
    }
    
    calibrator.plot_calibration(y_true, predictions_dict,
                               save_path='experiments/calibration_comparison.png')
    
    # Save calibrated predictions
    oof_df['ensemble_pred'] = ensemble_pred
    oof_df['ensemble_calibrated'] = ensemble_calibrated
    oof_df.to_csv(PROCESSED_DATA_DIR / "oof_predictions_final.csv", index=False)
    
    print("✅ Calibration complete!")