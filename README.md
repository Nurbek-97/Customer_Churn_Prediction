# 🎯 Customer Churn Prediction - Production ML System

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

A production-grade customer churn prediction system built with advanced machine learning techniques, following Kaggle Grandmaster best practices.

---

## 🌟 Highlights

- 🏆 **0.8401 AUC-ROC** - Excellent predictive performance
- 💰 **46.5% customer retention** - Strong business impact
- 🔧 **Production-ready** - Clean, modular, maintainable code
- 📊 **Comprehensive evaluation** - Multiple metrics, visualizations
- 🎯 **Profit-optimized** - Business-focused threshold selection
- 🔄 **Fully reproducible** - Fixed seeds, documented pipeline

---

## 📊 Dataset

- **Source**: IBM Telco Customer Churn Dataset
- **Size**: 7,043 customers
- **Features**: 21 original → 65 engineered features
- **Target**: Binary (Churn: Yes/No)
- **Churn Rate**: 26.5%

---

## 🏗️ Project Structure
```
customer-churn/
│
├── data/
│   ├── raw/                      # Original dataset
│   ├── processed/                # Processed & featured data
│   └── folds/                    # CV fold assignments
│
├── src/
│   ├── config.py                 # Configuration & hyperparameters
│   ├── data_loader.py            # Data download & splitting
│   ├── preprocessing.py          # Data cleaning pipeline
│   ├── feature_engineering.py    # Feature creation
│   ├── cv_strategy.py            # Cross-validation setup
│   ├── models.py                 # Model training (LGBM, CB, XGB)
│   ├── evaluation.py             # Model evaluation & metrics
│   ├── calibration.py            # Probability calibration
│   └── profit_optimizer.py       # Business metric optimization
│
├── experiments/
│   ├── RESULTS.md               # Complete results report
│   ├── roc_curves.png           # ROC curve visualization
│   ├── confusion_matrix.png     # Confusion matrix
│   ├── calibration_curves.png   # Calibration plots
│   └── profit_curve.png         # Profit optimization curve
│
├── models/
│   └── trained/                 # Saved model artifacts
│
├── submissions/
│   └── final_predictions.csv    # Test set predictions
│
├── train_pipeline.py            # Complete training pipeline
├── predict.py                   # Prediction script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 Quick Start

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2️⃣ Setup Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ Run Complete Pipeline
```bash
# Option A: Run full pipeline (all steps)
python train_pipeline.py

# Option B: Run individual steps
python src/data_loader.py           # Download & split data
python src/preprocessing.py         # Clean data
python src/feature_engineering.py   # Create features
python src/cv_strategy.py           # Setup CV folds
python src/models.py                # Train models (15-20 mins)
python src/calibration.py           # Calibrate predictions
python src/evaluation.py            # Evaluate models
python src/profit_optimizer.py      # Optimize threshold
```

### 4️⃣ Make Predictions
```bash
python predict.py
```

---

## 📈 Model Performance

### Out-of-Fold Results

| Model | AUC-ROC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| LightGBM | 0.8349 | 0.7821 | 0.6543 | 0.5234 | 0.5812 |
| CatBoost | 0.8298 | 0.7765 | 0.6421 | 0.5123 | 0.5689 |
| XGBoost | 0.8267 | 0.7734 | 0.6389 | 0.5089 | 0.5654 |
| **Ensemble** | **0.8401** | 0.7892 | 0.6678 | 0.5456 | 0.6001 |

### Business Impact

- **Customers Saved**: 695 / 1,495 (46.5%)
- **Optimal Threshold**: 0.540
- **Intervention Rate**: 17.9%
- **ROI**: Significant profit improvement vs. baseline

---

## 🔧 Key Features

### 1. Advanced Feature Engineering
- 46 engineered features from 21 original features
- Tenure-based, charge-based, service-based features
- Interaction features and aggregations

### 2. Robust Cross-Validation
- 5-Fold Stratified K-Fold
- Preserves class distribution
- Out-of-fold predictions for meta-modeling

### 3. Ensemble Learning
- Weighted ensemble (LightGBM 40%, CatBoost 35%, XGBoost 25%)
- Diverse base models for robustness
- Calibrated probability predictions

### 4. Business-Focused Optimization
- Profit-based threshold selection
- Cost-benefit analysis
- Customer Lifetime Value (CLV) consideration

### 5. Production-Ready Code
- Modular, clean architecture
- Comprehensive logging
- Error handling
- Reproducible (fixed seeds)

---

## 📊 Visualizations

The project generates comprehensive visualizations:

1. **ROC Curves** - Compare model discriminative power
2. **Confusion Matrices** - Understand prediction errors
3. **Calibration Curves** - Assess probability reliability
4. **Profit Curves** - Business metric optimization
5. **Feature Importance** - Understand model drivers

All saved in `experiments/` folder.

---

## 🎯 Model Deployment

### Recommended Configuration

**Model**: Ensemble (LightGBM + CatBoost + XGBoost)
**Threshold**: 0.540 (profit-optimized)
**Retraining**: Quarterly or when AUC < 0.80

### API Integration (Future)
```python
from predict import ensemble_predict

# Load new customer data
new_customers = pd.read_csv('new_customers.csv')

# Get predictions
predictions, _ = ensemble_predict(new_customers)

# Apply threshold
churn_flags = predictions >= 0.540
```

---

## 📚 Documentation

- **[RESULTS.md](experiments/RESULTS.md)** - Detailed results & analysis
- **Code Comments** - Inline documentation throughout

---

## 🛠️ Tech Stack

- **Python 3.10**
- **ML**: LightGBM, CatBoost, XGBoost, scikit-learn
- **Data**: pandas, numpy
- **Viz**: matplotlib, seaborn, plotly
- **Utils**: joblib, tqdm, pyyaml

---

## 📝 Methodology

This project follows **Kaggle Grandmaster** best practices:

1. ✅ **Data Understanding** - Thorough EDA
2. ✅ **Robust CV** - Stratified K-Fold
3. ✅ **Feature Engineering** - Domain-driven features
4. ✅ **Model Diversity** - Multiple algorithms
5. ✅ **Ensembling** - Weighted averaging
6. ✅ **Calibration** - Isotonic regression
7. ✅ **Business Focus** - Profit optimization
8. ✅ **Reproducibility** - Fixed seeds, versioning

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- **Dataset**: IBM Telco Customer Churn Dataset
- **Inspiration**: Kaggle Grandmaster methodologies
- **Libraries**: LightGBM, CatBoost, XGBoost teams

---

## 📧 Contact

For questions or collaboration:
- Email: nurbekkhalimjonov070797@gmail.com

---

---

**⭐ If you find this project helpful, please star the repository!**

---

**Built with ❤️ using Kaggle Grandmaster best practices**
