# Random Forest vs XGBoost: Income Prediction Comparison

A comprehensive machine learning project comparing Random Forest and XGBoost classifiers for binary income prediction using the Adult Income dataset.

## 📋 Project Overview

This project performs a detailed comparison between two popular ensemble learning algorithms:
- **Random Forest Classifier** (bagging-based)
- **XGBoost Classifier** (boosting-based)

The goal is to predict whether an individual's income exceeds $50K/year based on demographic and employment features.

## 📊 Dataset

**Adult Income Dataset** (Census Income)
- **Source**: UCI Machine Learning Repository
- **Features**: Age, gender, occupation, education, hours-per-week, etc.
- **Target**: Income (<=50K or >50K)
- **Size**: ~32,000+ samples after cleaning

## 🗂️ Project Structure

```
RF vs XGBoost Comparison/
│
├── data/
│   └── adult_income.csv          # Raw dataset
│
├── Notebook/
│   └── data_preparation.ipynb    # Main analysis notebook
│
├── artifacts/
│   ├── model/                    # Saved models
│   │   ├── best_rf.joblib
│   │   ├── best_xgb.json
│   │   ├── best_xgb.joblib
│   │   ├── preprocessor.joblib
│   │   └── label_encoder.joblib
│   └── figures/                  # Visualizations
│       ├── rf_feature_importance.png
│       └── precision_recall_curve.png
│
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
```

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd "RF vs XGBoost Comparison"
```

2. Create virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Required Libraries

```
numpy
pandas
scikit-learn
xgboost
matplotlib
joblib
jupyter
```

## 🔬 Methodology

### 1. Data Preprocessing
- **Missing Value Handling**: Dropped rows with missing values
- **Feature Engineering**:
  - Numerical features: `age`, `hours-per-week` → StandardScaler
  - Categorical features: `gender`, `occupation`, `education` → OneHotEncoder
- **Target Encoding**: LabelEncoder (0/1 binary labels)
- **Train-Test Split**: 80-20 stratified split

### 2. Model Training

#### Random Forest
- **Hyperparameter Tuning**: GridSearchCV (5-fold CV)
- **Parameters Tuned**:
  - `n_estimators`: [100, 200, 300]
  - `max_depth`: [None, 10, 20]
  - `min_samples_split`: [2, 5, 10]
- **Scoring Metric**: ROC-AUC

#### XGBoost
- **Hyperparameter Tuning**: RandomizedSearchCV (5-fold CV)
- **Parameters Tuned**:
  - `n_estimators`: [50, 100, 200]
  - `learning_rate`: [0.01, 0.1, 0.2]
  - `max_depth`: [3, 6, 10]
  - `subsample`: [0.7, 0.8, 1.0]
  - `colsample_bytree`: [0.7, 0.8, 1.0]
  - Regularization: `gamma`, `reg_alpha`, `reg_lambda`
- **Early Stopping**: 50 rounds on validation set

### 3. Evaluation Metrics
- **ROC-AUC Score**
- **F1 Score**
- **Precision-Recall Curve**
- **Cross-Validation Scores**
- **Training Time**

## 📈 Results

| Metric | Random Forest | XGBoost |
|--------|--------------|---------|
| Test ROC-AUC | 0.XX | 0.XX |
| Test F1 Score | 0.XX | 0.XX |
| CV ROC-AUC (mean ± std) | 0.XX ± 0.XX | 0.XX ± 0.XX |
| Training Time | XX.XX sec | XX.XX sec |

> **Note**: Update with your actual results after running the notebook.

## 🎯 Key Findings

1. **Performance**: [Which model performed better and why]
2. **Training Speed**: [Comparison of computational efficiency]
3. **Interpretability**: Random Forest feature importances vs XGBoost SHAP values
4. **Overfitting**: [Cross-validation stability comparison]

## 📊 Visualizations

- **Feature Importance Plot** (Random Forest)
- **Precision-Recall Curves** (RF vs XGBoost)
- **ROC Curves** (optional: add if created)
- **Learning Curves** (optional: add if created)

## 🔄 Usage

### Training Models

```python
# Load notebook
jupyter notebook Notebook/data_preparation.ipynb

# Or run all cells programmatically
jupyter nbconvert --execute --to notebook \
  --inplace Notebook/data_preparation.ipynb
```

### Loading Saved Models

```python
import joblib
import xgboost as xgb

# Load Random Forest
rf_model = joblib.load('artifacts/model/best_rf.joblib')
preprocessor = joblib.load('artifacts/model/preprocessor.joblib')
label_encoder = joblib.load('artifacts/model/label_encoder.joblib')

# Load XGBoost (native format)
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('artifacts/model/best_xgb.json')

# Or load XGBoost (joblib)
xgb_model = joblib.load('artifacts/model/best_xgb.joblib')
```

### Making Predictions

```python
import pandas as pd

# New data (raw format)
new_data = pd.DataFrame({
    'age': [35],
    'gender': ['Male'],
    'occupation': ['Tech-support'],
    'education': ['Bachelors'],
    'hours-per-week': [40]
})

# Preprocess
X_new = preprocessor.transform(new_data)

# Predict
rf_pred = rf_model.predict(X_new)
xgb_pred = xgb_model.predict(X_new)

# Decode labels
rf_income = label_encoder.inverse_transform(rf_pred)
xgb_income = label_encoder.inverse_transform(xgb_pred)

print(f"RF Prediction: {rf_income[0]}")
print(f"XGB Prediction: {xgb_income[0]}")
```

## 🛠️ Future Improvements

- [ ] Add SHAP/LIME explainability analysis
- [ ] Implement hyperparameter optimization with Optuna
- [ ] Create ensemble (stacking) of RF + XGBoost
- [ ] Add feature importance comparison plots
- [ ] Perform error analysis on misclassified samples
- [ ] Deploy as REST API (Flask/FastAPI)
- [ ] Create interactive dashboard (Streamlit/Dash)
- [ ] Add automated retraining pipeline

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- Your Name - [GitHub Profile](https://github.com/yourusername)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Adult Income dataset
- scikit-learn and XGBoost development teams
- Community tutorials and documentation

## 📧 Contact

For questions or collaboration:
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- GitHub Issues: [Project Issues](https://github.com/yourusername/repo/issues)

---

**Last Updated**: October 29, 2025