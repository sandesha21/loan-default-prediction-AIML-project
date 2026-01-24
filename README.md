# Loan Default Prediction - Capstone Project

A machine learning project to predict loan defaults using the Home Equity (HMEQ) dataset. This capstone project implements multiple classification algorithms to identify borrowers likely to default on their home equity loans.

## 🎯 Project Overview

**Objective**: Develop a predictive model to identify high-risk borrowers and minimize financial losses from loan defaults.

**Dataset**: Home Equity (HMEQ) dataset with 5,960 loan records and 20% default rate.

**Business Impact**: Enable data-driven loan approval decisions and risk assessment strategies.

## 📊 Dataset Information

- **Size**: 5,960 loan records
- **Features**: 12 input variables + 1 target variable
- **Target**: BAD (1 = defaulted, 0 = repaid)
- **Default Rate**: 1,189 cases (20% of total)

### Key Variables
| Variable | Description | Type |
|----------|-------------|------|
| BAD | Loan default status (target) | Binary |
| LOAN | Amount of loan approved | Continuous |
| MORTDUE | Amount due on existing mortgage | Continuous |
| VALUE | Current property value | Continuous |
| REASON | Loan purpose (HomeImp/DebtCon) | Categorical |
| JOB | Applicant's job type | Categorical |
| YOJ | Years at present job | Continuous |
| DEROG | Number of derogatory reports | Discrete |
| DELINQ | Number of delinquent credit lines | Discrete |
| CLAGE | Age of oldest credit line (months) | Continuous |
| NINQ | Number of recent credit inquiries | Discrete |
| CLNO | Number of existing credit lines | Discrete |
| DEBTINC | Debt-to-income ratio | Continuous |

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.7+
Jupyter Notebook
```

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Installation
1. Clone the repository
```bash
git clone <repository-url>
cd loan-default-prediction-project
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook
```bash
jupyter notebook
```

## 📁 Project Structure

```
loan-default-prediction-project/
├── hmeq.csv                       # Raw dataset (5,960 loan records)
├── Capstone_Project_Loan_Default_Prediction_Full_Code_Sandesh_Badwaik_v1.ipynb  # Main analysis notebook v1
├── PROJECT_REQUIREMENTS.md        # Comprehensive project requirements and FAQs
└── README.md                      # This file - project overview and guide
```

### File Descriptions

**Data Files:**
- `hmeq.csv` - Home Equity dataset with loan and borrower information

**Analysis Notebooks:**
- `Capstone_Project_Loan_Default_Prediction_Full_Code_Sandesh_Badwaik_v1.ipynb` - Initial implementation with basic modeling approach


**Documentation:**
- `PROJECT_REQUIREMENTS.md` - Complete project specifications, FAQs, and technical guidance
- `README.md` - Project overview, setup instructions, and usage guide

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Data quality assessment
- Missing value analysis
- Statistical summaries
- Distribution analysis
- Correlation analysis
- Target variable exploration

### 2. Data Preprocessing
- Missing value imputation
- Outlier detection and treatment
- Feature encoding (categorical variables)
- Feature scaling/normalization
- Feature engineering

### 3. Model Development
- **Algorithms Implemented**:
  - Logistic Regression (baseline)
  - Decision Tree Classifier
  - Random Forest Classifier
  - Bagging Classifier
- Train-test split with stratification
- Hyperparameter tuning using GridSearchCV
- Cross-validation for model validation

### 4. Model Evaluation
- Confusion Matrix analysis
- Performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
- Model comparison and selection
- Business impact analysis

## 📈 Performance Targets

| Metric | Target | Business Rationale |
|--------|--------|--------------------|
| Accuracy | > 75% | Overall model reliability |
| Precision | > 70% | Minimize false positives |
| Recall | > 70% | Minimize false negatives (critical for business) |
| F1-Score | > 0.70 | Balanced performance |
| AUC-ROC | > 0.75 | Model discrimination ability |

## 🔑 Key Findings

### Business Insights
- **High-Risk Factors**: Identified key predictors of loan default
- **Cost Analysis**: False negatives are more expensive than false positives
- **Recommendation**: Focus on maximizing Recall to minimize missed defaults

### Technical Insights
- **Class Imbalance**: 20% default rate requires careful handling
- **Feature Importance**: Debt-to-income ratio and credit history are strong predictors
- **Model Performance**: Random Forest shows best overall performance

## 🛠️ Technical Implementation

### Data Preprocessing Pipeline
```python
# Missing value treatment
# Outlier detection using IQR method
# Feature encoding for categorical variables
# Standard scaling for numerical features
```

### Model Training
```python
# Stratified train-test split
# Hyperparameter tuning with GridSearchCV
# Cross-validation for robust evaluation
# Class weight balancing for imbalanced data
```

### Evaluation Framework
```python
# Comprehensive metrics calculation
# ROC curve analysis
# Feature importance visualization
# Business impact assessment
```

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| Bagging Classifier | TBD | TBD | TBD | TBD | TBD |

*Results will be updated upon model completion*

## 💼 Business Recommendations

1. **Risk Assessment**: Implement model-based risk scoring for loan applications
2. **Threshold Optimization**: Adjust decision threshold based on business cost analysis
3. **Feature Monitoring**: Track key risk indicators for portfolio management
4. **Regulatory Compliance**: Ensure fair lending practices and bias mitigation

## 🔧 Common Issues & Solutions

### Technical Troubleshooting
- **AttributeError**: Pass model objects instead of strings
- **100% Accuracy**: Check proper variable dropping in train-test split
- **Scaling Errors**: Convert string variables to numerical before scaling
- **Class Imbalance**: Use class weights or sampling techniques

### Performance Optimization
- **GridSearchCV vs RandomizedSearchCV**: Use RandomizedSearchCV for faster computation
- **Feature Selection**: Apply statistical tests and domain knowledge
- **Cross-Validation**: Use stratified k-fold for imbalanced datasets

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)


## 👥 Contributors

- **Sandesh Badwaik** - Project Developer

## 📄 License

This project is part of the MIT AISP Program.


---

**Note**: This is an educational project for learning purposes. The models and recommendations should be validated with domain experts before any real-world implementation.


---
*Last Updated: March 2025*