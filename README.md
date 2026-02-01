# Loan Default Prediction - Capstone Project

![Python](https://img.shields.io/badge/Python-3.7+-blue?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat-square&logo=python&logoColor=white)

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-HMEQ-lightblue?style=flat-square)

A machine learning project to predict loan defaults using the Home Equity (HMEQ) dataset. This capstone project implements multiple classification algorithms to identify borrowers likely to default on their home equity loans.

---

## 🎯 Project Overview

**Objective**: Develop a predictive model to identify high-risk borrowers and minimize financial losses from loan defaults.

**Dataset**: Home Equity (HMEQ) dataset with 5,960 loan records and 20% default rate.

**Business Impact**: Enable data-driven loan approval decisions and risk assessment strategies.

---

## � Academic Context

This project was developed as part of the **MIT Professional Education Program - Applied AI and Data Science Program**.

**Program Details:**
- **Institution**: Massachusetts Institute of Technology (MIT)
- **Program**: Applied AI and Data Science Program
- **Focus**: Practical application of machine learning and data science techniques
- **Website**: [MIT Online Data Science Program](https://professional-education-gl.mit.edu/mit-online-data-science-program)

The project demonstrates real-world application of data science methodologies, following industry best practices and academic rigor expected in MIT's professional education curriculum.

---

## 🏷️ Keywords & Topics

**Primary Keywords**: Data Science • Machine Learning • Financial Analytics • Python • Loan Default Prediction

**Technical Stack**: Pandas • Scikit-Learn • Statistical Analysis • Data Visualization • Jupyter Notebook • Classification Algorithms

**Business Focus**: Risk Assessment • Credit Scoring • Financial Risk Management • Predictive Modeling • Loan Approval Optimization

**Industry**: Financial Services • Banking • Credit Risk • Lending • Financial Technology

**Project Type**: Financial Risk Analytics & Machine Learning | Industry: Banking & Finance | Focus: Credit Risk Assessment & Default Prevention

---

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

---

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
git clone https://github.com/sandesha21/loan-default-prediction-project.git
cd loan-default-prediction-project
```

2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

3. Launch Jupyter Notebook
```bash
jupyter notebook
```

---

## 📁 Project Structure

```
├── Capstone_Project_Loan_Default_Prediction_Full_Code_Sandesh_Badwaik_v1.ipynb  # Initial analysis and model implementation notebook
├── Capstone_Project_Loan_Default_Prediction_Full_Code_Sandesh_Badwaik_v2.ipynb  # Enhanced analysis with improved modeling approach
├── hmeq.csv                                                                     # Home Equity dataset (5,960 loan records, 13 features)
├── PROJECT_REQUIREMENTS.md                                                      # Detailed project documentation, business context & data dictionary
├── README.md                                                                    # Project overview and setup guide
└── LICENSE                                                                      # Project license information
```

---

## 🎮 Usage

### Quick Start
1. **Open the main notebook**: Start with `Capstone_Project_Loan_Default_Prediction_Full_Code_Sandesh_Badwaik_v2.ipynb` for the most comprehensive analysis
2. **Run all cells**: Execute cells sequentially to reproduce the complete analysis
3. **Explore results**: Review model performance comparisons and business insights

### Key Notebooks
- **v1.ipynb**: Initial implementation with basic modeling approach
- **v2.ipynb**: Enhanced analysis with improved preprocessing and model optimization

### Expected Runtime
- Complete analysis: ~15-20 minutes
- Individual model training: ~2-5 minutes each

---

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

---

## 📈 Performance Targets

| Metric | Target | Business Rationale |
|--------|--------|--------------------|
| Accuracy | > 75% | Overall model reliability |
| Precision | > 70% | Minimize false positives |
| Recall | > 70% | Minimize false negatives (critical for business) |
| F1-Score | > 0.70 | Balanced performance |
| AUC-ROC | > 0.75 | Model discrimination ability |

---

## 🔑 Key Findings

### Business Insights
- **High-Risk Factors**: Identified key predictors of loan default
- **Cost Analysis**: False negatives are more expensive than false positives
- **Recommendation**: Focus on maximizing Recall to minimize missed defaults

### Technical Insights
- **Class Imbalance**: 20% default rate requires careful handling
- **Feature Importance**: Debt-to-income ratio and credit history are strong predictors
- **Model Performance**: Random Forest shows best overall performance

---

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

---

## 📊 Results Summary

The project successfully implemented and compared multiple machine learning algorithms for loan default prediction. Based on the comprehensive analysis in the notebooks:

**Model Performance Comparison:**
- **Random Forest**: Demonstrated the best overall performance with balanced precision and recall
- **Decision Tree**: Showed good interpretability but slightly lower performance than ensemble methods
- **Logistic Regression**: Provided baseline performance with good interpretability
- **Bagging Classifier**: Improved upon single decision tree performance through ensemble approach

**Key Performance Insights:**
- All models exceeded the target performance thresholds (>75% accuracy, >70% precision/recall)
- Random Forest emerged as the recommended model for production deployment
- Feature importance analysis revealed debt-to-income ratio and credit history as top predictors
- Class imbalance handling significantly improved model performance

*Detailed metrics and performance comparisons are available in the analysis notebooks*

---

## 💼 Business Recommendations

1. **Risk Assessment**: Implement model-based risk scoring for loan applications
2. **Threshold Optimization**: Adjust decision threshold based on business cost analysis
3. **Feature Monitoring**: Track key risk indicators for portfolio management
4. **Regulatory Compliance**: Ensure fair lending practices and bias mitigation

---

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

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)


## �‍💻 Author

**Sandesh S. Badwaik**  
*Applied Data Scientist & Machine Learning Engineer*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sbadwaik/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sandesha21)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Academic Use
This project was developed as part of the **MIT Applied AI and Data Science Program (AISP)** and is intended for educational and portfolio purposes.

### Usage Rights
- ✅ **Personal Use**: Free to use for learning and personal projects
- ✅ **Academic Use**: Suitable for educational purposes and academic reference
- ✅ **Portfolio Use**: Can be showcased in professional portfolios
- ⚠️ **Commercial Use**: Requires validation with domain experts before production deployment

### Disclaimer
This is an educational project developed for learning purposes. The models and recommendations should be thoroughly validated with domain experts and tested with current data before any real-world implementation in financial decision-making processes.

### Attribution
If you use this project as a reference, please provide appropriate attribution:
```
Loan Default Prediction Project by Sandesh S. Badwaik
MIT Applied AI and Data Science Program
GitHub: https://github.com/sandesha21/loan-default-prediction-project
```

---

🌟 **If you found this project helpful, please give it a ⭐!**

---

**Note**: This is an educational project for learning purposes. The models and recommendations should be validated with domain experts before any real-world implementation.

---
*Last Updated: February 2026*