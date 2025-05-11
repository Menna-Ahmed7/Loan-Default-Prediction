# Loan Default Prediction System

## Project Overview
This project develops a machine learning solution to predict loan defaults for a finance company, enabling better decision-making when approving loans. The system analyzes customer information including income, employment, and credit history to classify potential borrowers as either likely to repay or likely to default.

## Business Motivation
- **Risk Mitigation**: Reduce financial losses from loan defaults
- **Business Growth**: Avoid rejecting creditworthy customers
- **Enhanced Decision Making**: Leverage customer data for smarter lending decisions
- **Cost Reduction**: Minimize expenses associated with debt collection

## Dataset
- **Size**: 27,000 rows (training), 47 features
- **Class Distribution**: Imbalanced (85.3% fully paid, 14.7% defaulted)
- **Features**: Income, loan amount, debt-to-income ratio, payment history, etc.

## Methodology

### Data Preprocessing
- **Data Cleaning**:
  - Removed columns with >60% missing values
  - Filled numeric missing values with median
  - Filled categorical missing values with mode
  
- **Feature Transformation**:
  - Applied log transformation for skewed and zero-inflated features
  - Standardized normally distributed features
  
- **Feature Selection & Encoding**:
  - Removed zero-variance features
  - Eliminated high-cardinality categorical features (>100 unique values)
  - Selected features based on correlation with target (threshold > 0.1)
  - Applied one-hot encoding to categorical variables
  - Converted boolean columns to numeric (0/1)

### Modeling Approach
- **Class Imbalance Handling**: Implemented class weighting in models
- **Cross-Validation**: Used 3-fold cross-validation with weighted F1 score
- **GPU Acceleration**: Utilized RAPIDS cuML for enhanced computational performance
- **Hyperparameter Optimization**: Performed grid search for all models

### Models Implemented
1. **Random Forest**
2. **Support Vector Machine**
3. **Logistic Regression**
4. **AdaBoost with Decision Tree base estimators**

## Results

| Metric | Random Forest | SVM | Logistic Regression | AdaBoost |
|--------|---------------|-----|---------------------|---------|
| Overall Test Accuracy | 0.9700 | 0.9639 | 0.9639 | **0.9746** |
| Class 0 Precision | 0.97 | 0.96 | 0.96 | **0.98** |
| Class 0 Recall | **1.00** | **1.00** | **1.00** | 0.99 |
| Class 0 F1-Score | 0.98 | 0.98 | 0.98 | **0.99** |
| Class 1 Precision | 0.97 | **1.00** | **1.00** | 0.95 |
| Class 1 Recall | 0.81 | 0.75 | 0.76 | **0.87** |
| Class 1 F1-Score | 0.88 | 0.86 | 0.86 | **0.91** |
| Macro Avg F1 | 0.93 | 0.92 | 0.92 | **0.95** |
| Weighted Avg F1 | 0.97 | 0.96 | 0.96 | **0.97** |

## Key Findings

1. **Best Model**: AdaBoost classifier emerged as the top performer with the highest overall accuracy (97.46%) and best minority class detection (Class 1 Recall: 0.87, F1: 0.91)

2. **Ensemble Methods Advantage**: Ensemble techniques (AdaBoost and Random Forest) demonstrated superior performance in handling class imbalance compared to linear models

3. **Outlier Handling**: Removing outliers proved detrimental to model performance, particularly for linear models, confirming that these data points contained valuable information

4. **Linear Models Limitations**: Both SVM and Logistic Regression achieved identical accuracy (96.39%) but were limited by lower recall for the minority class

## Model Configuration Details

### AdaBoost (Best Performer)
- Algorithm: SAMME.R
- Base estimator max depth: 3
- Learning rate: 0.1
- Number of estimators: 200

### Random Forest
- Estimators: 100
- Max depth: 20
- Min samples split: 10
- Min samples leaf: 1
- Max features: 'sqrt'

### SVM
- Kernel: Linear
- C: 0.1
- Gamma: 'auto'

### Logistic Regression
- Penalty: L2
- C: 1
- Solver: liblinear

## Future Work

- Explore more advanced feature engineering techniques
- Implement deep learning approaches for comparison
- Create a production-ready API for real-time loan approval decisions

## Contributors <a name = "Contributors"></a>

<table>
  <tr>
    <td align="center">
    <a href="https://github.com/Menna-Ahmed7" target="_black">
    <img src="https://avatars.githubusercontent.com/u/110634473?v=4" width="150px;" alt="https://github.com/Menna-Ahmed7"/>
    <br />
    <sub><b>Mennatallah Ahmed</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/MostafaBinHani" target="_black">
    <img src="https://avatars.githubusercontent.com/u/119853216?v=4" width="150px;" alt="https://github.com/MostafaBinHani"/>
    <br />
    <sub><b>Mostafa Hani</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/MohammadAlomar8" target="_black">
    <img src="https://avatars.githubusercontent.com/u/119791309?v=4" width="150px;" alt="https://github.com/MohammadAlomar8"/>
    <br />
    <sub><b>Mohammed Alomar</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/mou-code" target="_black">
    <img src="https://avatars.githubusercontent.com/u/123744354?v=4" width="150px;" alt="https://github.com/mou-code"/>
    <br />
    <sub><b>Moustafa Mohammed</b></sub></a>
    </td>
  </tr>
 </table>
