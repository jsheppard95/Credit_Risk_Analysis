# Credit_Risk_Analysis
Jupyter notebook files with classification models employing reampling
techniques to evaluate credit risk.

## Overview of the analysis
This project includes Jupyter Notebook files for the analysis of credit loan
data [`LoanStats_2019Q1.csv`](LoanStats_2019Q1.csv) to classify our target
variable `loan_status` with possible values `low_risk` and `high_risk` from
the other features in our data set. We first employ oversampling,
undersampling, and combination sampling techniques for logistic regression
models in [`credit_risk_resampling.ipynb`](credit_risk_resampling.ipynb).
Next, we use ensemble learning classification mdels in
[`credit_risk_ensemple.ipynb`](credit_risk_ensemple.ipynb). Finally, we
compare the accuracy, precision, and recall of each model.

### Resources
- Data Source:
    - [`LoanStats_2019Q1.csv`](LoanStats_2019Q1.csv)
- Software:
    - Python 3.7.9
    - NumPy 1.19.2
    - pandas 1.1.3
    - scikit-learn 0.24.1
    - imbalanced-learn 0.8.0