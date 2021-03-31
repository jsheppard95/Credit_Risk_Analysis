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

## Results
For each trained model, we generate predictions from our testing data and
calculate the balanced accuracy score, precisison, and sensitiviy (also known
as recall) to compare the success of each and form a recommendation to predict
credit risk from this data set.

[`credit_risk_resampling.ipynb`](credit_risk_resampling.ipynb)

- Logistic Regression: [`sklearn.linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) with Resampling
    - Naive Random Oversampling: [`imblearn.over_sampling.RandomOverSampler`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html)
        - [Balanced Accuracy Score](Images/ros_balanced_acc_score.png): 0.67
        - [Precision/Recall](Images/ros_prec_rec.png):
            - High Risk:
                - Precision: 0.01
                - Recall: 0.74
            - Low Risk:
                - Precision: 1.00
                - Recall: 0.61
    - SMOTE Oversampling: [`imblearn.over_sampling.SMOTE`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
        - [Balanced Accuracy Score](Images/SMOTE_balanced_acc_score): 0.66
        - [Precision/Recall](Images/SMOTE_prec_rec.png):
            - High Risk:
                - Precision: 0.01
                - Recall: 0.69
            - Low Risk:
                - Precision: 1.00
                - Recall: 0.69
    - Cluster Centroid Undersampling: [`imblearn.under_sampling.ClusterCentroids`](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.ClusterCentroids.html)
        - [Balanced Accuracy Score](Images/cc_balanced_acc_score.png): 0.54
        - [Precision/Recall](Images/cc_prec_rec.png):
            - High Risk:
                - Precision: 0.01
                - Recall: 0.69
            - Low Risk:
                - Precision: 1.00
                - Recall: 0.40
    - SMOTEENN Combination (Over and Under) Sampling: [`imblearn.combine.SMOTEENN`](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTEENN.html)
        - [Balanced Accuracy Score](Images/SMOTEENN_balanced_acc_score.png): 0.64
        - [Precision/Recall](Images/SMOTEENN_prec_rec.png):
            - High Risk:
                - Precision: 0.01
                - Recall: 0.71
            - Low Risk:
                - Precision: 1.00
                - Recall: 0.57

[`credit_risk_ensemble.ipynb`](credit_risk_ensemble.ipynb)

- Balanced Random Forest Classifier: [`imblearn.ensemble.BalancedRandomForestClassifier`](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html)
    - [Balanced Accuracy Score](Images/brfc_balanced_acc_score.png): 0.79
    - [Precision/Recall](Images/brfc_prec_rec.png):
        - High Risk:
            - Precision: 0.03
            - Recall: 0.70
        - Low Risk:
            - Precision: 1.00
            - Recall: 0.87
- Easy Ensemble AdaBoost Classifier: [`imblearn.ensemble.EasyEnsembleClassifier`](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html#imblearn.ensemble.EasyEnsembleClassifier)
    - [Balanced Accuracy Score](Images/eec_balanced_acc_score.png): 0.93
    - [Precision/Recall](Images/eec_prec_rec.png):
        - High Risk:
            - Precision: 0.09
            - Recall: 0.92
        - Low Risk:
            - Precision: 1.00
            - Recall: 0.94