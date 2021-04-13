# Credit_Risk_Analysis
Jupyter notebook files with classification models employing resampling
techniques to evaluate credit risk.

## Overview of the analysis
This project includes Jupyter Notebook files for the analysis of credit loan
data [`LoanStats_2019Q1.csv`](LoanStats_2019Q1.csv) to classify our target
variable `loan_status` with possible values `low_risk` and `high_risk` from
the other features in our data set. We first employ oversampling,
undersampling, and combination sampling techniques for logistic regression
models in [`credit_risk_resampling.ipynb`](credit_risk_resampling.ipynb).
Next, we use ensemble learning classification models in
[`credit_risk_ensemple.ipynb`](credit_risk_ensemble.ipynb). Finally, we
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
calculate the balanced accuracy score, precision, and recall to compare the
success of each and form a recommendation to predict credit risk from this
data set.

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
        - [Balanced Accuracy Score](Images/SMOTE_balanced_acc_score.png): 0.66
        - [Precision/Recall](Images/SMOTE_prec_rec.png):
            - High Risk:
                - Precision: 0.01
                - Recall: 0.63
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

## Summary
Comparing the six models, we find the easy ensemble adaboost classifier to
yield the best metrics for classification of high and low risk loan
applicants. It gave 9% precision for the classification of high risk data,
meaning 9% of high risk identifications were correct. Further, 92% recall
implies that 92% of actual high risk applicants were identified as such. Both
this and the balanced random forest classifier model perform significantly
better than the logistic models with resampling techniques. In the logistic
case, we do not see significant improvement between the four resampling
methods, and in fact we find that the cluster centroid undersampling method
worsens the logistic model. This is likely due to insufficient training data
with the now fewer low risk applicants.

# Usage
All code is contained in the Jupyter Notebook files
[`credit_risk_resampling.ipynb`](credit_risk_resampling.ipynb) and
[`credit_risk_resampling.ipynb`](credit_risk_resampling.ipynb). Therefore
replicating this analysis is accomplished by first cloning the repository
and installing dependencies into an isolated `conda` environment using
```
conda env create -f environment.yml
```
One can then open either Jupyter Notebook file and run all cells. In
`credit_risk_resampling.ipynb`, the classification reports are found at the
end of each resampling section, as shown for
[Naive Random Oversampling](Images/oversampling_full_code). Similarly, in
`credit_risk_ensemble.ipynb`, the classification reports are displayed after
training and testing each ensemble learning classification for model, as shown
for the [Easy Ensemble AdaBoost Classifier](Images/eeac_full_code.png). 