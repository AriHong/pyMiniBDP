import pandas as pd
import numpy as np
import anndata
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2
import statsmodels.api as sm
from scipy import stats

def hosmer_lemeshow_test(y_true, y_pred_proba, n_groups=10):
    """
    Performs the Hosmer-Lemeshow goodness-of-fit test.

    Args:
        y_true (array-like): True binary outcomes (0 or 1).
        y_pred_proba (array-like): Predicted probabilities from the logistic regression model.
        n_groups (int): Number of groups for the test (default is 10).

    Returns:
        tuple: HL statistic, p-value
    """
    # Create a DataFrame for easier grouping
    df = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba})

    # Create groups based on predicted probabilities
    df['group'] = pd.qcut(df['y_pred_proba'], q=n_groups, labels=False, duplicates='drop')

    # Calculate observed and expected frequencies for each group
    grouped_data = df.groupby('group').agg(
        n_j=('y_true', 'count'),
        o_j=('y_true', 'sum'),
        mean_pred_proba=('y_pred_proba', 'mean')
    ).reset_index()

    # Calculate expected events
    grouped_data['e_j'] = grouped_data['n_j'] * grouped_data['mean_pred_proba']

    # Calculate the Hosmer-Lemeshow statistic
    hl_statistic = np.sum(
        (grouped_data['o_j'] - grouped_data['e_j'])**2 / 
        (grouped_data['n_j'] * grouped_data['mean_pred_proba'] * (1 - grouped_data['mean_pred_proba']))
    )
    
    # Degrees of freedom
    df_hl = n_groups - 2

    # P-value
    p_value = 1 - chi2.cdf(hl_statistic, df_hl)

    return hl_statistic, p_value
    
def logistic_regression_with_stats(panel=None, BP=None, n_groups=10, use_addedcoef=True, y_col=None):

    idx = [np.where(BP.gene_names == g)[0][0] for g in panel]
    X = BP.X
    if y_col is not None:
        y_raw = BP.adata.obs[y_col].astype(str)
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
    else:
        y = BP.y
    Xsub = X[:, idx]
    if BP.added_coef:
        if use_addedcoef:
            X_ = Xsub.copy()
            X_ = np.concatenate([X_, BP.mcoef.reshape(-1,1)], axis=1)
        else:
            X_ = Xsub.copy()       
    else:
        X_ = Xsub.copy()
    model = LogisticRegression(max_iter=500)
    model.fit(X_, y)

    predict = model.predict(X_)
    predict_proba = model.predict_proba(X_)
    
    r2score = r2_score(y, predict)
    adj_r2score =  1 - (1 - r2score) * (len(y) - 1) / (len(y) - X_.shape[1] - 1)

    hl_stat, pval = hosmer_lemeshow_test(y, predict_proba[:, 1], n_groups=10)
       
    
    return(pval, r2score, adj_r2score, Xsub, model, predict_proba)