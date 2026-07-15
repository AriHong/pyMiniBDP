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
from sklearn.metrics import log_loss
from .utils import append_covariates
from scipy.stats import chi2

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

    
def get_r2(pMB_, eps=1e-12):
    X_dev_raw = pMB_.X_dev_raw
    M_dev_raw = pMB_.M_dev_raw
    y_dev = pMB_.y_dev
    standardscale = pMB_.standardscale
    
    if standardscale:
        X_dev = pMB_.final_x_scaler.fit_transform(X_dev_raw)
        if M_dev_raw is not None:
            M_dev = pMB_.final_m_scaler.fit_transform(M_dev_raw.reshape(-1,1))
        else:
            M_dev = None
    else:
        X_dev = np.asarray(X_dev_raw, dtype = float)
        M_dev = np.asarray(M_dev, dtype=float)

    final_idx = [
            np.where(pMB_.gene_names == g)[0][0]
            for g in pMB_.final_selected_genes
        ]
    X_final = X_dev[:, final_idx]
    X_final = append_covariates(X_final, M_dev)
    final_model = pMB_.final_model

    y_pred_proba = final_model.predict_proba(X_final)[:, 1]
    y_pred = final_model.predict(X_final)

    n_features = X_final.shape[1]

    ll_model = np.sum(
        y_dev * np.log(y_pred_proba)
        + (1 - y_dev) * np.log(1 - y_pred_proba)
    )
    
    prevalence = np.clip(np.mean(y_dev), eps, 1 - eps)

    ll_null = np.sum(
        y_dev * np.log(prevalence)
        + (1 - y_dev) * np.log(1 - prevalence)
    )

    # McFadden pseudo-R2
    pseudo_r2 = 1 - (ll_model / ll_null)
    n_parameters = n_features + 1

    # Adjusted McFadden pseudo-R2
    adjusted_pseudo_r2 = (
        1 - ((ll_model - n_parameters) / ll_null)
    )

    # Null model vs fitted model likelihood-ratio test
    lr_chi2 = 2 * (ll_model - ll_null)
    df = n_features
    p_value = chi2.sf(lr_chi2, df=df)


    return {
        "log_likelihood_model": ll_model,
        "log_likelihood_null": ll_null,
        "pseudo_r2": pseudo_r2,
        "adjusted_pseudo_r2": adjusted_pseudo_r2,
        "likelihood_ratio_chi2": lr_chi2,
        "degrees_of_freedom": df,
        "p_value": p_value,
    }
    