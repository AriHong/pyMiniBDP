import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def compute_frequencies(rf_panels):
    flat = rf_panels.flatten()
    gene_freq = pd.Series(flat).value_counts()
    gene_freq = gene_freq
    return gene_freq


def get_highfreq(gene_freq, quantile):
    high_freq = gene_freq[gene_freq >= np.quantile(gene_freq, quantile)]
    return high_freq


def check_sample_adequacy(n_samples, n_features):
    ratio = n_samples / n_features if n_features > 0 else float('inf')
    
    if ratio < 5:
        return 'critical', ratio  
    elif ratio < 10:
        return 'warning', ratio
    else:
        return 'ok', ratio



def univariate_filter(X, y, threshold=0.05, correction='fdr_bh'):
    """
    Initial filtering using Univariate statistical test

    """
    from scipy.stats import mannwhitneyu
    
    p_values = []
    for i in range(X.shape[1]):
        try:
            # Mann-Whitney U test (non-parametric)
            stat, p = mannwhitneyu(X[y==0, i], X[y==1, i], alternative='two-sided')
            p_values.append(p)
        except:
            p_values.append(1.0)
    
    p_values = np.array(p_values)
    
    # Multiple testing correction
    if correction:
        _, adjusted_p, _, _ = multipletests(p_values, method=correction)
    else:
        adjusted_p = p_values
    
    significant_idx = np.where(adjusted_p < threshold)[0]
    
    return significant_idx, adjusted_p


def elastic_net_selection(X, y, random_state=42, l1_ratio=0.5, n_alphas=20, cv=3):
    """
    Elastic Net 
    """
    
    try:
        # ElasticNet with CV
        model = LogisticRegressionCV(
            penalty='elasticnet',
            solver='saga',
            l1_ratios=[l1_ratio],
            cv=cv,
            max_iter=5000,
            random_state=random_state,
            n_jobs=-1,
            scoring='roc_auc'
        )
        model.fit(X, y)
        
        # coef != 0
        selected_mask = np.abs(model.coef_[0]) > 1e-5
        
        return np.where(selected_mask)[0], model
    except:
        return np.arange(X.shape[1]), None


def permutation_importance_test(X, y, selected_idx, n_permutations=100, random_state=42, l1_ratio=0.5, cv=3 ):
    """
    Permutation test

    """
    
    np.random.seed(random_state)
    X_sel = X[:, selected_idx]
    
    # Original model performance

    model = LogisticRegressionCV(
            penalty='elasticnet',
            solver='saga',
            l1_ratios=[l1_ratio],
            cv=cv,
            max_iter=5000,
            random_state=random_state,
            n_jobs=-1,
            scoring='roc_auc'
        )
    model.fit(X_sel, y)
    original_score = roc_auc_score(y, model.predict_proba(X_sel)[:, 1])
    
    # Permutation scores for each feature
    feature_p_values = []
    
    for feat_idx in range(X_sel.shape[1]):
        perm_scores = []
        
        for _ in range(n_permutations):
            X_perm = X_sel.copy()
            X_perm[:, feat_idx] = np.random.permutation(X_perm[:, feat_idx])
            
            try:
                model_perm = LogisticRegressionCV(
                            penalty='elasticnet',
                            solver='saga',
                            l1_ratios=[l1_ratio],
                            cv=cv,
                            max_iter=5000,
                            random_state=42,
                            n_jobs=-1,
                            scoring='roc_auc'
                        )
                model_perm.fit(X_perm, y)
                perm_score = roc_auc_score(y, model_perm.predict_proba(X_perm)[:, 1])
                perm_scores.append(perm_score)
            except:
                perm_scores.append(original_score)
        
        p_value = np.mean(np.array(perm_scores) >= original_score)
        feature_p_values.append(p_value)
    
    return np.array(feature_p_values)