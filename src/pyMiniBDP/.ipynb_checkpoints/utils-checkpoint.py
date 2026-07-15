import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
    

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from scipy import stats

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def _append_covariates(X, M):
    if M is None:
        return X
    return np.concatenate([X, M], axis=1)

@staticmethod
def _safe_inner_cv(y, max_cv=3):
    counts = np.bincount(y)
    if len(counts) < 2:
        return 0
    min_class = int(np.min(counts))
    cv = min(max_cv, min_class)
    return cv if cv >= 2 else 0

def _check_sample_adequacy(n_samples, n_features, sample_feature_min_ratio, sample_feature_warning_ratio):
    if n_features == 0:
        return "critical", 0.0

    ratio = n_samples / n_features

    if ratio < sample_feature_min_ratio:
        return "critical", ratio
    if ratio < sample_feature_warning_ratio:
        return "warning", ratio
    return "adequate", ratio

def _compute_frequencies(rf_panels):
    flattened = []
    for panel in rf_panels:
        flattened.extend(panel)
    return pd.Series(Counter(flattened)).sort_values(ascending=False)

def _get_highfreq(gene_freq, highfreq_quantile, rf_selection_size):
    if gene_freq is None or len(gene_freq) == 0:
        return pd.Series(dtype=float)

    cutoff = gene_freq.quantile(highfreq_quantile)
    high_freq = gene_freq[gene_freq >= cutoff].sort_values(ascending=False) #sort?

    if len(high_freq) == 0:
        high_freq = gene_freq.head(min(rf_selection_size, len(gene_freq)))

    return high_freq
        

def _univariate_filter(X_subset, y, univariate_threshold, correction='fdr_bh'):
    p_values = []

    for j in range(X_subset.shape[1]):
        x0 = X_subset[y == 0, j]
        x1 = X_subset[y == 1, j]

        try:
            _, p = stats.mannwhitneyu(x0, x1, alternative="two-sided")
        except Exception:
            p = 1.0

        if np.isnan(p):
            p = 1.0

        p_values.append(p)

    p_values = np.asarray(p_values, dtype=float)
    if correction:
        _, adj_p, _, _ = multipletests(p_values, method=correction)
    else:
        adj_p = p_values
    
    selected = np.where(adj_p < univariate_threshold)[0]

    return selected, adj_p

def _fit_elastic_net(X, y, l1_ratio, cv, random_state, class_weight=None):
    if len(np.unique(y)) != 2:
        return np.array([], dtype=int), None

    if cv >= 2:
        inner_cv = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=random_state
        )
        model = LogisticRegressionCV(
            penalty="elasticnet",
            solver="saga",
            l1_ratios=[l1_ratio],
            cv=inner_cv,
            max_iter=5000,
            random_state=random_state,
            n_jobs=-1,
            scoring="roc_auc",
            class_weight=class_weight,
            Cs=10,
            refit=True,
        )
    else:
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=l1_ratio,
            C=1.0,
            max_iter=5000,
            random_state=random_state,
            class_weight=class_weight,
        )

    model.fit(X, y)

    coef = np.ravel(model.coef_)
    selected = np.flatnonzero(np.abs(coef) > 1e-8)

    return selected, model



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