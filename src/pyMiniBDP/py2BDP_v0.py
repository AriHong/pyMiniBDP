# ============================================================
# 2BDP faithful reproduction (paper-consistent version)
# ============================================================
import numpy as np
import pandas as pd
from typing import Optional, Sequence, Union

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import statsmodels.api as sm

# ============================================================
# Utility
# ============================================================

def to_dense(X):
    X = np.asarray(X)
    X = X.astype(float)
    X[~np.isfinite(X)] = np.nan
    return X

def encode_binary(y, positive_label):
    y = pd.Series(y).astype(str)
    return (y == positive_label).astype(int).values

def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


class BiomarkerPipeline2BDP:
    def __init__(self,
                 adata,
                 layer="log10",
                 y_col="Prognosis",
                 positive_label="Poor",
                 standardscale = True,
                 n_iter=2000,
                 panel_size=10,
                 top_panels=200):

        self.adata = adata
        self.layer = layer
        self.y_col = y_col
        self.positive_label = positive_label

        self.X_raw = to_dense(adata.layers[layer])
        if standardscale:
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X_raw)
        self.gene_names = np.array(adata.var_names)
        self.y = encode_binary(adata.obs[y_col], positive_label)

        self.n_iter = n_iter
        self.panel_size = panel_size
        self.top_panels = top_panels

    def run_rf(self):
        panels = []

        for i in range(self.n_iter):
            Xtr, _, ytr, _ = train_test_split(
                self.X, self.y, test_size=0.3, stratify=self.y, random_state=i
            )

            rf = RandomForestClassifier(n_estimators=500, random_state=i)
            rf.fit(Xtr, ytr)

            imp = rf.feature_importances_
            idx = np.argsort(imp)[::-1][:self.panel_size]
            panels.append(self.gene_names[idx])

        return np.array(panels)

    def rank_panels(self, panels):
        pos_freq = [
            pd.Series(panels[:, j]).value_counts()
            for j in range(self.panel_size)
        ]

        scores = []
        for i in range(len(panels)):
            score = sum(
                pos_freq[j].get(panels[i, j], 0)
                for j in range(self.panel_size)
            )
            scores.append(score)

        order = np.argsort(scores)[::-1]
        return panels[order[:self.top_panels]]

    def generate_subpanels(self, ranked):
        subs = []
        for p in ranked:
            for k in range(2, len(p)+1):
                subs.append(p[:k])
        return subs

    def run(self):
        panels = self.run_rf()
        ranked = self.rank_panels(panels)
        subpanels = self.generate_subpanels(ranked)
        self.subpanels = subpanels
        return subpanels