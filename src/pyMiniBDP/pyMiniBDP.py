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
from tqdm import tqdm
from scipy import stats

    

def evaluate_kfold(X, y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    
    aucs, sens, specs = [], [], []   
    for tr, te in kf.split(X):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        if len(np.unique(yte))==1:
            continue
        model = LogisticRegression(max_iter=500)
        model.fit(Xtr, ytr)

        prob = model.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(yte, prob))

        pred = (prob > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        sens.append(tp / (tp + fn))
        specs.append(tn / (tn + fp))

    return np.mean(aucs), np.mean(sens), np.mean(specs), np.std(aucs), np.std(sens), np.std(specs)

def evaluate_rsbmr(X, y, repeat=5, test_size=0.3):
    aucs, sens, specs = [], [], []
    for r in range(repeat):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=r
        )
        if len(np.unique(yte))==1:
            continue
        
        model = LogisticRegression(max_iter=500)
        model.fit(Xtr, ytr)

        prob = model.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(yte, prob))

        pred = (prob > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        sens.append(tp / (tp + fn))
        specs.append(tn / (tn + fp))

    return np.mean(aucs), np.mean(sens), np.mean(specs), np.std(aucs), np.std(sens), np.std(specs)


class BiomarkerPipeline:
    """
    Full biomarker discovery workflow:
    - Random Forest repeated feature selection
    - Gene frequency scoring
    - Panel ranking
    - Sliding window sub-panels
    - Logistic regression evaluations (Strategy A & B)
    """

    def __init__(self, adata, layer="center", y_col="Prognosis",
                 n_iter=1000, panel_size=10, top_n=200, window=10, 
                 kfold=10, rsbmr_repeat=5, 
                 gen_subp_sliding=False,
                added_coef=False):
        
        # Config
        self.adata = adata
        self.layer = layer
        self.y_col = y_col
        self.n_iter = n_iter
        self.panel_size = panel_size
        self.top_n = top_n
        self.window = window
        self.gen_subp_sliding = gen_subp_sliding
        self.added_coef = added_coef
        self.kfold=kfold
        self.rsbmr_repeat= rsbmr_repeat
        
        
        # Extract data
        self.X = adata.layers[layer]
        self.gene_names = np.array(adata.var_names)

        if self.added_coef:
            self.mcoef = stats.zscore(adata.obs[self.added_coef]).values
        else:
            self.mcoef = None
            
        y_raw = adata.obs[y_col].astype(str)
        le = LabelEncoder()
        self.y = le.fit_transform(y_raw)  # convert Good/Poor → 0/1
        self.label_encoder = le

        # Storage
        self.rf_panels = None
        self.gene_ordered = None
        self.gene_freq = None
        self.gene_ordered = None
        self.gene_freq_values = None
        self.top_genes = None
        self.subpanels = None
        self.strategy_A_results = None
        self.strategy_B_results = None

    # -------------------------------------------------------------
    # 1) Random Forest repeated feature selection
    # -------------------------------------------------------------
    def run_random_forest(self, X, y):
        rf_panels = []

        for i in tqdm(range(self.n_iter), desc="Random Forest iterations"):
            if self.added_coef:
                X_ = X.copy()
                X_ = np.concatenate([X_, self.mcoef.reshape(-1,1)], axis=1)
            else:
                X_ = X.copy()
                
            Xtr, Xte, ytr, yte = train_test_split(
                X_, y, test_size=0.3, stratify=y, random_state=i
            )

            rf = RandomForestClassifier(
                n_estimators=500,
                n_jobs=-1,
                random_state=i,
                class_weight="balanced"
            )
            rf.fit(Xtr, ytr)
            idx = np.argsort(rf.feature_importances_[:-1])[::-1]
            top_genes = self.gene_names[idx[:self.panel_size]]
            rf_panels.append(top_genes.tolist())

        self.rf_panels = np.array(rf_panels)

    # -------------------------------------------------------------
    # 2) Compute gene frequency and order frequency and select top-N panels
    # -------------------------------------------------------------
    def compute_frequencies(self, rf_panels, top_n):
        flat = rf_panels.flatten()
        
        self.gene_freq = pd.Series(flat).value_counts()
        self.gene_freq = self.gene_freq[self.gene_freq>=2]
        self.gene_ordered = np.array(self.gene_freq.index)
        top_n = min(len(self.gene_ordered), top_n)
        self.top_genes = self.gene_ordered[:top_n]  

    # -------------------------------------------------------------
    # 3) Sliding window sub-panels
    # -------------------------------------------------------------

    def sliding_window(self, panel, window):
        sub = []
        n = len(panel)
        for s in range(0, n - window + 1):
            sub.append(panel[s:s + window])
        return sub
        
    def random_sampling(self, panel, window):
        n_sampling = len(panel)
        sub = []
        for i in range(n_sampling):
            sub.append(random.sample(panel.tolist(), window))
        return sub
        

    def generate_subpanels(self, top_genes, window, gen_subp_sliding):
        if gen_subp_sliding:
            self.subpanels = self.sliding_window(top_genes, window)
        else:
            self.subpanels = self.random_sampling(top_genes, window)


    # -------------------------------------------------------------
    # 5) Evaluate Strategy A & B for all top panels
    # -------------------------------------------------------------
    def evaluate_panels(self, X, y, subpanels, gene_freq, window):
        aucA_ = np.zeros((len(subpanels), window-1))
        aucB_ = np.zeros((len(subpanels), window-1))
        senA_ = np.zeros((len(subpanels), window-1))
        senB_ = np.zeros((len(subpanels), window-1))
        specA_ = np.zeros((len(subpanels), window-1))
        specB_ = np.zeros((len(subpanels), window-1))

        aucAstd_ = np.zeros((len(subpanels), window-1))
        aucBstd_ = np.zeros((len(subpanels), window-1))
        senAstd_ = np.zeros((len(subpanels), window-1))
        senBstd_ = np.zeros((len(subpanels), window-1))
        specAstd_ = np.zeros((len(subpanels), window-1))
        specBstd_ = np.zeros((len(subpanels), window-1))
        
        genes = []
                            
        
        for i, panel in enumerate(tqdm(subpanels, desc="Model evaluation")):
            panel = gene_freq[panel].sort_values(ascending=False).index.tolist()
            idx = [np.where(self.gene_names == g)[0][0] for g in panel]
            genes.append(panel)
            
            Xsub = X[:, idx]
                           
            for w, size in enumerate(range(2, len(idx)+1)):
                Xsel = Xsub[:, :size]
                if self.added_coef:
                    Xsel = np.concatenate([Xsel, self.mcoef.reshape(-1,1)], axis=1)
                
                Xsel = sm.add_constant(Xsel)
                evaluate_kfold(Xsel, y)
                # HL test
                #logit = sm.Logit(y, sm.add_constant(Xsel))
                #result = logit.fit(disp=False)
                
                #hl_p = hosmer_lemeshow_test(y, result.predict())
    
                aucA, senA, specA, aucAstd, senAstd, specAstd = evaluate_kfold(Xsel, y, self.kfold)
                aucB, senB, specB, aucBstd, senBstd, specBstd = evaluate_rsbmr(Xsel, y, self.rsbmr_repeat)
    
                aucA_[i, w] = aucA
                aucB_[i, w] = aucB
                senA_[i, w] = senA
                senB_[i, w] = senB
                specA_[i, w] = specA
                specB_[i, w] = specB

                aucAstd_[i, w] = aucAstd
                aucBstd_[i, w] = aucBstd
                senAstd_[i, w] = senAstd
                senBstd_[i, w] = senBstd
                specAstd_[i, w] = specAstd
                specBstd_[i, w] = specBstd

        self.aucA_ = aucA_
        self.aucB_ = aucB_
        self.senA_ = senA_
        self.senB_ = senB_
        self.specA_ = specA_
        self.specB_ = specB_

        self.aucAstd_ = aucAstd_
        self.aucBstd_ = aucBstd_
        self.senAstd_ = senAstd_
        self.senBstd_ = senBstd_
        self.specAstd_ = specAstd_
        self.specBstd_ = specBstd_
        
        self.genes = genes

    
    
    def run(self):
        print("\n[1] Random Forest feature selection")
        self.run_random_forest(self.X, self.y)

        print("\n[2] Computing gene frequencies and select top gene")
        self.compute_frequencies(self.rf_panels, self.top_n)

        print("\n[3] Generating sliding window sub-panels")
        self.generate_subpanels(self.top_genes, self.window, self.gen_subp_sliding)

        print("\n[4] Evaluating models")
        self.evaluate_panels(self.X, self.y, self.subpanels, self.gene_freq, self.window)

        print("\n[5] Pipeline complete!")