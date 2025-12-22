import pickle
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
import warnings
warnings.filterwarnings('ignore')

from .utils import *



class BiomarkerPipelineKFold:
    """
    - Repeated Stratified K-Fold 
    - Univariate filtering 
    - Elastic Net 
    - Permutation test 
    - Holdout option
    """

    def __init__(self, adata, layer="center", y_col="Prognosis",
                 n_iter=100,  # RF iteration 
                 rf_selection_size=15,  
                 n_splits=5,  # K-fold 
                 n_repeats=5,  # Repeated K-fold
                 added_coef=False,
                 univariate_threshold=0.2,  
                 highfreq_quantile=0.85,  
                 correlation_threshold=0.9,
                 elastic_net_l1_ratio=0.5,
                 use_holdout=False,  
                 holdout_size=0.2
                ):  
        
        # Config
        self.adata = adata
        self.layer = layer
        self.y_col = y_col
        self.n_iter = n_iter
        self.rf_selection_size = rf_selection_size
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.added_coef = added_coef
        
        # Feature selection 
        self.univariate_threshold = univariate_threshold
        self.highfreq_quantile = highfreq_quantile
        self.correlation_threshold = correlation_threshold
        self.elastic_net_l1_ratio = elastic_net_l1_ratio
        self.use_holdout = use_holdout
        self.holdout_size = holdout_size
        
        # Extract data
        self.X = adata.layers[layer]
        self.gene_names = np.array(adata.var_names)
        
        # Standardization 
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        if self.added_coef:
            self.mcoef = stats.zscore(adata.obs[self.added_coef]).values
        else:
            self.mcoef = None
            
        y_raw = adata.obs[y_col].astype(str)
        le = LabelEncoder()
        self.y = le.fit_transform(y_raw)
        self.label_encoder = le
        
        n_samples = len(self.X)
        n_features = self.X.shape[1]
        min_class_size = np.min(np.bincount(self.y))
        
        print(f"\n{'='*70}")
        print(f"SAMPLE SIZE ASSESSMENT")
        print(f"{'='*70}")
        print(f"Total samples: {n_samples}")
        print(f"Class distribution: {np.bincount(self.y)}")
        print(f"Minimum class size: {min_class_size}")
        print(f"Total features: {n_features}")
        
        # Holdout
        if self.use_holdout and n_samples >= 60:
            self.X_dev, self.X_holdout, self.y_dev, self.y_holdout, \
            self.dev_idx, self.holdout_idx = train_test_split(
                self.X, self.y, np.arange(len(self.X)),
                test_size=self.holdout_size,
                stratify=self.y,
                random_state=42
            )
            print(f"\nUsing holdout set:")
            print(f"  Development: {len(self.X_dev)}")
            print(f"  Holdout: {len(self.X_holdout)}")
        else:
            if self.use_holdout:
                print(f"\nSample size too small for holdout set.")
                print(f"Using full repeated CV instead.")
            self.X_dev = self.X
            self.y_dev = self.y
            self.dev_idx = np.arange(len(self.X))
            self.X_holdout = None
            self.y_holdout = None
            self.use_holdout = False
        
        print(f"{'='*70}\n")
        
        # Storage
        self.cv_results = []
        self.final_model = None
        self.final_selected_genes = None
        self.holdout_performance = None

    def run_random_forest_fold(self, X_train, y_train, fold_idx):
        rf_panels = []
        
        for i in tqdm(range(self.n_iter), 
                     desc=f"Fold {fold_idx} - RF", leave=False):
            
            # Bootstrap sampling
            boot_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
            X_boot = X_train[boot_idx]
            y_boot = y_train[boot_idx]
            
            if self.added_coef:
                mcoef_boot = self.mcoef[boot_idx]
                X_boot = np.concatenate([X_boot, mcoef_boot.reshape(-1,1)], axis=1)

            rf = RandomForestClassifier(
                n_estimators=300, 
                max_depth=5,  
                min_samples_leaf=max(2, len(X_train)//20),  
                min_samples_split=max(5, len(X_train)//10),
                n_jobs=-1,
                random_state=fold_idx*10000 + i,
                class_weight="balanced"
            )
            
            try:
                rf.fit(X_boot, y_boot)
                
                if self.added_coef:
                    importances = rf.feature_importances_[:-1]
                else:
                    importances = rf.feature_importances_
                
                idx = np.argsort(importances)[::-1]
                top_genes = self.gene_names[idx[:self.rf_selection_size]]
                rf_panels.append(top_genes.tolist())
            except:
                continue

        return np.array(rf_panels) if len(rf_panels) > 0 else np.array([])

    def select_features_robust(self, X_train, y_train, gene_freq, train_indices):
        print(f"    Starting robust feature selection...")
        
        #select high freq gene
        high_freq = get_highfreq(gene_freq, quantile=self.highfreq_quantile)
        
        if len(high_freq) == 0:
            print(f"    No high frequency genes found!")
            return [], None
        
        idx = [np.where(self.gene_names == g)[0][0] for g in high_freq.index]
        print(f"    High freq genes (>{self.highfreq_quantile} quantile): {len(idx)}")
        
        status, ratio = check_sample_adequacy(len(X_train), len(idx))
        print(f"    Sample/Feature ratio: {ratio:.2f} ({status})")
        
        # 3) Univariate filtering 
        if status == 'critical' or status == 'warning':
            print(f"    Applying univariate filtering...")
            X_subset = X_train[:, idx]
            
            univar_idx, adj_p = univariate_filter(
                X_subset, y_train, 
                threshold=self.univariate_threshold,
                correction='fdr_bh'
            )
            
            if len(univar_idx) > 10:
                idx = [idx[i] for i in univar_idx]
                print(f"    After univariate filtering: {len(idx)} genes")
            else:
                best_idx = np.argsort(adj_p)[:min(10, len(adj_p))]
                idx = [idx[i] for i in best_idx]
                print(f"    Using top {len(idx)} genes by p-value")
        
        # 4) correlation filtering
        if len(idx) > 1:
            X_subset = X_train[:, idx]
            correlation_matrix = np.corrcoef(X_subset, rowvar=False)
            
            to_remove = set()
            for i in range(len(correlation_matrix)):
                for j in range(i + 1, len(correlation_matrix)):
                    if abs(correlation_matrix[i, j]) >= self.correlation_threshold:
                        if gene_freq[high_freq.index[i]] < gene_freq[high_freq.index[j]]:
                            to_remove.add(idx[i])
                        else:
                            to_remove.add(idx[j])
            
            idx = [i for i in idx if i not in to_remove]
            print(f"    After correlation filtering: {len(idx)} genes")
        
        # 5) check sample ratio
        status, ratio = check_sample_adequacy(len(X_train), len(idx))
        
        if ratio < 5 and len(idx) > 5:
            max_features = max(5, len(X_train) // 5)
            idx = idx[:max_features]
            print(f"    Reduced to {len(idx)} genes for safety (ratio={len(X_train)/len(idx):.2f})")
        
        # 6) Elastic Net
        if len(idx) > 0:
            print(f"    Applying Elastic Net...")
            X_subset = X_train[:, idx]
            
            if self.added_coef:
                mcoef_train = self.mcoef[train_indices]
                X_subset = np.concatenate([X_subset, mcoef_train.reshape(-1,1)], axis=1)
            
            try:
                selected_mask, model = elastic_net_selection(
                    X_subset, y_train,
                    l1_ratio=self.elastic_net_l1_ratio,
                    cv=min(3, len(X_train)//10)  # CV fold
                )
                
                if len(selected_mask) > 0:
                    idx = [idx[i] for i in selected_mask if i < len(idx)]
                    print(f"    After Elastic Net: {len(idx)} genes")
                else:
                    # top 5개만
                    idx = idx[:min(5, len(idx))]
                    print(f"    Elastic Net removed all, keeping top {len(idx)} genes")
                    
            except Exception as e:
                print(f"    Elastic Net failed: {str(e)}")
                print(f"    Keeping current {len(idx)} genes")
                model = None
        
        print(f"    Final selected genes: {len(idx)}")
        
        return idx, model

    def evaluate_on_test(self, X_train, y_train, X_test, y_test, 
                        selected_idx, train_indices, test_indices, cv=3):
        """Evaluate test"""
        if len(selected_idx) == 0:
            return None, None
        
        X_train_sel = X_train[:, selected_idx]
        X_test_sel = X_test[:, selected_idx]
        
        if self.added_coef:
            mcoef_train = self.mcoef[train_indices]
            mcoef_test = self.mcoef[test_indices]
            X_train_sel = np.concatenate([X_train_sel, mcoef_train.reshape(-1,1)], axis=1)
            X_test_sel = np.concatenate([X_test_sel, mcoef_test.reshape(-1,1)], axis=1)
        
        try:

            model = LogisticRegressionCV(
                penalty='elasticnet',
                solver='saga',
                l1_ratios=[self.elastic_net_l1_ratio],
                cv=cv,
                max_iter=5000,
                random_state=42,
                n_jobs=-1,
                scoring='roc_auc'
            )

            model.fit(X_train_sel, y_train)
            
            # predict
            y_pred_proba = model.predict_proba(X_test_sel)[:, 1]
            y_pred = model.predict(X_test_sel)
            
            # evaulation metric
            auc = roc_auc_score(y_test, y_pred_proba)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            sen = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            acc = (tp + tn) / (tp + tn + fp + fn)
            
            test_metrics = {
                'auc': auc,
                'sensitivity': sen,
                'specificity': spec,
                'accuracy': acc,
                'y_true': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            return test_metrics, model
            
        except Exception as e:
            print(f"    Test evaluation failed: {str(e)}")
            return None, None

    def run_repeated_cv(self):
        """Repeated Stratified K-Fold CV"""
        print(f"\n{'='*70}")
        print(f"Starting Repeated {self.n_splits}-Fold CV ({self.n_repeats} repeats)")
        print(f"{'='*70}")
        
        rskf = RepeatedStratifiedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=42
        )
        
        fold_idx = 0
        for train_idx, test_idx in rskf.split(self.X_dev, self.y_dev):
            fold_idx += 1
            
            print(f"\n{'='*70}")
            print(f"FOLD {fold_idx}/{self.n_splits * self.n_repeats}")
            print(f"{'='*70}")
            
            X_train = self.X_dev[train_idx]
            X_test = self.X_dev[test_idx]
            y_train = self.y_dev[train_idx]
            y_test = self.y_dev[test_idx]
            
            actual_train_idx = self.dev_idx[train_idx]
            actual_test_idx = self.dev_idx[test_idx]
            
            print(f"Train: {len(X_train)} (class: {np.bincount(y_train)})")
            print(f"Test: {len(X_test)} (class: {np.bincount(y_test)})")
            
            # 1) RF feautre selection
            print(f"\n[1] Random Forest Feature Selection")
            rf_panels = self.run_random_forest_fold(X_train, y_train, fold_idx)
            
            if len(rf_panels) == 0:
                print(f"    RF failed, skipping fold")
                continue
            
            # 2) gene frequencies
            print(f"\n[2] Computing Gene Frequencies")
            gene_freq = compute_frequencies(rf_panels)
            print(f"    Unique genes: {len(gene_freq)}")
            
            # 3) robust feature selection
            print(f"\n[3] Robust Feature Selection")
            selected_idx, model = self.select_features_robust(
                X_train, y_train, gene_freq, actual_train_idx
            )
            
            if len(selected_idx) == 0:
                print(f"    No genes selected, skipping fold")
                continue
            
            selected_genes = self.gene_names[selected_idx]
            print(f"    Selected genes: {list(selected_genes)}")
            
            # 4) Evaluation
            print(f"\n[4] Evaluating on Test Set")
            test_metrics, fitted_model = self.evaluate_on_test(
                X_train, y_train, X_test, y_test,
                selected_idx, actual_train_idx, actual_test_idx,
                cv=min(3, len(X_train)//10)
            )
            
            if test_metrics is not None:
                print(f"    AUC: {test_metrics['auc']:.4f}")
                print(f"    Sensitivity: {test_metrics['sensitivity']:.4f}")
                print(f"    Specificity: {test_metrics['specificity']:.4f}")
                print(f"    Accuracy: {test_metrics['accuracy']:.4f}")
            
            fold_result = {
                'fold': fold_idx,
                'train_idx': actual_train_idx,
                'test_idx': actual_test_idx,
                'gene_freq': gene_freq,
                'selected_idx': selected_idx,
                'selected_genes': selected_genes,
                'fitted_model': fitted_model,
                'test_metrics': test_metrics,
            }
            
            self.cv_results.append(fold_result)
        
        print(f"\n{'='*70}")
        print(f"Repeated CV Complete!")
        print(f"{'='*70}")
        
        self.aggregate_results()
        
        return self.cv_results

    def aggregate_results(self):
        print(f"\n{'='*70}")
        print(f"AGGREGATING RESULTS")
        print(f"{'='*70}")
        
        all_selected_genes = []
        for fold in self.cv_results:
            if fold['selected_genes'] is not None:
                all_selected_genes.extend(fold['selected_genes'])
        
        if len(all_selected_genes) == 0:
            print("No genes selected across folds!")
            return
        
        gene_selection_freq = pd.Series(all_selected_genes).value_counts()
        
        self.min_folds = max(2, int(np.ceil(len(self.cv_results) * 0.3)))
        stable_genes = gene_selection_freq[gene_selection_freq >= self.min_folds]
        
        print(f"\nGenes selected in >= {self.min_folds}/{len(self.cv_results)} folds: {len(stable_genes)}")
        print(f"Top genes by frequency:")
        for gene, freq in stable_genes.head(15).items():
            print(f"  {gene}: {freq} times")
        
        self.final_selected_genes = np.array(stable_genes.index)
        
        test_metrics_list = [f['test_metrics'] for f in self.cv_results 
                            if f['test_metrics'] is not None]
        if len(test_metrics_list) == 0:
            return

        n_folds_per_repeat = self.n_splits
        n_repeats = self.n_repeats

        repeat_metrics = {
            'auc': [],
            'sensitivity': [],
            'specificity': [],
            'accuracy': []
        }
        print(f"\n{'='*70}")
        print(f"PERFORMANCE BY REPEAT")
        print(f"{'='*70}")

        for repeat_idx in range(n_repeats):
            start_idx = repeat_idx * n_folds_per_repeat
            end_idx = start_idx + n_folds_per_repeat

            repeat_folds = self.cv_results[start_idx:end_idx]
            repeat_test_metrics = [f['test_metrics'] for f in repeat_folds 
                                  if f['test_metrics'] is not None]
            
            if len(repeat_test_metrics) == 0:
                continue
            
            repeat_auc = np.mean([m['auc'] for m in repeat_test_metrics])
            repeat_sen = np.mean([m['sensitivity'] for m in repeat_test_metrics])
            repeat_spe = np.mean([m['specificity'] for m in repeat_test_metrics])
            repeat_acc = np.mean([m['accuracy'] for m in repeat_test_metrics])
            
            repeat_metrics['auc'].append(repeat_auc)
            repeat_metrics['sensitivity'].append(repeat_sen)
            repeat_metrics['specificity'].append(repeat_spe)
            repeat_metrics['accuracy'].append(repeat_acc)
            
            print(f"\nRepeat {repeat_idx + 1}:")
            print(f"  Successful folds: {len(repeat_test_metrics)}/{n_folds_per_repeat}")
            print(f"  AUC:         {repeat_auc:.4f}")
            print(f"  Sensitivity: {repeat_sen:.4f}")
            print(f"  Specificity: {repeat_spe:.4f}")
            print(f"  Accuracy:    {repeat_acc:.4f}")
        

        print(f"\n{'='*70}")
        print(f"OVERALL PERFORMANCE (Across Repeats)")
        print(f"{'='*70}")
        print(f"Number of repeats: {len(repeat_metrics['auc'])}")
        
        mean_auc = np.mean(repeat_metrics['auc'])
        std_auc = np.std(repeat_metrics['auc'], ddof=1)  
        se_auc = std_auc / np.sqrt(len(repeat_metrics['auc']))  
        
        mean_sen = np.mean(repeat_metrics['sensitivity'])
        std_sen = np.std(repeat_metrics['sensitivity'], ddof=1)
        se_sen = std_sen / np.sqrt(len(repeat_metrics['sensitivity']))
        
        mean_spe = np.mean(repeat_metrics['specificity'])
        std_spe = np.std(repeat_metrics['specificity'], ddof=1)
        se_spe = std_spe / np.sqrt(len(repeat_metrics['specificity']))
        
        mean_acc = np.mean(repeat_metrics['accuracy'])
        std_acc = np.std(repeat_metrics['accuracy'], ddof=1)
        se_acc = std_acc / np.sqrt(len(repeat_metrics['accuracy']))
        
        print(f"\nAUC:         {mean_auc:.4f} ± {std_auc:.4f} (SE: {se_auc:.4f})")
        print(f"Sensitivity: {mean_sen:.4f} ± {std_sen:.4f} (SE: {se_sen:.4f})")
        print(f"Specificity: {mean_spe:.4f} ± {std_spe:.4f} (SE: {se_spe:.4f})")
        print(f"Accuracy:    {mean_acc:.4f} ± {std_acc:.4f} (SE: {se_acc:.4f})")
        
        from scipy import stats
        n = len(repeat_metrics['auc'])
        t_critical = stats.t.ppf(0.975, n - 1)  
        
        ci_auc = (mean_auc - t_critical * se_auc, mean_auc + t_critical * se_auc)
        ci_sen = (mean_sen - t_critical * se_sen, mean_sen + t_critical * se_sen)
        ci_spe = (mean_spe - t_critical * se_spe, mean_spe + t_critical * se_spe)
        ci_acc = (mean_acc - t_critical * se_acc, mean_acc + t_critical * se_acc)
        
        print(f"\n95% Confidence Intervals:")
        print(f"AUC:         [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}]")
        print(f"Sensitivity: [{ci_sen[0]:.4f}, {ci_sen[1]:.4f}]")
        print(f"Specificity: [{ci_spe[0]:.4f}, {ci_spe[1]:.4f}]")
        print(f"Accuracy:    [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]")
        
        self.cv_mean_metrics = {
            'auc': (mean_auc, std_auc, se_auc, ci_auc),
            'sensitivity': (mean_sen, std_sen, se_sen, ci_sen),
            'specificity': (mean_spe, std_spe, se_spe, ci_spe),
            'accuracy': (mean_acc, std_acc, se_acc, ci_acc),
            'n_repeats': len(repeat_metrics['auc']),
            'repeat_metrics': repeat_metrics  
        }
        


    def train_final_model(self):
        print(f"\n{'='*70}")
        print(f"TRAINING FINAL MODEL")
        print(f"{'='*70}")
        
        if self.final_selected_genes is None or len(self.final_selected_genes) == 0:
            print("No stable genes found!")
            return None
        
        final_idx = [np.where(self.gene_names == g)[0][0] 
                     for g in self.final_selected_genes]
        
        print(f"Training with {len(final_idx)} genes")
        print(f"Genes: {', '.join(self.final_selected_genes[:10])}...")
        
        X_final = self.X_dev[:, final_idx]
        
        if self.added_coef:
            mcoef_dev = self.mcoef[self.dev_idx]
            X_final = np.concatenate([X_final, mcoef_dev.reshape(-1,1)], axis=1)
        
        self.final_model = LogisticRegressionCV(  
            penalty='elasticnet',
            solver='saga',
            l1_ratios =[self.elastic_net_l1_ratio],
            cv=min(3, len(X_final)//10),
            max_iter=5000,
            random_state=42,
            scoring='roc_auc'
        )
        
        self.final_model.fit(X_final, self.y_dev)
        
        print(f"Final model trained successfully")
        
        return self.final_model

    def evaluate_on_holdout(self):
        """Holdout"""
        if not self.use_holdout or self.X_holdout is None:
            print("\nNo holdout set available")
            return None
        
        print(f"\n{'='*70}")
        print(f"HOLDOUT SET EVALUATION")
        print(f"{'='*70}")
        
        if self.final_model is None:
            print("Training final model first...")
            self.train_final_model()
        
        if self.final_model is None:
            return None
        
        final_idx = [np.where(self.gene_names == g)[0][0] 
                     for g in self.final_selected_genes]
        
        X_holdout_final = self.X_holdout[:, final_idx]
        
        if self.added_coef:
            mcoef_holdout = self.mcoef[self.holdout_idx]
            X_holdout_final = np.concatenate([X_holdout_final, mcoef_holdout.reshape(-1,1)], axis=1)
        
        try:
            y_pred_proba = self.final_model.predict_proba(X_holdout_final)[:, 1]
            y_pred = self.final_model.predict(X_holdout_final)
            
            auc = roc_auc_score(self.y_holdout, y_pred_proba)
            tn, fp, fn, tp = confusion_matrix(self.y_holdout, y_pred).ravel()
            sen = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            acc = (tp + tn) / (tp + tn + fp + fn)
            
            self.holdout_performance = {
                'auc': auc,
                'sensitivity': sen,
                'specificity': spec,
                'accuracy': acc,
                'y_true': self.y_holdout,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"\nHoldout Performance:")
            print(f"  AUC:         {auc:.4f}")
            print(f"  Sensitivity: {sen:.4f}")
            print(f"  Specificity: {spec:.4f}")
            print(f"  Accuracy:    {acc:.4f}")
            
            return self.holdout_performance
            
        except Exception as e:
            print(f"Holdout evaluation failed: {str(e)}")
            return None

    def run_permutation_test(self, n_permutations=1000):
        """
        Permutation test
        """
        if self.final_selected_genes is None or len(self.final_selected_genes) == 0:
            print("No genes to test")
            return None
        
        print(f"\n{'='*70}")
        print(f"PERMUTATION TEST (n={n_permutations})")
        print(f"{'='*70}")
        
        final_idx = [np.where(self.gene_names == g)[0][0] 
                     for g in self.final_selected_genes]
        
        X_sel = self.X_dev[:, final_idx]
        
        if self.added_coef:
            mcoef_dev = self.mcoef[self.dev_idx]
            X_sel = np.concatenate([X_sel, mcoef_dev.reshape(-1,1)], axis=1)
        
        print("Computing permutation p-values for each gene...")
        
        feature_p_values = permutation_importance_test(
            self.X_dev, self.y_dev, final_idx,
            n_permutations=n_permutations,
            l1_ratio=self.elastic_net_l1_ratio,
            cv=min(3, len(X_sel)//10)
        )
        
        # FDR adjustment
        _, adjusted_p, _, _ = multipletests(feature_p_values, method='fdr_bh')
        
        results_df = pd.DataFrame({
            'Gene': self.final_selected_genes,
            'P-value': feature_p_values,
            'FDR': adjusted_p
        }).sort_values('FDR')
        
        print(f"\nPermutation Test Results:")
        print(results_df.to_string(index=False))
        
        significant_genes = results_df[results_df['FDR'] < 0.1]['Gene'].values
        print(f"\nSignificant genes (FDR < 0.1): {len(significant_genes)}/{len(self.final_selected_genes)}")
        
        self.permutation_results = results_df
        
        return results_df

    def run_complete_pipeline(self, run_permutation=True):
        # 1) Repeated CV
        self.run_repeated_cv()
        
        # 2) Final test
        self.train_final_model()
        
        # 3) Holdout test 
        if self.use_holdout:
            self.evaluate_on_holdout()
        
        # 4) Permutation test 
        if run_permutation and self.final_selected_genes is not None:
            self.run_permutation_test()
        
        # 5) final report
        self.print_final_report()
        
        return {
            'cv_results': self.cv_results,
            'final_genes': self.final_selected_genes,
            'final_model': self.final_model,
            'cv_performance': self.cv_mean_metrics if hasattr(self, 'cv_mean_metrics') else None,
            'holdout_performance': self.holdout_performance,
            'permutation_results': self.permutation_results if hasattr(self, 'permutation_results') else None
        }

    def print_final_report(self):
        print(f"\n{'='*70}")
        print(f"FINAL REPORT")
        print(f"{'='*70}")
        
        print(f"\n1. Dataset Information:")
        print(f"   Total samples: {len(self.X)}")
        print(f"   Development samples: {len(self.X_dev)}")
        if self.use_holdout:
            print(f"   Holdout samples: {len(self.X_holdout)}")
        print(f"   Class distribution: {np.bincount(self.y)}")
        
        if self.final_selected_genes is not None:
            print(f"\n2. Selected Biomarkers:")
            print(f"   Total: {len(self.final_selected_genes)}")
            print(f"   Genes: {', '.join(self.final_selected_genes)}")
        
        if hasattr(self, 'cv_mean_metrics'):
            print(f"\n3. Cross-Validation Performance:")
            for metric, value in self.cv_mean_metrics.items():
                if metric in ['auc', 'sensitivity', 'specificity', 'accuracy']:
                    mean, std, se, ci =  value
                    print(f"   {metric.capitalize()}: {mean:.4f} ± {std:.4f}")
                else:
                    continue
        
        if self.holdout_performance is not None:
            print(f"\n4. Holdout Performance:")
            print(f"   AUC: {self.holdout_performance['auc']:.4f}")
            print(f"   Sensitivity: {self.holdout_performance['sensitivity']:.4f}")
            print(f"   Specificity: {self.holdout_performance['specificity']:.4f}")
            print(f"   Accuracy: {self.holdout_performance['accuracy']:.4f}")
        
        if hasattr(self, 'permutation_results'):
            sig_genes = len(self.permutation_results[self.permutation_results['FDR'] < 0.1])
            print(f"\n5. Permutation Test:")
            print(f"   Significant genes (FDR<0.1): {sig_genes}/{len(self.permutation_results)}")
        
        print(f"\n{'='*70}")
        print(f"RECOMMENDATIONS FOR SMALL SAMPLE SIZE:")
        print(f"{'='*70}")
        print(f"1. Validate findings in independent cohort")
        print(f"2. Consider biological relevance of selected genes")
        print(f"3. Report confidence intervals and statistical tests")
        print(f"4. Be cautious about overfitting - these results need validation")
        print(f"5. Consider combining with biological pathway analysis")
        print(f"{'='*70}\n")