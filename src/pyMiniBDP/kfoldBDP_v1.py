import warnings

from typing import Optional, Union, Sequence, Dict, Any, List

import numpy as np
import pandas as pd

from scipy import stats
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

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

    def __init__(self, 
                 adata, 
                 layer: str = "log10", 
                 y_col: str = "Prognosis",
                 positive_label: Optional[str] = "Poor",
                 standardscale: bool = True,
                 n_iter: int = 100,  # RF iteration 
                 rf_selection_size: int = 15,  
                 n_splits: int =5,  # K-fold 
                 n_repeats: int = 5,  # Repeated K-fold
                 added_coef: Optional[Union[str, Sequence[str]]] = None,
                 univariate_threshold: float = 0.05,
                 highfreq_quantile: float = 0.8,
                 correlation_threshold: float = 0.9,
                 sample_feature_min_ratio: float = 5.0,
                 sample_feature_warning_ratio: float = 10.0,
                 elastic_net_l1_ratio: float = 0.5,
                 use_holdout: bool =False,  
                 holdout_size: float =0.2,
                 random_state: int = 42,
                 rf_success_fraction_min: float = 0.8,
                 class_weight: Optional[Union[str, Dict[int, float]]] = "balanced",
                 verbose: bool = True,
                ):  
        
        # Config
        self.adata = adata
        self.layer = layer
        self.y_col = y_col
        self.standardscale = standardscale
        self.n_iter = n_iter
        self.rf_selection_size = rf_selection_size
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.added_coef = added_coef
        
        # Feature selection 
        self.univariate_threshold = univariate_threshold
        self.highfreq_quantile = highfreq_quantile
        self.correlation_threshold = correlation_threshold
        self.sample_feature_min_ratio = sample_feature_min_ratio
        self.sample_feature_warning_ratio = sample_feature_warning_ratio

        self.elastic_net_l1_ratio = elastic_net_l1_ratio

        self.use_holdout = use_holdout
        self.holdout_size = holdout_size
        self.random_state = random_state
        self.rf_success_fraction_min = rf_success_fraction_min
        self.class_weight = class_weight
        self.verbose = verbose
        
        # Raw molecular feature matrix. Do not scale here.
        X = adata.layers[layer]
        if issparse(X):
            X = X.toarray()
        self.X_raw = X
        self.gene_names = np.array(adata.var_names)

        if self.X_raw.shape[1] != len(self.gene_names):
            raise ValueError(
                f"Feature dimension mismatch: X has {self.X_raw.shape[1]} columns, "
                f"but adata.var_names has {len(self.gene_names)} names."
            )
        
        if self.added_coef:
            self.M_raw = adata.obs[self.added_coef].values
        else:
            self.M_raw = None

        y_raw = adata.obs[y_col].astype(str)
        unique_labels = np.unique(y_raw)

        if positive_label is not None and positive_label in unique_labels:
            self.y = (y_raw == positive_label).astype(int)
            self.label_encoder = None
            self.class_names_ = np.array([f"not_{positive_label}", positive_label])
        else:
            le = LabelEncoder()
            self.y = le.fit_transform(y_raw)
            self.label_encoder = le
            self.class_names_ = le.classes_
            if len(self.class_names_) == 2 and self.verbose:
                warnings.warn(
                    f"positive_label={positive_label!r} was not found. "
                    f"Using LabelEncoder order: {self.class_names_[0]}=0, {self.class_names_[1]}=1."
                )
        

        
        n_samples = self.X_raw.shape[0]
        n_features = self.X_raw.shape[1]
        class_counts = np.bincount(self.y)
        min_class_size = np.min(class_counts)
        
        if self.verbose:
            print(f"\n{'=' * 70}")
            print("SAMPLE SIZE ASSESSMENT")
            print(f"{'=' * 70}")
            print(f"Total samples: {n_samples}")
            print(f"Class distribution: {class_counts}")
            print(f"Minimum class size: {min_class_size}")
            print(f"Total protein features: {n_features}")
            if self.M_raw is not None:
                print(f"Additional covariates: {self.added_coef}")
        
        # Optional holdout split occurs before any scaling.
        all_idx = np.arange(n_samples)
        if self.use_holdout and n_samples >= 60:
            (
                self.dev_idx,
                self.holdout_idx,
                self.y_dev,
                self.y_holdout,
            ) = train_test_split(
                all_idx,
                self.y,
                test_size=self.holdout_size,
                stratify=self.y,
                random_state=self.random_state,
            )
            self.X_dev_raw = self.X_raw[self.dev_idx]
            self.X_holdout_raw = self.X_raw[self.holdout_idx]

            if self.M_coef is not None:
                self.M_dev_raw = self.M_raw[self.dev_idx]
                self.M_holdout_raw = self.M_raw[self.holdout_idx]
            else:
                self.M_dev_raw = None
                self.M_holdout_raw = None

            if self.verbose:
                print("\nUsing holdout set:")
                print(f"  Development: {len(self.dev_idx)}")
                print(f"  Holdout:     {len(self.holdout_idx)}")
        else:
            if self.use_holdout and self.verbose:
                print("\nSample size too small for holdout set. Using full repeated CV instead.")
            self.use_holdout = False
            self.dev_idx = all_idx
            self.holdout_idx = None
            self.X_dev_raw = self.X_raw
            self.X_holdout_raw = None
            self.M_dev_raw = self.M_raw
            self.M_holdout_raw = None
            self.y_dev = self.y
            self.y_holdout = None

        if self.verbose:
            print(f"{'=' * 70}\n")

        self.cv_results: List[Dict[str, Any]] = []
        self.final_model = None
        self.final_selected_genes = None
        self.holdout_performance = None
        self.permutation_results = None

        self.final_x_scaler = None
        self.final_m_scaler = None
        self.cv_mean_metrics = None


    def _scale_train_test(self, X_train_raw, X_test_raw, M_train_raw=None, M_test_raw=None):
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train_raw)
        X_test = x_scaler.transform(X_test_raw)

        if M_train_raw is not None:
            m_scaler = StandardScaler()
            M_train = m_scaler.fit_transform(M_train_raw.reshape(-1,1))
            M_test = m_scaler.transform(M_test_raw.reshape(-1,1))
        else:
            m_scaler = None
            M_train = None
            M_test = None

        return X_train, X_test, M_train, M_test, x_scaler, m_scaler


    def run_random_forest_fold(self, X_train, y_train, M_train, fold_idx):
        rf_panels = []
        failure_messages = []

        for i in tqdm(range(self.n_iter), desc=f"Fold {fold_idx} - RF", leave=False):
            #Bootstrap sampling
            rng = np.random.default_rng(self.random_state + fold_idx * 10000 + i)
            boot_idx = rng.choice(len(X_train), size=len(X_train), replace=True)
            X_boot = X_train[boot_idx]
            y_boot = y_train.iloc[boot_idx]

            if len(np.unique(y_boot)) < 2:
                failure_messages.append("bootstrap sample contained one class")
                continue

            if M_train is not None:
                M_boot = M_train[boot_idx]
                X_boot_model = append_covariates(X_boot, M_boot)
            else:
                X_boot_model = X_boot

            rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=max(2, len(X_train) // 20),
                min_samples_split=max(5, len(X_train) // 10),
                random_state=self.random_state + fold_idx * 10000 + i,
                class_weight=self.class_weight,
            )

            try:
                rf.fit(X_boot_model, y_boot)

                importances = rf.feature_importances_

                # Exclude optional covariates from protein ranking.
                if M_train is not None:
                    importances = importances[: self.X_raw.shape[1]]

                idx = np.argsort(importances)[::-1]
                top_genes = self.gene_names[idx[: self.rf_selection_size]]
                rf_panels.append(top_genes.tolist())

            except Exception as e:
                failure_messages.append(str(e))
                continue

        success_count = len(rf_panels)
        min_success = int(np.ceil(self.n_iter * self.rf_success_fraction_min))

        if success_count < min_success:
            if self.verbose:
                print(f"    RF success count too low: {success_count}/{self.n_iter}")
                if failure_messages:
                    print(f"    Example RF failure: {failure_messages[0]}")
            return np.array([])

        if self.verbose and failure_messages:
            print(f"    RF completed with {success_count}/{self.n_iter} successful iterations.")
            print(f"    RF failures ignored: {len(failure_messages)}")

        return np.array(rf_panels)

    def select_features_robust(self, X_train, y_train, M_train, gene_freq, fold_idx):
        if self.verbose:
            print("    Starting robust feature selection...")
            
        high_freq = get_highfreq(gene_freq, self.highfreq_quantile, self.rf_selection_size)
        if len(high_freq) == 0:
            if self.verbose:
                print("    No high-frequency proteins found.")
            return [], None

        idx = [np.where(self.gene_names == g)[0][0] for g in high_freq.index]

        if self.verbose:
            print(f"    High-frequency proteins (>={self.highfreq_quantile} quantile): {len(idx)}")

        status, ratio = check_sample_adequacy(len(X_train), len(idx), 
                                              self.sample_feature_min_ratio, self.sample_feature_warning_ratio)
        if self.verbose:
            print(f"    Sample/feature ratio: {ratio:.2f} ({status})")

        if status in {"critical", "warning"}:
            if self.verbose:
                print("    Applying univariate filtering...")
            X_subset = X_train[:, idx]
            univar_idx, adj_p = univariate_filter(X_subset, y_train, self.univariate_threshold)

            if len(univar_idx) > 10:
                idx = [idx[i] for i in univar_idx]
                if self.verbose:
                    print(f"    After univariate filtering: {len(idx)} proteins")
            else:
                best_idx = np.argsort(adj_p)[: min(10, len(adj_p))]
                idx = [idx[i] for i in best_idx]
                if self.verbose:
                    print(
                        f"    Fewer than 10 proteins passed threshold; "
                        f"using top {len(idx)} by FDR-adjusted p value"
                    )

        # Correlation filtering.
        # Important: current_genes is recomputed after univariate filtering.
        if len(idx) > 1:
            X_subset = X_train[:, idx]
            correlation_matrix = np.corrcoef(X_subset, rowvar=False)

            current_genes = self.gene_names[idx]
            to_remove = set()

            for i in range(len(correlation_matrix)):
                for j in range(i + 1, len(correlation_matrix)):
                    corr_ij = correlation_matrix[i, j]
                    if np.isnan(corr_ij):
                        continue

                    if abs(corr_ij) >= self.correlation_threshold:
                        gene_i = current_genes[i]
                        gene_j = current_genes[j]

                        freq_i = gene_freq.get(gene_i, 0)
                        freq_j = gene_freq.get(gene_j, 0)

                        if freq_i < freq_j:
                            to_remove.add(idx[i])
                        elif freq_j < freq_i:
                            to_remove.add(idx[j])
                        else:
                            to_remove.add(idx[j])

            idx = [i for i in idx if i not in to_remove]

            if self.verbose:
                print(f"    After correlation filtering: {len(idx)} proteins")

        status, ratio = check_sample_adequacy(len(X_train), len(idx),
                                             self.sample_feature_min_ratio, self.sample_feature_warning_ratio)

        if ratio < self.sample_feature_min_ratio and len(idx) > 5:
            max_features = max(
                5,
                len(X_train) // int(np.ceil(self.sample_feature_min_ratio))
            )
            idx = idx[:max_features]
            if self.verbose:
                print(
                    f"    Reduced to {len(idx)} proteins for sample/feature safety "
                    f"(ratio={len(X_train) / len(idx):.2f})"
                )

        model = None
        if len(idx) > 0:
            if self.verbose:
                print("    Applying Elastic Net...")

            X_subset = X_train[:, idx]
            X_model = append_covariates(X_subset, M_train)

            inner_cv = safe_inner_cv(y_train, max_cv=3)

            try:
                selected_mask, model = fit_elastic_net(
                    X_model,
                    y_train,
                    l1_ratio=self.elastic_net_l1_ratio,
                    cv=inner_cv,
                    random_state=self.random_state + fold_idx,
                    class_weight=self.class_weight,
                )

                # selected_mask indexes X_model, which may include covariates.
                # Keep only selected protein columns.
                selected_protein_mask = [m for m in selected_mask if m < len(idx)]

                if len(selected_protein_mask) > 0:
                    idx = [idx[i] for i in selected_protein_mask]
                    if self.verbose:
                        print(f"    After Elastic Net: {len(idx)} proteins")
                else:
                    idx = idx[: min(5, len(idx))]
                    if self.verbose:
                        print(
                            f"    Elastic Net selected no protein; "
                            f"keeping top {len(idx)} proteins"
                        )

            except Exception as e:
                if self.verbose:
                    print(f"    Elastic Net failed: {str(e)}")
                    print(f"    Keeping current {len(idx)} proteins")
                model = None

        if self.verbose:
            print(f"    Final selected proteins: {len(idx)}")

        return idx, model

    #Evalutate test
    def evaluate_on_test(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        M_train,
        M_test,
        selected_idx,
        fold_idx,
    ):
        if len(selected_idx) == 0:
            return None, None

        X_train_sel = X_train[:, selected_idx]
        X_test_sel = X_test[:, selected_idx]

        X_train_model = append_covariates(X_train_sel, M_train)
        X_test_model = append_covariates(X_test_sel, M_test)

        inner_cv = safe_inner_cv(y_train, max_cv=3)

        try:
            _, model = fit_elastic_net(
                X_train_model,
                y_train,
                l1_ratio=self.elastic_net_l1_ratio,
                cv=inner_cv,
                random_state=self.random_state + fold_idx,
                class_weight=self.class_weight,
            )
            #predict
            y_pred_proba = model.predict_proba(X_test_model)[:, 1]
            #y_pred = (y_pred_proba >= 0.5).astype(int)
            y_pred = model.predict(X_test_sel)
            
            auc = roc_auc_score(y_test, y_pred_proba)

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            accuracy = (tp + tn) / np.sum(cm) if np.sum(cm) > 0 else np.nan

            test_metrics = {
                "auc": auc,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "accuracy": accuracy,
                "y_true": y_test.copy(),
                "y_pred": y_pred,
                "y_pred_proba": y_pred_proba,
            }

            return test_metrics, model

        except Exception as e:
            if self.verbose:
                print(f"    Test evaluation failed: {str(e)}")
            return None, None

    def run_repeated_cv(self):
        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"Starting Repeated {self.n_splits}-Fold CV ({self.n_repeats} repeats)")
            print(f"{'=' * 70}")

        rskf = RepeatedStratifiedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )

        self.cv_results = []
        fold_idx = 0

        for train_idx, test_idx in rskf.split(self.X_dev_raw, self.y_dev):
            fold_idx += 1

            if self.verbose:
                print(f"\n{'=' * 70}")
                print(f"FOLD {fold_idx}/{self.n_splits * self.n_repeats}")
                print(f"{'=' * 70}")

            X_train_raw = self.X_dev_raw[train_idx]
            X_test_raw = self.X_dev_raw[test_idx]
            y_train = self.y_dev[train_idx]
            y_test = self.y_dev[test_idx]

            M_train_raw = self.M_dev_raw[train_idx] if self.M_dev_raw is not None else None
            M_test_raw = self.M_dev_raw[test_idx] if self.M_dev_raw is not None else None

            # Fit scalers on training fold only.
            X_train, X_test, M_train, M_test, x_scaler, m_scaler = self._scale_train_test(
                X_train_raw,
                X_test_raw,
                M_train_raw,
                M_test_raw,
            )

            actual_train_idx = self.dev_idx[train_idx]
            actual_test_idx = self.dev_idx[test_idx]

            if self.verbose:
                print(f"Train: {len(X_train)} (class: {np.bincount(y_train)})")
                print(f"Test:  {len(X_test)} (class: {np.bincount(y_test)})")

            if self.verbose:
                print("\n[1] Random Forest Feature Selection")

            rf_panels = self.run_random_forest_fold(
                X_train=X_train,
                y_train=y_train,
                M_train=M_train,
                fold_idx=fold_idx,
            )

            if len(rf_panels) == 0:
                if self.verbose:
                    print("    RF failed or too few successful iterations; skipping fold.")
                continue

            if self.verbose:
                print("\n[2] Computing Protein Frequencies")

            gene_freq = compute_frequencies(rf_panels)

            if self.verbose:
                print(f"    Unique proteins: {len(gene_freq)}")

            if self.verbose:
                print("\n[3] Robust Feature Selection")

            selected_idx, selection_model = self.select_features_robust(
                X_train=X_train,
                y_train=y_train,
                M_train=M_train,
                gene_freq=gene_freq,
                fold_idx=fold_idx,
            )

            if len(selected_idx) == 0:
                if self.verbose:
                    print("    No proteins selected; skipping fold.")
                continue

            selected_genes = self.gene_names[selected_idx]

            if self.verbose:
                print(f"    Selected proteins: {list(selected_genes)}")

            if self.verbose:
                print("\n[4] Evaluating on Test Fold")

            test_metrics, fitted_model = self.evaluate_on_test(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                M_train=M_train,
                M_test=M_test,
                selected_idx=selected_idx,
                fold_idx=fold_idx,
            )

            if test_metrics is not None and self.verbose:
                print(f"    AUC:         {test_metrics['auc']:.4f}")
                print(f"    Sensitivity: {test_metrics['sensitivity']:.4f}")
                print(f"    Specificity: {test_metrics['specificity']:.4f}")
                print(f"    Accuracy:    {test_metrics['accuracy']:.4f}")

            fold_result = {
                "fold": fold_idx,
                "train_idx": actual_train_idx,
                "test_idx": actual_test_idx,
                "gene_freq": gene_freq,
                "selected_idx": selected_idx,
                "selected_genes": selected_genes,
                "selection_model": selection_model,
                "fitted_model": fitted_model,
                "x_scaler": x_scaler,
                "m_scaler": m_scaler,
                "test_metrics": test_metrics,
            }

            self.cv_results.append(fold_result)

        if self.verbose:
            print(f"\n{'=' * 70}")
            print("Repeated CV Complete")
            print(f"{'=' * 70}")

        self.aggregate_results()
        return self.cv_results

    def aggregate_results(self):
        if self.verbose:
            print(f"\n{'=' * 70}")
            print("AGGREGATING RESULTS")
            print(f"{'=' * 70}")

        all_selected_genes = []
        for fold in self.cv_results:
            if fold.get("selected_genes") is not None:
                all_selected_genes.extend(list(fold["selected_genes"]))

        if len(all_selected_genes) == 0:
            if self.verbose:
                print("No proteins selected across folds.")
            return

        gene_selection_freq = pd.Series(all_selected_genes).value_counts()

        successful_folds = len(self.cv_results)
        self.min_folds = max(2, int(np.ceil(successful_folds * 0.3)))
        stable_genes = gene_selection_freq[gene_selection_freq >= self.min_folds]

        self.gene_selection_freq = gene_selection_freq
        self.final_selected_genes = np.array(stable_genes.index)

        if self.verbose:
            print(
                f"\nProteins selected in >= {self.min_folds}/"
                f"{successful_folds} folds: {len(stable_genes)}"
            )
            print("Top proteins by frequency:")
            for gene, freq in stable_genes.head(15).items():
                print(f"  {gene}: {freq} times")

        repeat_metrics = {
            "auc": [],
            "sensitivity": [],
            "specificity": [],
            "accuracy": [],
        }

        if self.verbose:
            print(f"\n{'=' * 70}")
            print("PERFORMANCE BY REPEAT")
            print(f"{'=' * 70}")

        for repeat_idx in range(self.n_repeats):
            start_fold = repeat_idx * self.n_splits + 1
            end_fold = (repeat_idx + 1) * self.n_splits

            repeat_folds = [
                f for f in self.cv_results
                if start_fold <= f["fold"] <= end_fold
                and f.get("test_metrics") is not None
            ]

            if len(repeat_folds) == 0:
                continue

            repeat_test_metrics = [f["test_metrics"] for f in repeat_folds]

            for metric in repeat_metrics:
                repeat_metrics[metric].append(
                    float(np.nanmean([m[metric] for m in repeat_test_metrics]))
                )

            if self.verbose:
                print(f"\nRepeat {repeat_idx + 1}:")
                print(f"  Successful folds: {len(repeat_test_metrics)}/{self.n_splits}")
                print(f"  AUC:         {repeat_metrics['auc'][-1]:.4f}")
                print(f"  Sensitivity: {repeat_metrics['sensitivity'][-1]:.4f}")
                print(f"  Specificity: {repeat_metrics['specificity'][-1]:.4f}")
                print(f"  Accuracy:    {repeat_metrics['accuracy'][-1]:.4f}")

        n = len(repeat_metrics["auc"])
        if n == 0:
            if self.verbose:
                print("No successful repeats.")
            return

        summary = {}
        for metric, values in repeat_metrics.items():
            values = np.asarray(values, dtype=float)
            mean = float(np.nanmean(values))
            std = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            se = float(std / np.sqrt(len(values))) if len(values) > 1 else 0.0

            if len(values) > 1:
                t_critical = stats.t.ppf(0.975, len(values) - 1)
                ci = (mean - t_critical * se, mean + t_critical * se)
            else:
                ci = (np.nan, np.nan)

            summary[metric] = (mean, std, se, ci)

        summary["n_repeats"] = n
        summary["repeat_metrics"] = repeat_metrics
        self.cv_mean_metrics = summary

        if self.verbose:
            print(f"\n{'=' * 70}")
            print("OVERALL PERFORMANCE")
            print(f"{'=' * 70}")
            print(f"Number of successful repeats: {n}")

            for metric in ["auc", "sensitivity", "specificity", "accuracy"]:
                mean, std, se, ci = summary[metric]
                print(f"{metric.capitalize()}: {mean:.4f} ± {std:.4f} (SE: {se:.4f})")
                if not np.isnan(ci[0]):
                    print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    def train_final_model(self):
        if self.verbose:
            print(f"\n{'=' * 70}")
            print("TRAINING FINAL MODEL")
            print(f"{'=' * 70}")

        if self.final_selected_genes is None or len(self.final_selected_genes) == 0:
            if self.verbose:
                print("No stable proteins found.")
            return None

        final_idx = [
            np.where(self.gene_names == g)[0][0]
            for g in self.final_selected_genes
        ]

        # Fit scaler on full development set only.
        self.final_x_scaler = StandardScaler()
        X_dev = self.final_x_scaler.fit_transform(self.X_dev_raw)

        if self.M_dev_raw is not None:
            self.final_m_scaler = StandardScaler()
            M_dev = self.final_m_scaler.fit_transform(self.M_dev_raw.reshape(-1,1))
        else:
            self.final_m_scaler = None
            M_dev = None

        X_final = X_dev[:, final_idx]
        X_model = append_covariates(X_final, M_dev)

        inner_cv = safe_inner_cv(self.y_dev, max_cv=3)

        try:
            _, self.final_model = fit_elastic_net(
                X_model,
                self.y_dev,
                l1_ratio=self.elastic_net_l1_ratio,
                cv=inner_cv,
                random_state=self.random_state,
                class_weight=self.class_weight,
            )
        except Exception as e:
            if self.verbose:
                print(f"Final LogisticRegressionCV failed: {str(e)}")
            self.final_model = None
            return None

        if self.verbose:
            print(f"Final model trained with {len(final_idx)} protein features.")
            print(f"Proteins: {', '.join(self.final_selected_genes)}")
            if self.M_raw is not None:
                print(f"Covariates included in model: {list(self.added_coef)}")

        return self.final_model

    def evaluate_on_holdout(self):
        if not self.use_holdout or self.X_holdout_raw is None:
            if self.verbose:
                print("\nNo holdout set available.")
            return None

        if self.final_model is None:
            self.train_final_model()

        if self.final_model is None:
            return None

        final_idx = [
            np.where(self.gene_names == g)[0][0]
            for g in self.final_selected_genes
        ]

        # Transform holdout using development-set scalers.
        X_holdout = self.final_x_scaler.transform(self.X_holdout_raw)
        X_holdout_final = X_holdout[:, final_idx]

        if self.M_holdout_raw is not None:
            M_holdout = self.final_m_scaler.transform(self.M_holdout_raw)
        else:
            M_holdout = None

        X_holdout_model = append_covariates(X_holdout_final, M_holdout)

        try:
            y_pred_proba = self.final_model.predict_proba(X_holdout_model)[:, 1]
            y_pred = model.predict(X_holdout_model)

            auc = roc_auc_score(self.y_holdout, y_pred_proba)
            cm = confusion_matrix(self.y_holdout, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            accuracy = (tp + tn) / np.sum(cm) if np.sum(cm) > 0 else np.nan

            self.holdout_performance = {
                "auc": auc,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "accuracy": accuracy,
                "y_true": self.y_holdout.copy(),
                "y_pred": y_pred,
                "y_pred_proba": y_pred_proba,
            }

            if self.verbose:
                print(f"\n{'=' * 70}")
                print("HOLDOUT SET EVALUATION")
                print(f"{'=' * 70}")
                print(f"  AUC:         {auc:.4f}")
                print(f"  Sensitivity: {sensitivity:.4f}")
                print(f"  Specificity: {specificity:.4f}")
                print(f"  Accuracy:    {accuracy:.4f}")

            return self.holdout_performance

        except Exception as e:
            if self.verbose:
                print(f"Holdout evaluation failed: {str(e)}")
            return None

    def run_permutation_test(self, n_permutations=1000):
        """
        Secondary permutation-based robustness assessment.

        This is not an external validation test because final proteins were
        selected using the same development data.
        """
        if self.final_selected_genes is None or len(self.final_selected_genes) == 0:
            if self.verbose:
                print("No proteins to test.")
            return None

        if self.final_model is None:
            self.train_final_model()

        final_idx = [
            np.where(self.gene_names == g)[0][0]
            for g in self.final_selected_genes
        ]

        if self.final_x_scaler is None:
            self.final_x_scaler = StandardScaler().fit(self.X_dev_raw)

        X_dev = self.final_x_scaler.transform(self.X_dev_raw)
        X_sel = X_dev[:, final_idx]

        if self.M_dev_raw is not None:
            if self.final_m_scaler is None:
                self.final_m_scaler = StandardScaler().fit(self.M_dev_raw)
            M_dev = self.final_m_scaler.transform(self.M_dev_raw)
        else:
            M_dev = None

        X_model = append_covariates(X_sel, M_dev)

        observed_coef = np.abs(np.ravel(self.final_model.coef_)[: len(final_idx)])

        rng = np.random.default_rng(self.random_state)
        null_coef = np.zeros((n_permutations, len(final_idx)), dtype=float)

        inner_cv = safe_inner_cv(self.y_dev, max_cv=3)

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"PERMUTATION TEST (n={n_permutations})")
            print(f"{'=' * 70}")

        for b in tqdm(range(n_permutations), desc="Permutation", leave=False):
            y_perm = rng.permutation(self.y_dev)

            try:
                _, perm_model = fit_elastic_net(
                    X_model,
                    y_perm,
                    l1_ratio=self.elastic_net_l1_ratio,
                    cv=inner_cv,
                    random_state=self.random_state + b + 100000,
                    class_weight=self.class_weight,
                )
                null_coef[b, :] = np.abs(np.ravel(perm_model.coef_)[: len(final_idx)])
            except Exception:
                null_coef[b, :] = np.nan

        p_values = []
        for j in range(len(final_idx)):
            null_j = null_coef[:, j]
            null_j = null_j[~np.isnan(null_j)]

            if len(null_j) == 0:
                p = np.nan
            else:
                p = (1 + np.sum(null_j >= observed_coef[j])) / (len(null_j) + 1)

            p_values.append(p)

        p_values = np.asarray(p_values, dtype=float)

        valid = ~np.isnan(p_values)
        adjusted_p = np.full_like(p_values, fill_value=np.nan, dtype=float)

        if np.any(valid):
            _, adjusted_p_valid, _, _ = multipletests(p_values[valid], method="fdr_bh")
            adjusted_p[valid] = adjusted_p_valid

        results_df = pd.DataFrame({
            "Protein": self.final_selected_genes,
            "Observed_abs_coef": observed_coef,
            "P_value": p_values,
            "FDR": adjusted_p,
        }).sort_values("FDR")

        self.permutation_results = results_df

        if self.verbose:
            print("\nPermutation Test Results:")
            print(results_df.to_string(index=False))
            sig = results_df[results_df["FDR"] < 0.1]
            print(f"\nPermutation-supported proteins (FDR < 0.1): {len(sig)}/{len(results_df)}")

        return results_df

    def run_complete_pipeline(self, run_permutation=True, n_permutations=1000):
        self.run_repeated_cv()
        self.train_final_model()

        if self.use_holdout:
            self.evaluate_on_holdout()

        if run_permutation and self.final_selected_genes is not None:
            self.run_permutation_test(n_permutations=n_permutations)

        self.print_final_report()

        return {
            "cv_results": self.cv_results,
            "final_genes": self.final_selected_genes,
            "final_model": self.final_model,
            "cv_performance": self.cv_mean_metrics,
            "holdout_performance": self.holdout_performance,
            "permutation_results": self.permutation_results,
        }

    def print_final_report(self):
        print(f"\n{'=' * 70}")
        print("FINAL REPORT")
        print(f"{'=' * 70}")

        print("\n1. Dataset Information:")
        print(f"   Total samples:       {len(self.X_raw)}")
        print(f"   Development samples: {len(self.X_dev_raw)}")
        if self.use_holdout:
            print(f"   Holdout samples:     {len(self.X_holdout_raw)}")
        print(f"   Class distribution:  {np.bincount(self.y)}")
        print(f"   Class names:         {list(self.class_names_)}")

        if self.M_raw is not None:
            print(f"   Covariates:          {list(self.added_coef)}")

        if self.final_selected_genes is not None:
            print("\n2. Selected Biomarkers:")
            print(f"   Total: {len(self.final_selected_genes)}")
            print(f"   Proteins: {', '.join(self.final_selected_genes)}")
        else:
            print("\n2. Selected Biomarkers:")
            print("   None")

        if self.cv_mean_metrics is not None:
            print("\n3. Cross-Validation Performance:")
            for metric in ["auc", "sensitivity", "specificity", "accuracy"]:
                mean, std, se, ci = self.cv_mean_metrics[metric]
                print(f"   {metric.capitalize()}: {mean:.4f} ± {std:.4f}")
                if not np.isnan(ci[0]):
                    print(f"      95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

        if self.holdout_performance is not None:
            print("\n4. Holdout Performance:")
            print(f"   AUC:         {self.holdout_performance['auc']:.4f}")
            print(f"   Sensitivity: {self.holdout_performance['sensitivity']:.4f}")
            print(f"   Specificity: {self.holdout_performance['specificity']:.4f}")
            print(f"   Accuracy:    {self.holdout_performance['accuracy']:.4f}")

        if self.permutation_results is not None:
            sig_genes = len(self.permutation_results[self.permutation_results["FDR"] < 0.1])
            print("\n5. Permutation Test:")
            print(
                f"   Permutation-supported proteins "
                f"(FDR < 0.1): {sig_genes}/{len(self.permutation_results)}"
            )

        print(f"\n{'=' * 70}")
        print("INTERPRETATION NOTE")
        print(f"{'=' * 70}")
        print("1. Cross-validation estimates are internal resampling performance.")
        print("2. Final biomarkers are internally reproducible candidate biomarkers.")
        print("3. External validation is still required before claiming validated prognostic biomarkers.")
        print(f"{'=' * 70}\n")