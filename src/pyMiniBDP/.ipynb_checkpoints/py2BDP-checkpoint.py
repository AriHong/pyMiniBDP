
import warnings
from typing import Optional, Dict, Any, Sequence

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, *args, **kwargs):
        return x
class BiomarkerPipeline2BDP:
    """
    Reproduction of the original 2BDP workflow using AnnData-style input.

    Statistical workflow retained from the original 2BDP concept:
      1. Repeated 70:30 stratified train/test splitting
      2. Random Forest feature ranking within each training split
      3. Position-specific frequency scoring of RF panels
      4. Selection of top-ranked panels
      5. Generation of nested subpanels of size 2 to panel_size
      6. Panel validation using RSBMR and K-fold cross-validation
      7. Logistic-regression fitting and AUC/sensitivity/specificity reporting

    Important:
      - Feature ranking is performed before downstream validation, following
        the original 2BDP structure.
      - Results are intended for method comparison/internal evaluation rather
        than unbiased external validation.
    """

    def __init__(
        self,
        adata,
        layer: str = "center",
        y_col: str = "Prognosis",
        positive_label: str = "Poor",
        n_iter: int = 2000,
        panel_size: int = 10,
        top_panels: int = 200,
        rf_n_estimators: int = 500,
        rsbmr_repeats: int = 10,
        kfold_splits: int = 10,
        random_state: int = 42,
        class_weight: Optional[str] = None,
        auc_threshold: float = 0.80,
        p_threshold: float = 0.05,
        verbose: bool = True,
    ):
        self.adata = adata
        self.layer = layer
        self.y_col = y_col
        self.positive_label = positive_label

        self.X = self._to_dense(adata.layers[layer]).astype(float)
        self.gene_names = np.asarray(adata.var_names).astype(str)
        self.y = self._encode_binary(adata.obs[y_col], positive_label)

        self.n_iter = int(n_iter)
        self.panel_size = int(panel_size)
        self.top_panels = int(top_panels)
        self.rf_n_estimators = int(rf_n_estimators)
        self.rsbmr_repeats = int(rsbmr_repeats)
        self.kfold_splits = int(kfold_splits)
        self.random_state = int(random_state)
        self.class_weight = class_weight
        self.auc_threshold = float(auc_threshold)
        self.p_threshold = float(p_threshold)
        self.verbose = verbose

        self._validate_inputs()

        self.rf_panels = None
        self.rf_panel_table = None
        self.position_frequency_table = None
        self.gene_frequency_table = None
        self.ranked_panels = None
        self.ranked_panel_table = None
        self.subpanels = None
        self.subpanel_table = None
        self.rsbmr_results = None
        self.kfold_results = None
        self.best_panels = None
        self.summary_table = None

    @staticmethod
    def _to_dense(X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return np.asarray(X)

    @staticmethod
    def _encode_binary(y, positive_label):
        y = pd.Series(y).astype(str)
        labels = y.dropna().unique()
        if len(labels) != 2:
            raise ValueError(
                f"Binary outcome required, but found {len(labels)} labels: {labels.tolist()}"
            )
        if positive_label not in labels:
            raise ValueError(
                f"positive_label={positive_label!r} not found in outcome labels: {labels.tolist()}"
            )
        return np.asarray((y == positive_label).astype(int), dtype=int)

    def _validate_inputs(self):
        if self.X.ndim != 2:
            raise ValueError("Input feature matrix must be two-dimensional.")
        if self.X.shape[0] != len(self.y):
            raise ValueError("X and y contain different numbers of samples.")
        if self.X.shape[1] != len(self.gene_names):
            raise ValueError("Feature matrix and adata.var_names are inconsistent.")
        if len(np.unique(self.gene_names)) != len(self.gene_names):
            raise ValueError("Duplicate protein names were detected.")
        if not np.isfinite(self.X).all():
            raise ValueError("X contains NaN or infinite values.")
        class_counts = np.bincount(self.y)
        if len(class_counts) != 2 or np.min(class_counts) < 2:
            raise ValueError("Each outcome class must contain at least two samples.")
        if self.panel_size < 2:
            raise ValueError("panel_size must be at least 2.")
        if self.panel_size > self.X.shape[1]:
            raise ValueError("panel_size exceeds the total number of features.")

    @staticmethod
    def _safe_metric_values(y_true, y_prob, y_pred):
        out = {
            "auc": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
            "accuracy": np.nan,
            "validation_error": np.nan,
        }
        if len(np.unique(y_true)) == 2:
            out["auc"] = roc_auc_score(y_true, y_prob)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        out["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        out["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        out["accuracy"] = accuracy_score(y_true, y_pred)
        out["validation_error"] = 1.0 - out["accuracy"]
        return out

    @staticmethod
    def _fit_logistic_and_fit_stats(X, y, random_state, class_weight=None):
        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=5000,
            random_state=random_state,
            class_weight=class_weight,
        )
        model.fit(X, y)

        prob = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        ll_model = np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))

        prevalence = np.clip(np.mean(y), 1e-12, 1 - 1e-12)
        ll_null = np.sum(y * np.log(prevalence) + (1 - y) * np.log(1 - prevalence))

        k = X.shape[1]
        if ll_null == 0:
            pseudo_r2 = np.nan
            adjusted_pseudo_r2 = np.nan
            lr_pvalue = np.nan
        else:
            pseudo_r2 = 1.0 - (ll_model / ll_null)
            adjusted_pseudo_r2 = 1.0 - ((ll_model - k) / ll_null)
            lr_stat = max(0.0, 2.0 * (ll_model - ll_null))
            lr_pvalue = stats.chi2.sf(lr_stat, df=max(1, k))

        fit_stats = {
            "r_squared": pseudo_r2,
            "adjusted_r_squared": adjusted_pseudo_r2,
            "model_p_value": lr_pvalue,
            "intercept": float(model.intercept_[0]),
            "coefficients": np.asarray(model.coef_[0], dtype=float),
        }
        return model, fit_stats

    def _protein_indices(self, panel: Sequence[str]):
        lookup = {g: i for i, g in enumerate(self.gene_names)}
        return np.asarray([lookup[g] for g in panel], dtype=int)

    def run_rf(self):
        panels = []
        for i in tqdm(range(self.n_iter), desc="2BDP RF ranking", leave=False):
            Xtr, _, ytr, _ = train_test_split(
                self.X,
                self.y,
                test_size=0.30,
                stratify=self.y,
                random_state=self.random_state + i,
            )
            rf = RandomForestClassifier(
                n_estimators=self.rf_n_estimators,
                random_state=self.random_state + i,
                class_weight=self.class_weight,
            )
            rf.fit(Xtr, ytr)
            idx = np.argsort(rf.feature_importances_)[::-1][: self.panel_size]
            panels.append(self.gene_names[idx])

        self.rf_panels = np.asarray(panels, dtype=str)
        rows = []
        for i, panel in enumerate(self.rf_panels, start=1):
            row = {"RF_iteration": i}
            row.update({f"Protein_{j + 1}": panel[j] for j in range(len(panel))})
            row["Panel"] = ";".join(panel)
            rows.append(row)
        self.rf_panel_table = pd.DataFrame(rows)
        return self.rf_panels

    def rank_panels(self, panels=None):
        if panels is None:
            panels = self.run_rf() if self.rf_panels is None else self.rf_panels

        pos_freq = [pd.Series(panels[:, j]).value_counts() for j in range(self.panel_size)]

        pos_rows = []
        for position, freq_series in enumerate(pos_freq, start=1):
            for protein, count in freq_series.items():
                pos_rows.append(
                    {
                        "Position": position,
                        "Protein": protein,
                        "Count": int(count),
                        "Frequency": float(count / len(panels)),
                    }
                )
        self.position_frequency_table = pd.DataFrame(pos_rows).sort_values(
            ["Position", "Count", "Protein"], ascending=[True, False, True]
        )

        overall_freq = pd.Series(panels.ravel()).value_counts()
        self.gene_frequency_table = overall_freq.rename_axis("Protein").reset_index(name="Count")
        self.gene_frequency_table["Frequency_per_RF_panel"] = (
            self.gene_frequency_table["Count"] / len(panels)
        )
        self.gene_frequency_table["Fraction_of_all_panel_slots"] = (
            self.gene_frequency_table["Count"] / panels.size
        )

        scores = []
        for i in range(len(panels)):
            scores.append(
                sum(pos_freq[j].get(panels[i, j], 0) for j in range(self.panel_size))
            )
        scores = np.asarray(scores, dtype=float)
        order = np.argsort(scores)[::-1]
        selected_order = order[: min(self.top_panels, len(order))]
        self.ranked_panels = panels[selected_order]

        ranked_rows = []
        for rank, source_idx in enumerate(selected_order, start=1):
            panel = panels[source_idx]
            row = {
                "Panel_rank": rank,
                "Position_frequency_score": float(scores[source_idx]),
                "Source_RF_iteration": int(source_idx + 1),
            }
            row.update({f"Protein_{j + 1}": panel[j] for j in range(len(panel))})
            row["Panel"] = ";".join(panel)
            ranked_rows.append(row)
        self.ranked_panel_table = pd.DataFrame(ranked_rows)
        return self.ranked_panels

    def generate_subpanels(self, ranked=None):
        if ranked is None:
            ranked = self.rank_panels() if self.ranked_panels is None else self.ranked_panels

        subpanels = []
        rows = []
        subpanel_id = 0
        for parent_rank, panel in enumerate(ranked, start=1):
            for k in range(2, len(panel) + 1):
                subpanel_id += 1
                subpanel = np.asarray(panel[:k], dtype=str)
                subpanels.append(subpanel)
                row = {
                    "Subpanel_ID": subpanel_id,
                    "Parent_panel_rank": parent_rank,
                    "Panel_size": k,
                    "Panel": ";".join(subpanel),
                }
                for j in range(self.panel_size):
                    row[f"Protein_{j + 1}"] = subpanel[j] if j < len(subpanel) else None
                rows.append(row)
        self.subpanels = subpanels
        self.subpanel_table = pd.DataFrame(rows)
        return self.subpanels

    def evaluate_panel_rsbmr(self, panel, subpanel_id, parent_panel_rank):
        idx = self._protein_indices(panel)
        X_panel = self.X[:, idx]
        repeat_rows = []
        fit_stats_rows = []

        for repeat in range(self.rsbmr_repeats):
            seed = self.random_state + subpanel_id * 1000 + repeat
            Xtr, Xte, ytr, yte = train_test_split(
                X_panel,
                self.y,
                test_size=0.30,
                stratify=self.y,
                random_state=seed,
            )
            try:
                model, fit_stats = self._fit_logistic_and_fit_stats(
                    Xtr, ytr, random_state=seed, class_weight=self.class_weight
                )
                y_prob = model.predict_proba(Xte)[:, 1]
                y_pred = model.predict(Xte)
                repeat_rows.append(self._safe_metric_values(yte, y_prob, y_pred))
                fit_stats_rows.append(fit_stats)
            except Exception:
                repeat_rows.append(
                    {
                        "auc": np.nan,
                        "sensitivity": np.nan,
                        "specificity": np.nan,
                        "accuracy": np.nan,
                        "validation_error": np.nan,
                    }
                )

        metric_df = pd.DataFrame(repeat_rows)
        fit_df = pd.DataFrame(
            [
                {
                    "r_squared": x["r_squared"],
                    "adjusted_r_squared": x["adjusted_r_squared"],
                    "model_p_value": x["model_p_value"],
                    "intercept": x["intercept"],
                }
                for x in fit_stats_rows
            ]
        )

        result = {
            "Validation": "RSBMR",
            "Subpanel_ID": subpanel_id,
            "Parent_panel_rank": parent_panel_rank,
            "Panel_size": len(panel),
            "Panel": ";".join(panel),
            "AUC": float(metric_df["auc"].mean()),
            "AUC_SD": float(metric_df["auc"].std(ddof=1)) if metric_df["auc"].notna().sum() > 1 else 0.0,
            "Sensitivity": float(metric_df["sensitivity"].mean()),
            "Specificity": float(metric_df["specificity"].mean()),
            "Accuracy": float(metric_df["accuracy"].mean()),
            "Validation_error": float(metric_df["validation_error"].mean()),
            "R_squared": float(fit_df["r_squared"].mean()) if len(fit_df) else np.nan,
            "Adjusted_R_squared": float(fit_df["adjusted_r_squared"].mean()) if len(fit_df) else np.nan,
            "Model_p_value": float(fit_df["model_p_value"].mean()) if len(fit_df) else np.nan,
            "Successful_repeats": int(metric_df["auc"].notna().sum()),
            "Total_repeats": self.rsbmr_repeats,
        }
        for j in range(self.panel_size):
            result[f"Protein_{j + 1}"] = panel[j] if j < len(panel) else None
        return result

    def run_rsbmr(self):
        if self.subpanels is None:
            self.generate_subpanels()
        rows = []
        iterator = tqdm(
            self.subpanel_table.itertuples(index=False),
            total=len(self.subpanel_table),
            desc="RSBMR panel validation",
            leave=False,
        )
        for row in iterator:
            rows.append(
                self.evaluate_panel_rsbmr(
                    panel=row.Panel.split(";"),
                    subpanel_id=int(row.Subpanel_ID),
                    parent_panel_rank=int(row.Parent_panel_rank),
                )
            )
        self.rsbmr_results = pd.DataFrame(rows).sort_values(
            ["AUC", "Panel_size"], ascending=[False, True]
        ).reset_index(drop=True)
        return self.rsbmr_results

    def evaluate_panel_kfold(self, panel, subpanel_id, parent_panel_rank):
        idx = self._protein_indices(panel)
        X_panel = self.X[:, idx]
        min_class = int(np.min(np.bincount(self.y)))
        n_splits = min(self.kfold_splits, min_class)
        if n_splits < 2:
            raise ValueError("Not enough minority-class samples for K-fold CV.")

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state + subpanel_id,
        )
        oof_prob = np.full(len(self.y), np.nan, dtype=float)
        oof_pred = np.full(len(self.y), -1, dtype=int)
        fold_fit_stats = []
        successful_folds = 0

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_panel, self.y), start=1):
            Xtr, Xte = X_panel[train_idx], X_panel[test_idx]
            ytr = self.y[train_idx]
            try:
                model, fit_stats = self._fit_logistic_and_fit_stats(
                    Xtr,
                    ytr,
                    random_state=self.random_state + subpanel_id + fold_idx,
                    class_weight=self.class_weight,
                )
                oof_prob[test_idx] = model.predict_proba(Xte)[:, 1]
                oof_pred[test_idx] = model.predict(Xte)
                fold_fit_stats.append(fit_stats)
                successful_folds += 1
            except Exception:
                continue

        valid = np.isfinite(oof_prob) & (oof_pred >= 0)
        if valid.sum() == 0:
            metrics = {
                "auc": np.nan,
                "sensitivity": np.nan,
                "specificity": np.nan,
                "accuracy": np.nan,
                "validation_error": np.nan,
            }
        else:
            metrics = self._safe_metric_values(self.y[valid], oof_prob[valid], oof_pred[valid])

        fit_df = pd.DataFrame(
            [
                {
                    "r_squared": x["r_squared"],
                    "adjusted_r_squared": x["adjusted_r_squared"],
                    "model_p_value": x["model_p_value"],
                    "intercept": x["intercept"],
                }
                for x in fold_fit_stats
            ]
        )

        result = {
            "Validation": "KFold",
            "Subpanel_ID": subpanel_id,
            "Parent_panel_rank": parent_panel_rank,
            "Panel_size": len(panel),
            "Panel": ";".join(panel),
            "AUC": float(metrics["auc"]),
            "AUC_SD": np.nan,
            "Sensitivity": float(metrics["sensitivity"]),
            "Specificity": float(metrics["specificity"]),
            "Accuracy": float(metrics["accuracy"]),
            "Validation_error": float(metrics["validation_error"]),
            "R_squared": float(fit_df["r_squared"].mean()) if len(fit_df) else np.nan,
            "Adjusted_R_squared": float(fit_df["adjusted_r_squared"].mean()) if len(fit_df) else np.nan,
            "Model_p_value": float(fit_df["model_p_value"].mean()) if len(fit_df) else np.nan,
            "Successful_folds": successful_folds,
            "Total_folds": n_splits,
        }
        for j in range(self.panel_size):
            result[f"Protein_{j + 1}"] = panel[j] if j < len(panel) else None
        return result

    def run_kfold(self):
        if self.subpanels is None:
            self.generate_subpanels()
        rows = []
        iterator = tqdm(
            self.subpanel_table.itertuples(index=False),
            total=len(self.subpanel_table),
            desc="K-fold panel validation",
            leave=False,
        )
        for row in iterator:
            rows.append(
                self.evaluate_panel_kfold(
                    panel=row.Panel.split(";"),
                    subpanel_id=int(row.Subpanel_ID),
                    parent_panel_rank=int(row.Parent_panel_rank),
                )
            )
        self.kfold_results = pd.DataFrame(rows).sort_values(
            ["AUC", "Panel_size"], ascending=[False, True]
        ).reset_index(drop=True)
        return self.kfold_results

    def select_best_panels(self):
        frames = []
        if self.rsbmr_results is not None:
            frames.append(self.rsbmr_results.copy())
        if self.kfold_results is not None:
            frames.append(self.kfold_results.copy())
        if not frames:
            raise RuntimeError("Run RSBMR and/or K-fold validation first.")

        all_results = pd.concat(frames, ignore_index=True)
        eligible = all_results[
            (all_results["AUC"] >= self.auc_threshold)
            & (all_results["Model_p_value"] < self.p_threshold)
        ].copy()

        best_rows = []
        for validation in all_results["Validation"].unique():
            subset = eligible[eligible["Validation"] == validation].copy()
            if subset.empty:
                subset = all_results[all_results["Validation"] == validation].copy()
                subset["Passed_thresholds"] = False
            else:
                subset["Passed_thresholds"] = True
            subset = subset.sort_values(["AUC", "Panel_size"], ascending=[False, True])
            best_rows.append(subset.iloc[0])
        self.best_panels = pd.DataFrame(best_rows).reset_index(drop=True)
        return self.best_panels

    def build_summary_table(self):
        self.summary_table = pd.DataFrame(
            [
                {"Category": "Dataset", "Metric": "Samples", "Value": len(self.y)},
                {"Category": "Dataset", "Metric": "Protein features", "Value": self.X.shape[1]},
                {"Category": "Dataset", "Metric": "Positive samples", "Value": int(np.sum(self.y == 1))},
                {"Category": "Dataset", "Metric": "Negative samples", "Value": int(np.sum(self.y == 0))},
                {"Category": "RF", "Metric": "RF iterations", "Value": self.n_iter},
                {"Category": "RF", "Metric": "Trees per RF", "Value": self.rf_n_estimators},
                {"Category": "Ranking", "Metric": "Panel size", "Value": self.panel_size},
                {"Category": "Ranking", "Metric": "Top-ranked panels retained", "Value": self.top_panels},
                {"Category": "Subpanels", "Metric": "Generated subpanels", "Value": len(self.subpanel_table) if self.subpanel_table is not None else 0},
                {"Category": "Validation", "Metric": "RSBMR repeats", "Value": self.rsbmr_repeats},
                {"Category": "Validation", "Metric": "K-fold splits requested", "Value": self.kfold_splits},
                {"Category": "Selection", "Metric": "AUC threshold", "Value": self.auc_threshold},
                {"Category": "Selection", "Metric": "Model p-value threshold", "Value": self.p_threshold},
            ]
        )
        return self.summary_table

    def print_final_report(self):
        print("\n" + "=" * 72)
        print("2BDP FINAL REPORT")
        print("=" * 72)
        print("\n1. Dataset")
        print(f"   Samples:          {len(self.y)}")
        print(f"   Protein features: {self.X.shape[1]}")
        print(f"   Class counts:     negative={np.sum(self.y == 0)}, positive={np.sum(self.y == 1)}")
        print("\n2. Feature ranking")
        print(f"   RF iterations:    {self.n_iter}")
        print(f"   RF panel size:    {self.panel_size}")
        print(f"   Ranked panels:    {len(self.ranked_panels)}")
        print(f"   Subpanels:        {len(self.subpanels)}")

        if self.best_panels is not None:
            print("\n3. Best panels")
            for _, row in self.best_panels.iterrows():
                print(f"\n   [{row['Validation']}]")
                print(f"   Panel:        {row['Panel']}")
                print(f"   Panel size:   {int(row['Panel_size'])}")
                print(f"   AUC:          {row['AUC']:.4f}")
                print(f"   Sensitivity:  {row['Sensitivity']:.4f}")
                print(f"   Specificity:  {row['Specificity']:.4f}")
                print(f"   Accuracy:     {row['Accuracy']:.4f}")
                print(f"   Pseudo-R2:    {row['R_squared']:.4f}")
                print(f"   Adj. R2:      {row['Adjusted_R_squared']:.4f}")
                print(f"   Model p:      {row['Model_p_value']:.4g}")
                print(f"   Passed thresholds: {bool(row.get('Passed_thresholds', False))}")

        print("\n" + "=" * 72)
        print("INTERPRETATION NOTE")
        print("=" * 72)
        print("These results reproduce the original 2BDP-style internal feature-ranking and panel-validation workflow.")
        print("Because feature ranking precedes downstream validation, performance is intended for method comparison rather than unbiased external validation.")
        print("=" * 72 + "\n")

    def save_tables(self, output_prefix: str = "2bdp_results"):
        tables = {
            "rf_panels": self.rf_panel_table,
            "position_frequencies": self.position_frequency_table,
            "gene_frequencies": self.gene_frequency_table,
            "ranked_panels": self.ranked_panel_table,
            "subpanels": self.subpanel_table,
            "rsbmr_results": self.rsbmr_results,
            "kfold_results": self.kfold_results,
            "best_panels": self.best_panels,
            "summary": self.summary_table,
        }
        saved_files = {}
        for suffix, table in tables.items():
            if table is None:
                continue
            path = f"{output_prefix}_{suffix}.csv"
            table.to_csv(path, index=False)
            saved_files[suffix] = path
        return saved_files

    def run_complete_pipeline(self, run_rsbmr: bool = True, run_kfold: bool = True) -> Dict[str, Any]:
        if self.verbose:
            print("\nStarting original 2BDP-style feature ranking...")
        self.run_rf()
        self.rank_panels()
        self.generate_subpanels()

        if run_rsbmr:
            if self.verbose:
                print("\nRunning RSBMR validation...")
            self.run_rsbmr()

        if run_kfold:
            if self.verbose:
                print("\nRunning K-fold validation...")
            self.run_kfold()

        self.select_best_panels()
        self.build_summary_table()
        if self.verbose:
            self.print_final_report()

        return {
            "rf_panels": self.rf_panels,
            "rf_panel_table": self.rf_panel_table,
            "position_frequency_table": self.position_frequency_table,
            "gene_frequency_table": self.gene_frequency_table,
            "ranked_panels": self.ranked_panels,
            "ranked_panel_table": self.ranked_panel_table,
            "subpanels": self.subpanels,
            "subpanel_table": self.subpanel_table,
            "rsbmr_results": self.rsbmr_results,
            "kfold_results": self.kfold_results,
            "best_panels": self.best_panels,
            "summary_table": self.summary_table,
        }