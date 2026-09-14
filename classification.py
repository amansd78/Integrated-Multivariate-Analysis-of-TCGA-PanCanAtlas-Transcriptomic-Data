from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*", category=UserWarning)

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from stat662_tcga.preprocessing import ExpressionPCTransformer, load_expression_bundle, load_tabular_dataset, sanitize_expression_matrix
from stat662_tcga.utils import ensure_directory, save_json, set_seed
from stat662_tcga.visualization import (
    prettify_label,
    save_confusion_matrix_plot,
    save_metric_heatmap,
    save_model_comparison_plot,
    save_paired_comparison_plot,
    save_ranked_lollipop_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark cancer-type classifiers on the TCGA PC representation.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-genes", type=int, default=2000)
    parser.add_argument(
        "--feature-space",
        choices=["shared_top_gene_space", "training_selected_top_genes"],
        default="shared_top_gene_space",
        help="Use the shared precomputed top-gene matrix or reselect top genes on the training split.",
    )
    parser.add_argument("--pc-components", type=int, default=50)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_directory(args.output_dir)
    processed_dir = Path(args.processed_dir)

    manifest = load_tabular_dataset(processed_dir / "representative_manifest.parquet").set_index("sample_barcode")
    if args.feature_space == "shared_top_gene_space":
        expression_matrix = load_expression_bundle(processed_dir / "tcga_top2000_expression.npz")
        requested_genes = min(args.top_genes, expression_matrix.shape[1])
        fixed_genes = expression_matrix.columns[:requested_genes].tolist()
        expression_matrix = expression_matrix.loc[:, fixed_genes]
        transformer = ExpressionPCTransformer(
            n_top_genes=requested_genes,
            n_components=args.pc_components,
            fixed_genes=fixed_genes,
        )
        feature_space_source = "tcga_top2000_expression.npz"
    else:
        expression_matrix = load_expression_bundle(processed_dir / "tcga_clean_expression_full.npz")
        transformer = ExpressionPCTransformer(n_top_genes=args.top_genes, n_components=args.pc_components)
        feature_space_source = "tcga_clean_expression_full.npz"
    expression_matrix, cleaning_summary = sanitize_expression_matrix(expression_matrix)
    manifest = manifest.loc[expression_matrix.index]

    split = json.loads((processed_dir / "supervised_split.json").read_text(encoding="utf-8"))
    train_ids = [sample_id for sample_id in split["train_sample_ids"] if sample_id in expression_matrix.index]
    test_ids = [sample_id for sample_id in split["test_sample_ids"] if sample_id in expression_matrix.index]
    if not test_ids:
        raise ValueError("The supervised split did not contain any held-out test samples.")

    X_train = expression_matrix.loc[train_ids]
    X_test = expression_matrix.loc[test_ids]
    y_train = manifest.loc[train_ids, "acronym"].astype(str)
    y_test = manifest.loc[test_ids, "acronym"].astype(str)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    X_train_pc = transformer.fit_transform(X_train)
    X_test_pc = transformer.transform(X_test)
    joblib.dump(transformer, output_dir / "pc_transformer.joblib")
    save_json(transformer.to_metadata(), output_dir / "pc_transformer_metadata.json")

    min_class_count = int(pd.Series(y_train_encoded).value_counts().min())
    cv_folds = max(2, min(args.cv_folds, min_class_count))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=args.seed)

    rows = []
    failed_models: list[dict[str, str]] = []
    class_labels = label_encoder.classes_.tolist()
    models_dir = ensure_directory(output_dir / "models")
    model_reports: dict[str, dict[str, object]] = {}
    model_predictions: dict[str, np.ndarray] = {}

    for model_name, estimator, param_grid in _model_grid(random_state=args.seed):
        try:
            search, predictions_encoded = _fit_grid_search(
                estimator=estimator,
                param_grid=param_grid,
                X_train=X_train_pc,
                y_train=y_train_encoded,
                X_test=X_test_pc,
                cv=cv,
                n_jobs=args.n_jobs,
            )
        except Exception as exc:
            failed_models.append({"model": model_name, "error": f"{type(exc).__name__}: {exc}"})
            save_json(
                {
                    "model": model_name,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                output_dir / f"{model_name}_failure.json",
            )
            continue

        predictions_encoded = np.asarray(predictions_encoded, dtype=int)
        predictions = label_encoder.inverse_transform(predictions_encoded)
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        metrics = {
            "model": model_name,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            "best_cv_macro_f1": float(search.best_score_),
            "best_params": search.best_params_,
        }
        rows.append(metrics)
        failure_path = output_dir / f"{model_name}_failure.json"
        if failure_path.exists():
            failure_path.unlink()
        joblib.dump(search.best_estimator_, models_dir / f"{model_name}.joblib")
        pd.DataFrame(
            {
                "sample_barcode": X_test_pc.index,
                "true_label": y_test.values,
                "predicted_label": predictions,
            }
        ).to_csv(output_dir / f"{model_name}_test_predictions.csv", index=False)
        save_json(
            {
                **metrics,
                "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
            },
            output_dir / f"{model_name}_metrics.json",
        )
        model_reports[model_name] = report
        model_predictions[model_name] = predictions
        save_confusion_matrix_plot(
            y_test.tolist(),
            predictions.tolist(),
            labels=class_labels,
            path=output_dir / f"{model_name}_confusion_matrix.png",
            title=f"{prettify_label(model_name)} confusion matrix",
        )

    if not rows:
        raise RuntimeError("All classification models failed. Inspect the *_failure.json files in the output directory.")

    comparison = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    save_model_comparison_plot(
        comparison,
        output_dir / "model_comparison.png",
        title="Held-out cancer-type classification",
        subtitle="Dark green marks accuracy; ochre marks macro-F1 on the held-out cohort",
    )
    save_metric_heatmap(
        comparison.sort_values("macro_f1", ascending=False),
        output_dir / "model_comparison_heatmap.png",
        index_col="model",
        metric_cols=["accuracy", "macro_f1", "weighted_f1", "best_cv_macro_f1"],
        title="Held-out and cross-validated classifier metrics",
        subtitle="Comparison across all benchmarked cancer-type classifiers",
        fill_label="Score",
    )
    save_paired_comparison_plot(
        comparison.sort_values("macro_f1", ascending=False),
        output_dir / "cv_vs_test_macro_f1.png",
        category_col="model",
        left_col="best_cv_macro_f1",
        right_col="macro_f1",
        title="Cross-validated versus held-out macro-F1",
        subtitle="Smaller gaps indicate more faithful generalization from CV to the held-out cohort",
        left_label="CV macro-F1",
        right_label="Test macro-F1",
        y_label="Model",
        x_label="Macro-F1",
    )

    best_model_name = str(comparison.iloc[0]["model"])
    best_report = model_reports[best_model_name]
    per_class_f1 = (
        pd.Series(
            {
                label: best_report[label]["f1-score"]
                for label in class_labels
                if label in best_report
            }
        )
        .sort_values(ascending=True)
    )
    save_ranked_lollipop_plot(
        per_class_f1,
        output_dir / "best_model_per_class_f1.png",
        title=f"Per-class F1 for {prettify_label(best_model_name)}",
        subtitle="Lower-performing cancer types are emphasized at the top of the ranking",
        xlabel="F1 score",
        ylabel="Cancer type",
        color="#B4852B",
    )
    save_json(
        {
            "train_rows": int(len(X_train_pc)),
            "test_rows": int(len(X_test_pc)),
            "pc_components_used": int(X_train_pc.shape[1]),
            "top_genes_used": int(len(transformer.selected_genes_)),
            "feature_space_mode": args.feature_space,
            "feature_space_source": feature_space_source,
            "cv_folds": int(cv_folds),
            "best_model": best_model_name,
            "matrix_cleaning_summary": cleaning_summary,
            "failed_models": failed_models,
        },
        output_dir / "classification_summary.json",
    )


def _model_grid(random_state: int) -> list[tuple[str, object, dict[str, list[object]]]]:
    return [
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=4000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=random_state,
                tol=1e-3,
            ),
            {"C": [0.1, 1.0]},
        ),
        (
            "svm_linear",
            SVC(kernel="linear", class_weight="balanced"),
            {"C": [0.5, 1.0]},
        ),
        (
            "svm_rbf",
            SVC(kernel="rbf", class_weight="balanced"),
            {"C": [1.0, 3.0], "gamma": ["scale"]},
        ),
        (
            "random_forest",
            RandomForestClassifier(
                random_state=random_state,
                class_weight="balanced_subsample",
                n_jobs=1,
            ),
            {"n_estimators": [300], "max_depth": [None]},
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(random_state=random_state),
            {"learning_rate": [0.05, 0.10], "max_depth": [8]},
        ),
        (
            "mlp",
            MLPClassifier(
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.10,
                n_iter_no_change=15,
                max_iter=400,
                learning_rate_init=5e-4,
            ),
            {"hidden_layer_sizes": [(128, 64)], "alpha": [1e-4]},
        ),
    ]


def _fit_grid_search(
    *,
    estimator: object,
    param_grid: dict[str, list[object]],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    cv: StratifiedKFold,
    n_jobs: int,
) -> tuple[GridSearchCV, np.ndarray]:
    last_error: Exception | None = None
    search_n_jobs = [n_jobs] if n_jobs == 1 else [n_jobs, 1]
    for candidate_n_jobs in search_n_jobs:
        search = GridSearchCV(
            estimator=clone(estimator),
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=candidate_n_jobs,
            refit=True,
            error_score="raise",
        )
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                search.fit(X_train, y_train)
                predictions_encoded = search.predict(X_test)
            return search, np.asarray(predictions_encoded, dtype=int)
        except PermissionError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Grid search did not execute and no underlying error was captured.")


if __name__ == "__main__":
    main()
