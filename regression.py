from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*", category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from stat662_tcga.constants import DEFAULT_REGRESSION_COVARIATES
from stat662_tcga.preprocessing import load_expression_bundle, load_tabular_dataset, sanitize_expression_matrix
from stat662_tcga.utils import ensure_directory, save_json, set_seed
from stat662_tcga.visualization import save_heatmap, save_ranked_lollipop_plot


RIDGE_ALPHAS = np.array([0.1, 1.0, 10.0, 100.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit multivariate regression on TCGA PC scores.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-pcs", type=int, default=10)
    parser.add_argument("--covariates", default=",".join(DEFAULT_REGRESSION_COVARIATES))
    parser.add_argument("--permutations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_directory(args.output_dir)
    processed_dir = Path(args.processed_dir)

    manifest = load_tabular_dataset(processed_dir / "representative_manifest.parquet").set_index("sample_barcode")
    X = load_expression_bundle(processed_dir / "tcga_top2000_expression.npz")
    X, cleaning_summary = sanitize_expression_matrix(X)
    manifest = manifest.loc[X.index]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_pcs = max(1, min(args.num_pcs, X_scaled.shape[0] - 1, X_scaled.shape[1]))
    pca = PCA(n_components=n_pcs, random_state=args.seed)
    Y = pd.DataFrame(
        pca.fit_transform(X_scaled),
        index=X.index,
        columns=[f"PC{i + 1}" for i in range(n_pcs)],
    )

    covariates = [item.strip() for item in args.covariates.split(",") if item.strip()]
    design, design_blocks = _build_design_matrix(manifest, covariates)
    common_index = design.index.intersection(Y.index)
    design = design.loc[common_index]
    Y = Y.loc[common_index]
    design_blocks = {
        block: [column for column in columns if column in design.columns]
        for block, columns in design_blocks.items()
    }

    model = RidgeCV(alphas=RIDGE_ALPHAS)
    model.fit(design, Y)
    fitted = pd.DataFrame(model.predict(design), index=Y.index, columns=Y.columns)

    cv_r2, per_pc_r2 = _cross_validated_r2(design, Y, seed=args.seed)
    permutation_p_value = None
    if args.permutations > 0:
        permutation_p_value = _permutation_p_value(design, Y, observed_score=cv_r2, n_permutations=args.permutations, seed=args.seed)
    full_in_sample_r2 = float(r2_score(Y, fitted, multioutput="variance_weighted"))
    block_summary = _blockwise_association_summary(
        design,
        Y,
        blocks=design_blocks,
        full_cv_r2=cv_r2,
        full_in_sample_r2=full_in_sample_r2,
        permutations=args.permutations,
        seed=args.seed,
    )

    coefficients = pd.DataFrame(model.coef_.T, index=design.columns, columns=Y.columns)
    coefficients.reset_index().rename(columns={"index": "predictor"}).to_csv(output_dir / "regression_coefficients.csv", index=False)
    fitted.join(Y, lsuffix="_pred", rsuffix="_true").reset_index().to_csv(output_dir / "regression_fitted_scores.csv", index=False)
    block_summary.to_csv(output_dir / "regression_block_summary.csv", index=False)

    top_predictors = coefficients.abs().mean(axis=1).sort_values(ascending=False).head(20).index
    save_heatmap(
        coefficients.loc[top_predictors].to_numpy(),
        output_dir / "regression_coefficients_heatmap.png",
        title="Top predictors by mean absolute coefficient",
        xlabel="Principal component",
        ylabel="Predictor",
        xticklabels=coefficients.columns.tolist(),
        yticklabels=top_predictors.tolist(),
        cmap="coolwarm",
    )
    if not block_summary.empty:
        delta_metrics = ["cv_delta_r2", "in_sample_delta_r2"]
        has_p_values = "p_value" in block_summary.columns and block_summary["p_value"].notna().any()
        if has_p_values:
            delta_metrics.append("p_value")
        save_heatmap(
            block_summary[delta_metrics].to_numpy(),
            output_dir / "regression_block_summary_heatmap.png",
            title="Blockwise multivariate association summary",
            xlabel="Metric",
            ylabel="Covariate block",
            xticklabels=delta_metrics,
            yticklabels=block_summary["block"].tolist(),
            cmap="coolwarm",
        )
        save_ranked_lollipop_plot(
            block_summary.set_index("block")["cv_delta_r2"],
            output_dir / "regression_block_cv_delta_r2.png",
            title="Cross-validated multivariate contribution by covariate block",
            subtitle="Each point shows the drop in variance-weighted R2 when a block is removed",
            xlabel="Delta variance-weighted R2",
            ylabel="Covariate block",
            color="#173F35",
        )
        if has_p_values:
            save_ranked_lollipop_plot(
                block_summary.dropna(subset=["p_value"]).set_index("block")["p_value"],
                output_dir / "regression_block_p_values.png",
                title="Permutation p-values for covariate blocks",
                subtitle="Smaller values indicate stronger multivariate association with the latent PC response",
                xlabel="Permutation p-value",
                ylabel="Covariate block",
                color="#A85D43",
            )

    save_json(
        {
            "covariates": covariates,
            "rows_used": int(len(design)),
            "design_columns": int(design.shape[1]),
            "num_pcs": int(n_pcs),
            "ridge_alpha": float(model.alpha_),
            "in_sample_variance_weighted_r2": full_in_sample_r2,
            "cross_validated_variance_weighted_r2": float(cv_r2),
            "per_pc_cross_validated_r2": {column: float(score) for column, score in per_pc_r2.items()},
            "permutation_p_value": permutation_p_value,
            "blockwise_association": block_summary.to_dict(orient="records"),
            "matrix_cleaning_summary": cleaning_summary,
        },
        output_dir / "regression_summary.json",
    )


def _build_design_matrix(manifest: pd.DataFrame, covariates: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    frame = manifest.copy()
    if "age_at_initial_pathologic_diagnosis" in frame.columns:
        age_numeric = pd.to_numeric(frame["age_at_initial_pathologic_diagnosis"], errors="coerce")
        if "days_to_birth" in frame.columns:
            fallback_age = pd.to_numeric(frame["days_to_birth"], errors="coerce").abs() / 365.25
            age_numeric = age_numeric.fillna(fallback_age)
        frame["age_at_initial_pathologic_diagnosis"] = age_numeric

    selected = frame.loc[:, [column for column in covariates if column in frame.columns]].copy()
    numeric_columns = selected.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in selected.columns if column not in numeric_columns]

    for column in categorical_columns:
        selected[column] = selected[column].fillna("Missing").astype(str)

    if numeric_columns:
        selected = selected.dropna(subset=numeric_columns)
    design = pd.DataFrame(index=selected.index)
    blocks: dict[str, list[str]] = {}
    for column in numeric_columns:
        design[column] = pd.to_numeric(selected[column], errors="coerce")
        blocks[column] = [column]
    for column in categorical_columns:
        dummies = pd.get_dummies(selected[column], prefix=column, prefix_sep="_", drop_first=False)
        design = pd.concat([design, dummies], axis=1)
        blocks[column] = dummies.columns.tolist()
    return design.astype(float), blocks


def _cross_validated_r2(X: pd.DataFrame, Y: pd.DataFrame, *, seed: int) -> tuple[float, dict[str, float]]:
    splitter = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=seed)
    overall_scores = []
    per_pc_scores: dict[str, list[float]] = {column: [] for column in Y.columns}
    for train_idx, test_idx in splitter.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        Y_train = Y.iloc[train_idx]
        Y_test = Y.iloc[test_idx]
        predicted = _predict_multivariate_ridge(X_train, Y_train, X_test)
        overall_scores.append(r2_score(Y_test, predicted, multioutput="variance_weighted"))
        for column in Y.columns:
            per_pc_scores[column].append(r2_score(Y_test[column], predicted[column]))
    return float(np.mean(overall_scores)), {column: float(np.mean(scores)) for column, scores in per_pc_scores.items()}


def _predict_multivariate_ridge(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    if X_train.shape[1] == 0:
        mean_vector = Y_train.mean(axis=0)
        repeated = np.tile(mean_vector.to_numpy(), (len(X_test), 1))
        return pd.DataFrame(repeated, index=X_test.index, columns=Y_train.columns)
    model = RidgeCV(alphas=RIDGE_ALPHAS)
    model.fit(X_train, Y_train)
    return pd.DataFrame(model.predict(X_test), index=X_test.index, columns=Y_train.columns)


def _blockwise_association_summary(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    blocks: dict[str, list[str]],
    full_cv_r2: float,
    full_in_sample_r2: float,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | None]] = []
    rng = np.random.default_rng(seed)
    for block, columns in blocks.items():
        if not columns:
            continue
        reduced_columns = [column for column in X.columns if column not in columns]
        X_reduced = X.loc[:, reduced_columns]
        reduced_cv_r2, _ = _cross_validated_r2(X_reduced, Y, seed=seed)
        reduced_fitted = _predict_multivariate_ridge(X_reduced, Y, X_reduced)
        reduced_in_sample_r2 = float(r2_score(Y, reduced_fitted, multioutput="variance_weighted"))
        observed_delta = float(full_cv_r2 - reduced_cv_r2)
        p_value: float | None = None
        if permutations > 0:
            null_deltas = []
            for _ in range(permutations):
                X_permuted = X.copy()
                permutation_order = rng.permutation(len(X_permuted))
                for column in columns:
                    X_permuted[column] = X_permuted[column].to_numpy()[permutation_order]
                permuted_cv_r2, _ = _cross_validated_r2(X_permuted, Y, seed=int(rng.integers(0, 1_000_000)))
                null_deltas.append(float(permuted_cv_r2 - reduced_cv_r2))
            p_value = float((sum(delta >= observed_delta for delta in null_deltas) + 1) / (len(null_deltas) + 1))
        rows.append(
            {
                "block": block,
                "n_columns": int(len(columns)),
                "reduced_cv_r2": float(reduced_cv_r2),
                "cv_delta_r2": observed_delta,
                "reduced_in_sample_r2": reduced_in_sample_r2,
                "in_sample_delta_r2": float(full_in_sample_r2 - reduced_in_sample_r2),
                "p_value": p_value,
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values("cv_delta_r2", ascending=True).reset_index(drop=True)


def _permutation_p_value(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    observed_score: float,
    n_permutations: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(n_permutations):
        shuffled = Y.sample(frac=1.0, replace=False, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
        shuffled.index = Y.index
        score, _ = _cross_validated_r2(X, shuffled, seed=int(rng.integers(0, 1_000_000)))
        if score >= observed_score:
            exceedances += 1
    return float((exceedances + 1) / (n_permutations + 1))


if __name__ == "__main__":
    main()
