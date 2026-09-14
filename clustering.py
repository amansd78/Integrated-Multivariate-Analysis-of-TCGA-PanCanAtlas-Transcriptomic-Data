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
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from stat662_tcga.preprocessing import load_expression_bundle, load_tabular_dataset, sanitize_expression_matrix
from stat662_tcga.utils import ensure_directory, save_json, set_seed
from stat662_tcga.visualization import (
    prettify_label,
    save_cluster_scatter,
    save_heatmap,
    save_metric_heatmap,
    save_model_comparison_plot,
    save_ranked_lollipop_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PCA-based clustering comparisons for TCGA.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pc-components", type=int, default=20)
    parser.add_argument("--n-clusters", type=int)
    parser.add_argument("--stability-repeats", type=int, default=5)
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
    X = X.astype(np.float64)
    manifest = manifest.loc[X.index]
    labels = manifest["acronym"].astype(str)

    X_scaled = _stable_standardize(X)
    n_components = max(2, min(args.pc_components, X_scaled.shape[0] - 1, X_scaled.shape[1]))
    scores = PCA(n_components=n_components, random_state=args.seed).fit_transform(X_scaled).astype(np.float64, copy=False)
    score_columns = [f"PC{i + 1}" for i in range(scores.shape[1])]
    scores_frame = pd.DataFrame(scores, index=X.index, columns=score_columns)

    n_clusters = args.n_clusters or labels.nunique()
    rows = []
    failed_methods: list[dict[str, str]] = []
    for method in ["kmeans", "agglomerative", "gaussian_mixture"]:
        method_display = prettify_label(method)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                predicted = _fit_predict(method, scores, n_clusters=n_clusters, random_state=args.seed)
        except Exception as exc:
            failed_methods.append({"method": method, "error": f"{type(exc).__name__}: {exc}"})
            save_json(
                {
                    "method": method,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                output_dir / f"{method}_failure.json",
            )
            continue

        silhouette = _safe_silhouette(scores, predicted, seed=args.seed)
        ari = adjusted_rand_score(labels, predicted)
        nmi = normalized_mutual_info_score(labels, predicted)
        stability = _subsample_stability(method, scores, n_clusters=n_clusters, repeats=args.stability_repeats, seed=args.seed)
        rows.append(
            {
                "model": method,
                "silhouette": float(silhouette),
                "ari": float(ari),
                "nmi": float(nmi),
                "subsample_ari": float(stability),
            }
        )
        failure_path = output_dir / f"{method}_failure.json"
        if failure_path.exists():
            failure_path.unlink()

        assignments = pd.DataFrame(
            {
                "sample_barcode": scores_frame.index,
                "cluster": predicted,
                "acronym": labels.values,
            }
        )
        assignments.to_csv(output_dir / f"{method}_assignments.csv", index=False)
        contingency = pd.crosstab(assignments["acronym"], assignments["cluster"])
        contingency.to_csv(output_dir / f"{method}_cluster_vs_cancer.csv")
        cluster_sizes = assignments["cluster"].value_counts().sort_values(ascending=True)
        dominant_share = contingency.max(axis=0).div(contingency.sum(axis=0)).sort_values(ascending=True)
        save_ranked_lollipop_plot(
            cluster_sizes,
            output_dir / f"{method}_cluster_sizes.png",
            title=f"{method_display} cluster size profile",
            subtitle="Cluster sizes reveal whether a method fragments the cohort or creates broad phenotypic basins",
            xlabel="Samples in cluster",
            ylabel="Cluster",
            color="#325C80",
        )
        save_ranked_lollipop_plot(
            dominant_share,
            output_dir / f"{method}_cluster_purity.png",
            title=f"{method_display} cluster purity profile",
            subtitle="Each point shows the dominant cancer-type share within a cluster",
            xlabel="Dominant cancer-type share",
            ylabel="Cluster",
            color="#B4852B",
        )
        save_heatmap(
            contingency.to_numpy(),
            output_dir / f"{method}_cluster_vs_cancer.png",
            title=f"{method_display} cluster vs cancer-type contingency",
            xlabel="Cluster",
            ylabel="Cancer type",
            xticklabels=[str(value) for value in contingency.columns.tolist()],
            yticklabels=contingency.index.tolist(),
            cmap="Blues",
        )
        save_cluster_scatter(
            scores_frame,
            pd.Series([f"Cluster {value}" for value in predicted], index=scores_frame.index),
            output_dir / f"{method}_pc1_pc2_clusters.png",
            x="PC1",
            y="PC2",
            title=f"{method_display} clusters on PC1/PC2",
        )

    if not rows:
        raise RuntimeError("All clustering methods failed. Inspect the *_failure.json files in the output directory.")

    summary = pd.DataFrame(rows).sort_values("ari", ascending=False).reset_index(drop=True)
    summary.to_csv(output_dir / "clustering_summary.csv", index=False)
    save_model_comparison_plot(
        summary.rename(columns={"ari": "macro_f1", "silhouette": "accuracy"})[["model", "accuracy", "macro_f1"]],
        output_dir / "clustering_summary.png",
        title="Clustering quality across unsupervised methods",
        subtitle="Dark green marks silhouette; ochre marks adjusted Rand index",
        left_label="Silhouette",
        right_label="ARI",
    )
    save_metric_heatmap(
        summary,
        output_dir / "clustering_metric_heatmap.png",
        index_col="model",
        metric_cols=["silhouette", "ari", "nmi", "subsample_ari"],
        title="Clustering scorecard",
        subtitle="Internal separation, external agreement, and subsample stability",
        fill_label="Score",
    )
    save_json(
        {
            "n_clusters": int(n_clusters),
            "pc_components_used": int(scores.shape[1]),
            "best_method_by_ari": summary.iloc[0]["model"],
            "matrix_cleaning_summary": cleaning_summary,
            "failed_methods": failed_methods,
        },
        output_dir / "clustering_run_summary.json",
    )


def _stable_standardize(frame: pd.DataFrame) -> np.ndarray:
    matrix = np.asarray(frame, dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = StandardScaler().fit_transform(matrix)
    scaled = np.asarray(scaled, dtype=np.float64)
    if not np.isfinite(scaled).all():
        raise ValueError("Encountered non-finite values after standardization.")
    return scaled


def _fit_predict(method: str, X: np.ndarray, *, n_clusters: int, random_state: int) -> np.ndarray:
    if method == "kmeans":
        return KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state).fit_predict(X)
    if method == "agglomerative":
        return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(X)
    if method == "gaussian_mixture":
        return GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",
            reg_covar=1e-5,
            n_init=5,
            max_iter=300,
            init_params="kmeans",
            random_state=random_state,
        ).fit_predict(X.astype(np.float64, copy=False))
    raise ValueError(f"Unknown clustering method: {method}")


def _safe_silhouette(X: np.ndarray, labels: np.ndarray, *, seed: int) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    sample_size = min(len(X), 5000)
    return float(silhouette_score(X, labels, sample_size=sample_size, random_state=seed))


def _subsample_stability(
    method: str,
    X: np.ndarray,
    *,
    n_clusters: int,
    repeats: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    assignments = []
    for _ in range(repeats):
        indices = np.sort(rng.choice(X.shape[0], size=max(50, int(0.8 * X.shape[0])), replace=False))
        try:
            labels = _fit_predict(method, X[indices], n_clusters=n_clusters, random_state=int(rng.integers(0, 1_000_000)))
        except Exception:
            continue
        assignments.append((indices, labels))

    if len(assignments) < 2:
        return float("nan")

    scores = []
    for i in range(len(assignments)):
        for j in range(i + 1, len(assignments)):
            idx_i, labels_i = assignments[i]
            idx_j, labels_j = assignments[j]
            overlap, pos_i, pos_j = np.intersect1d(idx_i, idx_j, return_indices=True)
            if len(overlap) < 20:
                continue
            scores.append(adjusted_rand_score(labels_i[pos_i], labels_j[pos_j]))
    return float(np.mean(scores)) if scores else float("nan")


if __name__ == "__main__":
    main()
