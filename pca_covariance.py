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
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV, LedoitWolf, OAS
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from stat662_tcga.preprocessing import load_expression_bundle, load_tabular_dataset, sanitize_expression_matrix
from stat662_tcga.utils import ensure_directory, save_json, set_seed
from stat662_tcga.visualization import (
    save_explained_variance_plot,
    save_heatmap,
    save_metric_heatmap,
    save_ranked_lollipop_plot,
    save_pca_scatter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PCA and covariance analysis on the TCGA processed matrices.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--stability-resamples", type=int, default=10)
    parser.add_argument("--stability-fraction", type=float, default=0.80)
    parser.add_argument("--pca-stability-components", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_directory(args.output_dir)
    processed_dir = Path(args.processed_dir)

    manifest = load_tabular_dataset(processed_dir / "representative_manifest.parquet").set_index("sample_barcode")
    top2000 = load_expression_bundle(processed_dir / "tcga_top2000_expression.npz")
    top500 = load_expression_bundle(processed_dir / "tcga_top500_expression.npz")
    top2000, top2000_cleaning = sanitize_expression_matrix(top2000)
    top500, top500_cleaning = sanitize_expression_matrix(top500)
    top2000 = top2000.astype(np.float64)
    top500 = top500.astype(np.float64)
    manifest = manifest.loc[top2000.index]

    X_pca = _stable_standardize(top2000)
    n_components = max(1, min(args.pca_components, X_pca.shape[0] - 1, X_pca.shape[1]))
    pca = PCA(n_components=n_components, random_state=args.seed)
    scores = pca.fit_transform(X_pca)
    score_columns = [f"PC{i + 1}" for i in range(scores.shape[1])]
    scores_frame = pd.DataFrame(scores, index=top2000.index, columns=score_columns)
    scores_frame = scores_frame.join(manifest[["acronym", "gender", "pathologic_stage"]], how="left")
    scores_frame.reset_index().to_parquet(output_dir / "pca_scores.parquet", index=False)
    scores_frame.reset_index().to_csv(output_dir / "pca_scores.csv", index=False)

    explained = pd.DataFrame(
        {
            "component": score_columns,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": pca.explained_variance_ratio_.cumsum(),
        }
    )
    explained.to_csv(output_dir / "pca_explained_variance.csv", index=False)
    save_explained_variance_plot(pca.explained_variance_ratio_, output_dir / "pca_explained_variance.png")
    loadings = pd.DataFrame(pca.components_.T, index=top2000.columns, columns=score_columns)
    loadings.to_csv(output_dir / "pca_loadings.csv")
    save_pca_scatter(
        scores_frame,
        scores_frame["acronym"],
        output_dir / "pca_pc1_pc2_by_cancer.png",
        x="PC1",
        y="PC2",
        title="PCA: PC1 vs PC2 by cancer type",
    )
    if "PC3" in scores_frame.columns:
        save_pca_scatter(
            scores_frame,
            scores_frame["acronym"],
            output_dir / "pca_pc1_pc3_by_cancer.png",
            x="PC1",
            y="PC3",
            title="PCA: PC1 vs PC3 by cancer type",
        )
    if "gender" in scores_frame.columns:
        save_pca_scatter(
            scores_frame,
            scores_frame["gender"].fillna("Unknown").astype(str),
            output_dir / "pca_pc1_pc2_by_gender.png",
            x="PC1",
            y="PC2",
            top_k_labels=4,
            title="PCA: PC1 vs PC2 by sex",
        )
    if "pathologic_stage" in scores_frame.columns:
        save_pca_scatter(
            scores_frame,
            scores_frame["pathologic_stage"].fillna("Unknown").astype(str),
            output_dir / "pca_pc1_pc2_by_stage.png",
            x="PC1",
            y="PC2",
            top_k_labels=8,
            title="PCA: PC1 vs PC2 by pathologic stage",
        )
    save_ranked_lollipop_plot(
        _top_component_loadings(loadings, "PC1", top_n=20),
        output_dir / "pc1_top_loadings.png",
        title="Strongest absolute gene loadings on PC1",
        subtitle="Genes are ranked by absolute contribution to the first principal component",
        xlabel="Absolute loading",
        ylabel="Gene",
        color="#325C80",
    )
    if "PC2" in loadings.columns:
        save_ranked_lollipop_plot(
            _top_component_loadings(loadings, "PC2", top_n=20),
            output_dir / "pc2_top_loadings.png",
            title="Strongest absolute gene loadings on PC2",
            subtitle="Genes are ranked by absolute contribution to the second principal component",
            xlabel="Absolute loading",
            ylabel="Gene",
            color="#A85D43",
        )
    stability_summary: dict[str, object] = {}
    n_stability_components = min(args.pca_stability_components, len(score_columns))
    if args.stability_resamples > 0 and n_stability_components > 0:
        component_stability, subspace_stability = _pca_subsample_stability(
            top2000,
            reference_components=pca.components_[:n_stability_components],
            reference_explained_variance=pca.explained_variance_ratio_[:n_stability_components],
            n_resamples=args.stability_resamples,
            subsample_fraction=args.stability_fraction,
            seed=args.seed,
        )
        component_stability.to_csv(output_dir / "pca_subsample_stability.csv", index=False)
        subspace_stability.to_csv(output_dir / "pca_subspace_stability.csv", index=False)
        component_summary = (
            component_stability.groupby("component", sort=False)
            .agg(
                mean_abs_loading_correlation=("abs_loading_correlation", "mean"),
                min_abs_loading_correlation=("abs_loading_correlation", "min"),
                mean_abs_explained_variance_diff=("abs_explained_variance_diff", "mean"),
            )
            .reset_index()
        )
        component_summary.to_csv(output_dir / "pca_stability_component_summary.csv", index=False)
        save_metric_heatmap(
            component_summary,
            output_dir / "pca_stability_component_heatmap.png",
            index_col="component",
            metric_cols=[
                "mean_abs_loading_correlation",
                "min_abs_loading_correlation",
            ],
            title="PCA subsampling stability",
            subtitle="Agreement between the full-data components and repeated subsample refits",
            fill_label="Similarity",
        )
        save_ranked_lollipop_plot(
            component_summary.set_index("component")["mean_abs_loading_correlation"],
            output_dir / "pca_stability_mean_loading_similarity.png",
            title="Mean PCA loading stability by component",
            subtitle="Higher values indicate more reproducible loading directions across subsamples",
            xlabel="Mean absolute loading correlation",
            ylabel="Principal component",
            color="#6E8D64",
        )
        stability_summary = {
            "n_resamples": int(args.stability_resamples),
            "subsample_fraction": float(args.stability_fraction),
            "components_compared": int(n_stability_components),
            "mean_subspace_similarity": float(subspace_stability["mean_principal_cosine"].mean()),
            "min_subspace_similarity": float(subspace_stability["min_principal_cosine"].min()),
            "mean_abs_loading_correlation_by_component": {
                row["component"]: float(row["mean_abs_loading_correlation"])
                for _, row in component_summary.iterrows()
            },
        }

    X_graph = _stable_standardize(top500)
    empirical_covariance = np.cov(X_graph, rowvar=False)
    ledroit = LedoitWolf().fit(X_graph)
    oas = OAS().fit(X_graph)
    graph_summary: dict[str, object] = {}

    graph_model = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", message="invalid value encountered in subtract", category=RuntimeWarning)
            graph_model = GraphicalLassoCV(max_iter=500, tol=1e-3, enet_tol=1e-3).fit(X_graph)
        graph_summary["graphical_lasso_alpha"] = float(graph_model.alpha_)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        graph_summary["graphical_lasso_error"] = str(exc)

    covariance_summary = {
        "empirical_condition_number": float(np.linalg.cond(empirical_covariance + np.eye(empirical_covariance.shape[0]) * 1e-6)),
        "ledoit_wolf_shrinkage": float(ledroit.shrinkage_),
        "oas_shrinkage": float(oas.shrinkage_),
        "top2000_cleaning_summary": top2000_cleaning,
        "top500_cleaning_summary": top500_cleaning,
        "pca_stability_summary": stability_summary,
        **graph_summary,
    }
    save_json(covariance_summary, output_dir / "covariance_summary.json")

    subset_genes = top500.columns[:50].tolist()
    subset_index = [top500.columns.get_loc(gene) for gene in subset_genes]
    save_heatmap(
        empirical_covariance[np.ix_(subset_index, subset_index)],
        output_dir / "empirical_covariance_heatmap_top50.png",
        title="Empirical covariance on top-500 subset (first 50 genes)",
        xlabel="Gene",
        ylabel="Gene",
        xticklabels=subset_genes,
        yticklabels=subset_genes,
        cmap="coolwarm",
    )
    save_heatmap(
        ledroit.covariance_[np.ix_(subset_index, subset_index)],
        output_dir / "ledoit_wolf_covariance_heatmap_top50.png",
        title="Ledoit-Wolf covariance on top-500 subset (first 50 genes)",
        xlabel="Gene",
        ylabel="Gene",
        xticklabels=subset_genes,
        yticklabels=subset_genes,
        cmap="coolwarm",
    )

    if graph_model is not None:
        partial_correlations = _precision_to_partial_correlations(graph_model.precision_)
        save_heatmap(
            partial_correlations[np.ix_(subset_index, subset_index)],
            output_dir / "graphical_lasso_partial_corr_heatmap_top50.png",
            title="Graphical-lasso partial correlations (first 50 genes)",
            xlabel="Gene",
            ylabel="Gene",
            xticklabels=subset_genes,
            yticklabels=subset_genes,
            cmap="coolwarm",
        )
        edges = _top_precision_edges(partial_correlations, top500.columns.tolist(), top_n=250)
        edges.to_csv(output_dir / "graphical_lasso_top_edges.csv", index=False)
        top_edges = edges.head(30).copy()
        top_edges.index = top_edges.apply(lambda row: f"{row['gene_a']} - {row['gene_b']}", axis=1)
        save_ranked_lollipop_plot(
            top_edges["abs_partial_correlation"],
            output_dir / "graphical_lasso_top_edges_ranked.png",
            title="Largest graphical-lasso partial correlations",
            subtitle="Top edges ranked by absolute partial-correlation magnitude",
            xlabel="Absolute partial correlation",
            ylabel="Gene-pair edge",
            color="#6E5876",
        )
        stability = _bootstrap_edge_stability(
            X_graph,
            top500.columns.tolist(),
            alpha=float(graph_model.alpha_),
            n_resamples=args.stability_resamples,
            seed=args.seed,
        )
        stability.to_csv(output_dir / "graphical_lasso_edge_stability.csv", index=False)
        if not stability.empty:
            top_stable = stability.head(30).copy()
            top_stable.index = top_stable.apply(lambda row: f"{row['gene_a']} - {row['gene_b']}", axis=1)
            save_ranked_lollipop_plot(
                top_stable["selection_rate"],
                output_dir / "graphical_lasso_edge_stability_top30.png",
                title="Most stable graphical-lasso edges",
                subtitle="Bootstrap selection frequency highlights edges that persist across resamples",
                xlabel="Selection rate",
                ylabel="Gene-pair edge",
                color="#6E8D64",
            )


def _stable_standardize(frame: pd.DataFrame) -> np.ndarray:
    matrix = np.asarray(frame, dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = StandardScaler().fit_transform(matrix)
    scaled = np.asarray(scaled, dtype=np.float64)
    if not np.isfinite(scaled).all():
        raise ValueError("Encountered non-finite values after standardization.")
    return scaled


def _top_component_loadings(loadings: pd.DataFrame, component: str, *, top_n: int) -> pd.Series:
    series = loadings[component].abs().sort_values(ascending=False).head(top_n).sort_values(ascending=True)
    series.name = component
    return series


def _pca_subsample_stability(
    X: pd.DataFrame,
    *,
    reference_components: np.ndarray,
    reference_explained_variance: np.ndarray,
    n_resamples: int,
    subsample_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_rows = X.shape[0]
    n_compare = int(reference_components.shape[0])
    sample_size = max(n_compare + 5, int(round(subsample_fraction * n_rows)))
    sample_size = min(sample_size, n_rows)
    component_rows: list[dict[str, float | int | str]] = []
    subspace_rows: list[dict[str, float | int]] = []

    for resample_idx in range(1, n_resamples + 1):
        sample_ids = rng.choice(X.index.to_numpy(), size=sample_size, replace=False)
        X_sample = X.loc[sample_ids]
        sample_scaled = _stable_standardize(X_sample)
        sample_pca = PCA(n_components=n_compare, random_state=int(rng.integers(0, 1_000_000)))
        sample_pca.fit(sample_scaled)

        singular_values = np.linalg.svd(reference_components @ sample_pca.components_.T, compute_uv=False)
        subspace_rows.append(
            {
                "resample": resample_idx,
                "sample_size": sample_size,
                "mean_principal_cosine": float(np.mean(singular_values)),
                "min_principal_cosine": float(np.min(singular_values)),
            }
        )

        for component_idx in range(n_compare):
            reference = reference_components[component_idx]
            ref_std = float(np.std(reference))
            sample_component = sample_pca.components_[component_idx]
            sample_std = float(np.std(sample_component))
            if ref_std == 0 or sample_std == 0:
                correlation = float("nan")
            else:
                correlation = float(np.corrcoef(reference, sample_component)[0, 1])
            explained_difference = float(sample_pca.explained_variance_ratio_[component_idx] - reference_explained_variance[component_idx])
            component_rows.append(
                {
                    "resample": resample_idx,
                    "component": f"PC{component_idx + 1}",
                    "loading_correlation": correlation,
                    "abs_loading_correlation": abs(correlation) if np.isfinite(correlation) else float("nan"),
                    "explained_variance_difference": explained_difference,
                    "abs_explained_variance_diff": abs(explained_difference),
                }
            )

    return pd.DataFrame(component_rows), pd.DataFrame(subspace_rows)


def _precision_to_partial_correlations(precision: np.ndarray) -> np.ndarray:
    diagonal = np.sqrt(np.diag(precision))
    denominator = np.outer(diagonal, diagonal)
    partial = -precision / denominator
    np.fill_diagonal(partial, 1.0)
    return partial


def _top_precision_edges(partial: np.ndarray, genes: list[str], top_n: int = 250) -> pd.DataFrame:
    records = []
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            weight = float(partial[i, j])
            if abs(weight) > 1e-6:
                records.append((genes[i], genes[j], weight, abs(weight)))
    records.sort(key=lambda row: row[3], reverse=True)
    return pd.DataFrame(records[:top_n], columns=["gene_a", "gene_b", "partial_correlation", "abs_partial_correlation"])


def _bootstrap_edge_stability(
    X: np.ndarray,
    genes: list[str],
    *,
    alpha: float,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    edge_counts: dict[tuple[str, str], int] = {}
    successful = 0
    for _ in range(n_resamples):
        sample_indices = rng.integers(0, X.shape[0], size=X.shape[0])
        X_sample = X[sample_indices]
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", message="invalid value encountered in subtract", category=RuntimeWarning)
                model = GraphicalLasso(alpha=alpha, max_iter=500, tol=1e-3, enet_tol=1e-3).fit(X_sample)
        except Exception:  # pragma: no cover - runtime safeguard
            continue
        successful += 1
        partial = _precision_to_partial_correlations(model.precision_)
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                if abs(partial[i, j]) > 1e-6:
                    key = (genes[i], genes[j])
                    edge_counts[key] = edge_counts.get(key, 0) + 1
    rows = [
        {
            "gene_a": gene_a,
            "gene_b": gene_b,
            "selected_count": count,
            "selection_rate": count / max(successful, 1),
        }
        for (gene_a, gene_b), count in edge_counts.items()
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["selection_rate", "selected_count"], ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    main()
