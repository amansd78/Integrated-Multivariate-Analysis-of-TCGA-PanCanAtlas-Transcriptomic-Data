from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from stat662_tcga.preprocessing import load_tabular_dataset
from stat662_tcga.utils import ensure_directory, save_json
from stat662_tcga.visualization import (
    prettify_label,
    save_bar_plot,
    save_metric_heatmap,
    save_model_comparison_plot,
    save_paired_comparison_plot,
    save_ranked_lollipop_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble summary tables and report plots for the TCGA project.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--load-dir", required=True)
    parser.add_argument("--pca-dir", required=True)
    parser.add_argument("--regression-dir", required=True)
    parser.add_argument("--classification-dir", required=True)
    parser.add_argument("--clustering-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)

    manifest = load_tabular_dataset(Path(args.processed_dir) / "representative_manifest.parquet")
    cancer_counts = manifest["acronym"].value_counts().rename_axis("acronym").reset_index(name="count")
    cancer_counts.to_csv(output_dir / "report_cancer_counts.csv", index=False)
    save_bar_plot(
        cancer_counts.set_index("acronym")["count"].head(25),
        output_dir / "report_cancer_counts.png",
        title="Representative tumor sample counts",
        xlabel="Cancer type",
        ylabel="Count",
        rotation=45,
    )

    explained = pd.read_csv(Path(args.pca_dir) / "pca_explained_variance.csv")
    explained.head(10).to_csv(output_dir / "top10_pca_explained_variance.csv", index=False)
    save_ranked_lollipop_plot(
        explained.head(15).set_index("component")["explained_variance_ratio"],
        output_dir / "top15_pc_explained_variance.png",
        title="Explained variance across the leading principal components",
        subtitle="The first 15 components capture the dominant axes of transcriptomic variation",
        xlabel="Explained variance ratio",
        ylabel="Principal component",
        color="#325C80",
    )
    pca_stability_summary_path = Path(args.pca_dir) / "pca_stability_component_summary.csv"
    pca_stability_component_summary = None
    if pca_stability_summary_path.exists():
        pca_stability_component_summary = pd.read_csv(pca_stability_summary_path)
        save_ranked_lollipop_plot(
            pca_stability_component_summary.set_index("component")["mean_abs_loading_correlation"],
            output_dir / "pca_stability_mean_loading_similarity.png",
            title="PCA loading stability across repeated subsamples",
            subtitle="Higher values indicate more reproducible principal directions",
            xlabel="Mean absolute loading correlation",
            ylabel="Principal component",
            color="#6E8D64",
        )
        save_metric_heatmap(
            pca_stability_component_summary,
            output_dir / "pca_stability_component_heatmap.png",
            index_col="component",
            metric_cols=["mean_abs_loading_correlation", "min_abs_loading_correlation"],
            title="PCA subsampling stability",
            subtitle="Agreement between the full-data PCs and repeated subsample refits",
            fill_label="Similarity",
        )

    model_comparison = pd.read_csv(Path(args.classification_dir) / "model_comparison.csv")
    classification_summary = json.loads((Path(args.classification_dir) / "classification_summary.json").read_text(encoding="utf-8"))
    model_comparison.to_csv(output_dir / "classification_model_comparison.csv", index=False)
    save_model_comparison_plot(
        model_comparison,
        output_dir / "classification_model_comparison.png",
        title="Held-out cancer-type classification",
        subtitle="Dark green marks accuracy; ochre marks macro-F1 on the held-out cohort",
    )
    save_metric_heatmap(
        model_comparison.sort_values("macro_f1", ascending=False),
        output_dir / "classification_model_heatmap.png",
        index_col="model",
        metric_cols=["accuracy", "macro_f1", "weighted_f1", "best_cv_macro_f1"],
        title="Classifier scorecard",
        subtitle="Held-out performance and cross-validated ranking in a single view",
        fill_label="Score",
    )
    best_classifier = str(model_comparison.iloc[0]["model"])
    best_classifier_metrics = json.loads((Path(args.classification_dir) / f"{best_classifier}_metrics.json").read_text(encoding="utf-8"))
    best_classifier_report = best_classifier_metrics["classification_report"]
    per_class_f1 = (
        pd.Series(
            {
                label: metrics["f1-score"]
                for label, metrics in best_classifier_report.items()
                if isinstance(metrics, dict) and "f1-score" in metrics and label not in {"macro avg", "weighted avg", "micro avg"}
            }
        )
        .sort_values(ascending=True)
    )
    save_ranked_lollipop_plot(
        per_class_f1,
        output_dir / "best_classifier_per_class_f1.png",
        title=f"Per-class F1 for {prettify_label(best_classifier)}",
        subtitle="Lower-performing cancer types are surfaced first for diagnostic review",
        xlabel="F1 score",
        ylabel="Cancer type",
        color="#B4852B",
    )

    clustering_summary = pd.read_csv(Path(args.clustering_dir) / "clustering_summary.csv")
    clustering_summary.to_csv(output_dir / "clustering_summary.csv", index=False)
    save_model_comparison_plot(
        clustering_summary.rename(columns={"ari": "macro_f1", "silhouette": "accuracy"})[["model", "accuracy", "macro_f1"]],
        output_dir / "clustering_summary_report.png",
        title="Clustering quality across unsupervised methods",
        subtitle="Dark green marks silhouette; ochre marks adjusted Rand index",
        left_label="Silhouette",
        right_label="ARI",
    )
    save_metric_heatmap(
        clustering_summary,
        output_dir / "clustering_metric_heatmap.png",
        index_col="model",
        metric_cols=["silhouette", "ari", "nmi", "subsample_ari"],
        title="Clustering scorecard",
        subtitle="Internal separation, external agreement, and stability across resamples",
        fill_label="Score",
    )
    train_counts = pd.read_csv(Path(args.processed_dir) / "train_cancer_counts.csv").rename(columns={"count": "train_count"})
    test_counts = pd.read_csv(Path(args.processed_dir) / "test_cancer_counts.csv").rename(columns={"count": "test_count"})
    split_counts = train_counts.merge(test_counts, on="acronym", how="outer").fillna(0)
    split_counts[["train_count", "test_count"]] = split_counts[["train_count", "test_count"]].astype(int)
    split_counts = split_counts.sort_values("train_count", ascending=False).head(20)
    save_paired_comparison_plot(
        split_counts,
        output_dir / "train_test_cancer_balance_top20.png",
        category_col="acronym",
        left_col="train_count",
        right_col="test_count",
        title="Train versus test representation by cancer type",
        subtitle="The top 20 cohorts are shown to audit stratified split fidelity",
        left_label="Train",
        right_label="Test",
        y_label="Cancer type",
        x_label="Samples",
        left_color="#173F35",
        right_color="#A85D43",
        value_limits=None,
    )
    stable_edges_path = Path(args.pca_dir) / "graphical_lasso_edge_stability.csv"
    if stable_edges_path.exists():
        stable_edges = pd.read_csv(stable_edges_path)
        if not stable_edges.empty:
            top_edges = stable_edges.head(25).copy()
            top_edges.index = top_edges.apply(lambda row: f"{row['gene_a']} - {row['gene_b']}", axis=1)
            save_ranked_lollipop_plot(
                top_edges["selection_rate"],
                output_dir / "top_stable_graphical_lasso_edges.png",
                title="Most stable graphical-lasso edges",
                subtitle="Bootstrap selection frequencies identify the most reproducible inferred dependencies",
                xlabel="Selection rate",
                ylabel="Gene-pair edge",
                color="#6E8D64",
            )

    regression_summary_path = Path(args.regression_dir) / "regression_summary.json"
    regression_summary = json.loads(regression_summary_path.read_text(encoding="utf-8"))
    regression_block_summary_path = Path(args.regression_dir) / "regression_block_summary.csv"
    if regression_block_summary_path.exists():
        regression_block_summary = pd.read_csv(regression_block_summary_path)
        save_ranked_lollipop_plot(
            regression_block_summary.set_index("block")["cv_delta_r2"],
            output_dir / "regression_block_cv_delta_r2.png",
            title="Cross-validated multivariate contribution by covariate block",
            subtitle="Each point shows the loss in variance-weighted R2 when the block is removed",
            xlabel="Delta variance-weighted R2",
            ylabel="Covariate block",
            color="#173F35",
        )
        if "p_value" in regression_block_summary.columns and regression_block_summary["p_value"].notna().any():
            save_ranked_lollipop_plot(
                regression_block_summary.dropna(subset=["p_value"]).set_index("block")["p_value"],
                output_dir / "regression_block_p_values.png",
                title="Permutation p-values for regression covariate blocks",
                subtitle="Smaller values indicate stronger multivariate association with the latent PC response",
                xlabel="Permutation p-value",
                ylabel="Covariate block",
                color="#A85D43",
            )
        strongest_regression_block = str(regression_block_summary.sort_values("cv_delta_r2", ascending=False).iloc[0]["block"])
    else:
        strongest_regression_block = "not available"

    report_lines = [
        "# TCGA STAT 662 Report Summary",
        "",
        "## Dataset",
        f"- Representative tumor samples: {len(manifest)}",
        f"- Cancer types: {manifest['acronym'].nunique()}",
        "",
        "## PCA",
        f"- PC1 explained variance: {explained.iloc[0]['explained_variance_ratio']:.4f}",
        f"- Top 10 PCs cumulative variance: {explained.head(10)['explained_variance_ratio'].sum():.4f}",
        (
            f"- Mean loading stability for PC1: {pca_stability_component_summary.loc[pca_stability_component_summary['component'] == 'PC1', 'mean_abs_loading_correlation'].iloc[0]:.4f}"
            if pca_stability_component_summary is not None and not pca_stability_component_summary.empty
            else "- PCA subsampling stability: not available"
        ),
        "",
        "## Classification",
        f"- Best held-out model: {best_classifier}",
        f"- Best held-out macro-F1: {model_comparison.iloc[0]['macro_f1']:.4f}",
        f"- Classifier feature space: {classification_summary['feature_space_mode']}",
        "",
        "## Clustering",
        f"- Best clustering method by ARI: {clustering_summary.iloc[0]['model']}",
        f"- Best clustering ARI: {clustering_summary.iloc[0]['ari']:.4f}",
        "",
        "## Regression",
        f"- Cross-validated variance-weighted R2: {float(regression_summary['cross_validated_variance_weighted_r2']):.4f}",
        f"- Strongest covariate block by CV delta R2: {strongest_regression_block}",
    ]
    (output_dir / "report_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    save_json(
        {
            "representative_samples": int(len(manifest)),
            "cancer_types": int(manifest["acronym"].nunique()),
            "best_classifier": model_comparison.iloc[0]["model"],
            "best_classifier_macro_f1": float(model_comparison.iloc[0]["macro_f1"]),
            "classification_feature_space": classification_summary["feature_space_mode"],
            "best_clustering_method": clustering_summary.iloc[0]["model"],
            "best_clustering_ari": float(clustering_summary.iloc[0]["ari"]),
            "regression_cv_r2": float(regression_summary["cross_validated_variance_weighted_r2"]),
            "strongest_regression_block": strongest_regression_block,
        },
        output_dir / "project_summary.json",
    )


if __name__ == "__main__":
    main()
