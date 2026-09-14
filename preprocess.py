from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from stat662_tcga.datasets import build_sample_manifest, load_expression_for_samples
from stat662_tcga.preprocessing import (
    PreprocessingConfig,
    apply_expression_transform,
    config_to_dict,
    create_supervised_split,
    decide_expression_transform,
    load_tabular_dataset,
    remove_low_variance_genes,
    sanitize_expression_matrix,
    save_expression_bundle,
    select_top_variable_genes,
)
from stat662_tcga.utils import ensure_directory, save_json
from stat662_tcga.visualization import save_bar_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the TCGA representative-tumor expression analysis sets.")
    parser.add_argument("--raw-dir", required=True, help="Directory containing the PanCanAtlas raw files.")
    parser.add_argument("--manifest", help="Existing sample manifest parquet/csv. If omitted it will be rebuilt.")
    parser.add_argument("--output-dir", required=True, help="Directory for processed artifacts.")
    parser.add_argument("--top-genes", type=int, default=2000)
    parser.add_argument("--graph-genes", type=int, default=500)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transform", choices=["auto", "none", "log1p"], default="auto")
    parser.add_argument("--variance-epsilon", type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    config = PreprocessingConfig(
        top_genes=args.top_genes,
        graph_genes=args.graph_genes,
        test_size=args.test_size,
        random_state=args.seed,
        variance_epsilon=args.variance_epsilon,
        transform=args.transform,
    )

    if args.manifest:
        manifest = load_tabular_dataset(args.manifest)
    else:
        manifest = build_sample_manifest(args.raw_dir)

    representative = (
        manifest.loc[manifest["representative_tumor_sample"].fillna(False)]
        .copy()
        .drop_duplicates(subset=["sample_barcode"])
    )
    representative = representative[representative["acronym"].notna()].copy()
    representative["sample_barcode"] = representative["sample_barcode"].astype(str)
    representative = representative.sort_values("sample_barcode").reset_index(drop=True)
    if representative.empty:
        raise ValueError("No representative tumor samples with cancer-type labels were found.")

    expression_path = Path(args.raw_dir) / "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv"
    expression = load_expression_for_samples(expression_path, representative["sample_barcode"].tolist())
    representative = representative.set_index("sample_barcode").loc[expression.index].reset_index()

    effective_transform = args.transform if args.transform != "auto" else decide_expression_transform(expression)
    transformed = apply_expression_transform(expression, config.transform)
    transformed, cleaning_summary = sanitize_expression_matrix(transformed)
    representative = representative.set_index("sample_barcode").loc[transformed.index].reset_index()
    transformed = remove_low_variance_genes(transformed, variance_epsilon=config.variance_epsilon)
    top2000 = select_top_variable_genes(transformed, config.top_genes)
    top500 = select_top_variable_genes(transformed, config.graph_genes)

    label_series = representative.set_index("sample_barcode")["acronym"]
    split = create_supervised_split(
        label_series,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    representative.to_parquet(output_dir / "representative_manifest.parquet", index=False)
    representative.to_csv(output_dir / "representative_manifest.csv", index=False)
    save_expression_bundle(output_dir / "tcga_clean_expression_full.npz", transformed)
    save_expression_bundle(output_dir / "tcga_top2000_expression.npz", top2000)
    save_expression_bundle(output_dir / "tcga_top500_expression.npz", top500)

    train_counts = label_series.loc[split["train_sample_ids"]].value_counts().sort_index()
    test_counts = label_series.loc[split["test_sample_ids"]].value_counts().sort_index() if split["test_sample_ids"] else pd.Series(dtype=int)
    train_counts.rename_axis("acronym").reset_index(name="count").to_csv(output_dir / "train_cancer_counts.csv", index=False)
    test_counts.rename_axis("acronym").reset_index(name="count").to_csv(output_dir / "test_cancer_counts.csv", index=False)
    save_bar_plot(
        label_series.value_counts().head(25),
        output_dir / "representative_cancer_counts.png",
        title="Representative tumor sample counts by cancer type",
        xlabel="Cancer type",
        ylabel="Count",
        rotation=45,
    )

    save_json(
        {
            "config": config_to_dict(config),
            "raw_dir": str(Path(args.raw_dir).resolve()),
            "sample_count": int(transformed.shape[0]),
            "gene_count_after_variance_filter": int(transformed.shape[1]),
            "top2000_gene_count": int(top2000.shape[1]),
            "top500_gene_count": int(top500.shape[1]),
            "train_count": int(len(split["train_sample_ids"])),
            "test_count": int(len(split["test_sample_ids"])),
            "rare_train_only_count": int(len(split["rare_train_only_sample_ids"])),
            "train_sample_ids": split["train_sample_ids"],
            "test_sample_ids": split["test_sample_ids"],
            "rare_train_only_sample_ids": split["rare_train_only_sample_ids"],
        },
        output_dir / "supervised_split.json",
    )
    save_json(
        {
            "transform_used": effective_transform,
            "representative_patients": int(representative["patient_barcode"].nunique()),
            "cancer_types": int(label_series.nunique()),
            "full_expression_cache": str((output_dir / "tcga_clean_expression_full.npz").resolve()),
            "cleaning_summary": cleaning_summary,
        },
        output_dir / "preprocess_summary.json",
    )


if __name__ == "__main__":
    main()
