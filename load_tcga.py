from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from stat662_tcga.datasets import build_sample_manifest
from stat662_tcga.utils import ensure_directory, save_json
from stat662_tcga.visualization import save_bar_plot, save_missingness_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the merged TCGA sample manifest.")
    parser.add_argument("--raw-dir", required=True, help="Directory containing the downloaded PanCanAtlas files.")
    parser.add_argument("--output-dir", required=True, help="Directory for manifest outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)

    manifest = build_sample_manifest(args.raw_dir)
    manifest.to_parquet(output_dir / "tcga_sample_manifest.parquet", index=False)
    manifest.to_csv(output_dir / "tcga_sample_manifest.csv", index=False)

    sample_type_counts = manifest["sample_type_label"].fillna("Unknown").value_counts()
    sample_type_counts.rename_axis("sample_type").reset_index(name="count").to_csv(
        output_dir / "sample_type_counts.csv",
        index=False,
    )

    representative = manifest[manifest["representative_tumor_sample"]].copy()
    cancer_counts = representative["acronym"].dropna().value_counts()
    cancer_counts.rename_axis("acronym").reset_index(name="count").to_csv(
        output_dir / "representative_cancer_type_counts.csv",
        index=False,
    )

    missingness = manifest.isna().mean().sort_values(ascending=False).rename("missing_fraction")
    missingness.rename_axis("column").reset_index().to_csv(output_dir / "manifest_missingness.csv", index=False)

    save_bar_plot(
        sample_type_counts,
        output_dir / "sample_type_counts.png",
        title="TCGA sample-type composition",
        xlabel="Sample type",
        ylabel="Count",
        rotation=45,
    )
    if not cancer_counts.empty:
        save_bar_plot(
            cancer_counts.head(20),
            output_dir / "representative_cancer_type_counts.png",
            title="Representative tumor profiles by cancer type",
            xlabel="Cancer type",
            ylabel="Count",
            rotation=45,
        )
    save_missingness_plot(manifest, output_dir / "manifest_missingness_top20.png")

    save_json(
        {
            "raw_dir": str(Path(args.raw_dir).resolve()),
            "manifest_rows": int(len(manifest)),
            "unique_patients": int(manifest["patient_barcode"].nunique()),
            "tumor_sample_rows": int(manifest["is_tumor_sample"].fillna(False).sum()),
            "representative_tumor_rows": int(manifest["representative_tumor_sample"].sum()),
            "representative_cancer_types": int(representative["acronym"].dropna().nunique()),
        },
        output_dir / "load_summary.json",
    )


if __name__ == "__main__":
    main()
