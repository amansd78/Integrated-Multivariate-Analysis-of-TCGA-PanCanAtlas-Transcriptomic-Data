# STAT 662 TCGA PanCanAtlas Pipeline

This code implements the project proposed in `stat662_tcga_proposal.tex`.

## Main scripts

- `load_tcga.py`: build the merged TCGA sample manifest from RNA, clinical, and CDR files
- `preprocess.py`: retain one tumor profile per patient, cache the cleaned expression matrix, and create train/test splits
- `pca_covariance.py`: run PCA, shrinkage covariance estimation, and graphical-lasso diagnostics
- `regression.py`: relate top PC scores to clinical covariates using multivariate regression and blockwise association summaries
- `classification.py`: benchmark cancer-type classifiers on the shared top-gene space with training-only scaling and PCA
- `clustering.py`: compare PCA-based clustering methods and stability
- `plots_tables.py`: assemble summary tables and comparison plots from prior outputs
- `run_tcga_pipeline.py`: run the full pipeline in order with `tqdm` progress bars and per-step timing

## Install

```bash
python3 -m pip install -e .
```

## Execution order

Run these in this order:

1. `load_tcga.py`
2. `preprocess.py`
3. `pca_covariance.py`
4. `regression.py`, `classification.py`, and `clustering.py`
5. `plots_tables.py`

The only strict dependency is that steps `3` and `4` require the processed outputs from step `2`, and step `5` requires outputs from all prior steps. After `preprocess.py`, you can run `pca_covariance.py`, `regression.py`, `classification.py`, and `clustering.py` in any order.

## One-command pipeline

```bash
python3 run_tcga_pipeline.py --raw-dir data/raw/tcga_pancanatlas
```

By default this writes to `artifacts/tcga` and shows:

- overall pipeline progress
- a live `tqdm` timer for each section
- a machine-readable timing summary at `artifacts/tcga/pipeline_run_summary.json`
- proposal-aligned defaults for shared top-gene classification, PCA subsampling stability, and regression permutation summaries

Useful options:

- `--artifacts-root artifacts/tcga_final`
- `--start-at preprocess`
- `--stop-after classification`
- `--python .venv/bin/python`
- `--classification-feature-space training_selected_top_genes`

## Suggested commands

```bash
python3 run_tcga_pipeline.py --raw-dir data/raw/tcga_pancanatlas

python3 load_tcga.py --raw-dir data/raw/tcga_pancanatlas --output-dir artifacts/tcga/load
python3 preprocess.py --raw-dir data/raw/tcga_pancanatlas --manifest artifacts/tcga/load/tcga_sample_manifest.parquet --output-dir artifacts/tcga/preprocess
python3 pca_covariance.py --processed-dir artifacts/tcga/preprocess --output-dir artifacts/tcga/pca_covariance
python3 regression.py --processed-dir artifacts/tcga/preprocess --output-dir artifacts/tcga/regression
python3 classification.py --processed-dir artifacts/tcga/preprocess --output-dir artifacts/tcga/classification
python3 clustering.py --processed-dir artifacts/tcga/preprocess --output-dir artifacts/tcga/clustering
python3 plots_tables.py --processed-dir artifacts/tcga/preprocess --load-dir artifacts/tcga/load --pca-dir artifacts/tcga/pca_covariance --regression-dir artifacts/tcga/regression --classification-dir artifacts/tcga/classification --clustering-dir artifacts/tcga/clustering --output-dir artifacts/tcga/report
```

## Primary outputs

- sample manifest with representative tumor selection
- cleaned expression matrix cache
- top-2000 and top-500 gene subsets
- stratified supervised split metadata
- PCA scores, scree plots, subsampling-stability summaries, covariance summaries, and graph-stability tables
- multivariate regression summaries and covariate-block contribution diagnostics
- classifier comparison tables, confusion matrices, and held-out metrics
- clustering comparison tables, contingency heatmaps, and stability summaries

## Figure style

All project figures are now rendered through `plotnine` with a consistent editorial theme. Each plotting function writes both:

- high-resolution `.png`
- vector `.pdf`
