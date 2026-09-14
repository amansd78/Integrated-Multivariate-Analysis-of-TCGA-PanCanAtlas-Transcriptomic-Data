# Integrated Multivariate Analysis of TCGA PanCanAtlas Transcriptomic Data

An end-to-end analysis pipeline for 10,055 patients across 32 cancer types using PCA, covariance regularization, multivariate ridge regression, classification, and clustering. Logistic regression achieved a held-out macro-F1 score of 0.893.

See [`README_tcga_pipeline.md`](README_tcga_pipeline.md) for the pipeline stages, execution order, command-line examples, and generated outputs.

## Data

This project uses the official open-access TCGA PanCanAtlas files published by the NCI Genomic Data Commons.

The repository includes the clinical tables, GDC manifest, and RNA expression matrix. Because the RNA matrix is approximately 1.88 GB, it is stored using Git Large File Storage (Git LFS). Install Git LFS before cloning if you want Git to retrieve that file automatically:

```bash
git lfs install
```

The data can also be downloaded directly from the GDC using the included script.

### Download from the source

Run:

```bash
bash scripts/download_tcga_pancanatlas.sh
```

The script downloads these files into `data/raw/tcga_pancanatlas/`:

- `EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv`
- `clinical_PANCAN_patient_with_followup.tsv`
- `TCGA-CDR-SupplementalTableS1.xlsx`
- `PanCan-General_Open_GDC-Manifest_2.txt`

### Why these files

- The RNA matrix is the main high-dimensional feature table for PCA, covariance estimation, graphical models, classification, and clustering.
- The clinical files provide survival and outcome labels for regression and classification tasks.

### Source

- PanCanAtlas publication page: <https://gdc.cancer.gov/about-data/publications/pancanatlas>
