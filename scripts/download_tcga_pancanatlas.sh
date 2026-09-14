#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
output_dir="${1:-${project_root}/data/raw/tcga_pancanatlas}"

manifest_url="https://gdc.cancer.gov/system/files/public/file/PanCan-General_Open_GDC-Manifest_2.txt"
base_url="https://api.gdc.cancer.gov/data"

mkdir -p "${output_dir}"

download() {
  local uuid="$1"
  local filename="$2"
  local target="${output_dir}/${filename}"

  if [[ -f "${target}" ]]; then
    printf 'Skipping existing file: %s\n' "${filename}"
    return
  fi

  printf 'Downloading %s\n' "${filename}"
  curl --fail --location --output "${target}" "${base_url}/${uuid}"
}

printf 'Output directory: %s\n' "${output_dir}"

manifest_target="${output_dir}/PanCan-General_Open_GDC-Manifest_2.txt"
if [[ ! -f "${manifest_target}" ]]; then
  printf 'Downloading manifest\n'
  curl --fail --location --output "${manifest_target}" "${manifest_url}"
else
  printf 'Skipping existing file: %s\n' "$(basename "${manifest_target}")"
fi

# Core analysis-ready files for a multivariate project.
download "3586c0da-64d0-4b74-a449-5ff4d9136611" "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv"
download "0fc78496-818b-4896-bd83-52db1f533c5c" "clinical_PANCAN_patient_with_followup.tsv"
download "1b5f413e-a8d1-4d10-92eb-7c4ae739ed81" "TCGA-CDR-SupplementalTableS1.xlsx"

cat <<EOF

Download complete.

Files:
  - EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv
  - clinical_PANCAN_patient_with_followup.tsv
  - TCGA-CDR-SupplementalTableS1.xlsx
  - PanCan-General_Open_GDC-Manifest_2.txt

Notes:
  - The RNA matrix is about 1.88 GB.
  - The clinical follow-up table is about 19 MB.
  - The TCGA-CDR outcome sheet is about 3 MB.
EOF
