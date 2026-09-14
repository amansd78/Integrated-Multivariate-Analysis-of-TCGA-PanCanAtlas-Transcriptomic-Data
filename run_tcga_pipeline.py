from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for environments that have not installed tqdm yet
    class tqdm:  # type: ignore[no-redef]
        def __init__(
            self,
            total: int | None = None,
            desc: str = "",
            unit: str = "",
            dynamic_ncols: bool = False,
            position: int = 0,
            leave: bool = True,
        ) -> None:
            self.total = total
            self.desc = desc
            self.unit = unit
            self.n = 0.0
            self.postfix = ""
            self._last_reported = -1
            if desc:
                print(f"{desc} started", flush=True)

        def update(self, value: float = 1.0) -> None:
            self.n += value
            reported = int(self.n)
            if reported > self._last_reported and reported % 10 == 0:
                self._last_reported = reported
                suffix = f" | {self.postfix}" if self.postfix else ""
                unit = self.unit or "step"
                print(f"{self.desc}: {reported} {unit}{suffix}", flush=True)

        def set_postfix_str(self, text: str) -> None:
            self.postfix = text

        def close(self) -> None:
            if self.desc:
                suffix = f" | {self.postfix}" if self.postfix else ""
                print(f"{self.desc} finished{suffix}", flush=True)

        @staticmethod
        def write(message: str) -> None:
            print(message, flush=True)


PROJECT_ROOT = Path(__file__).resolve().parent
STEP_NAMES = [
    "load",
    "preprocess",
    "pca_covariance",
    "regression",
    "classification",
    "clustering",
    "report",
]


@dataclass(slots=True)
class PipelineStep:
    name: str
    label: str
    description: str
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full STAT 662 TCGA pipeline in the correct order.")
    parser.add_argument("--raw-dir", required=True, help="Directory containing the TCGA PanCanAtlas raw input files.")
    parser.add_argument("--artifacts-root", default="artifacts/tcga", help="Root directory for all pipeline outputs.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use for child scripts.")
    parser.add_argument("--start-at", choices=STEP_NAMES, default=STEP_NAMES[0], help="First pipeline step to execute.")
    parser.add_argument("--stop-after", choices=STEP_NAMES, default=STEP_NAMES[-1], help="Last pipeline step to execute.")
    parser.add_argument("--top-genes", type=int, default=2000)
    parser.add_argument("--graph-genes", type=int, default=500)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--transform", choices=["auto", "none", "log1p"], default="auto")
    parser.add_argument("--variance-epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=50, help="Number of PCs for pca_covariance.py.")
    parser.add_argument("--graph-stability-resamples", type=int, default=10)
    parser.add_argument("--pca-stability-fraction", type=float, default=0.80)
    parser.add_argument("--pca-stability-components", type=int, default=5)
    parser.add_argument("--regression-pcs", type=int, default=10)
    parser.add_argument("--regression-permutations", type=int, default=25)
    parser.add_argument(
        "--classification-feature-space",
        choices=["shared_top_gene_space", "training_selected_top_genes"],
        default="shared_top_gene_space",
    )
    parser.add_argument("--classification-pc-components", type=int, default=50)
    parser.add_argument("--classification-cv-folds", type=int, default=5)
    parser.add_argument("--classification-n-jobs", type=int, default=1)
    parser.add_argument("--clustering-pc-components", type=int, default=20)
    parser.add_argument("--clustering-stability-repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if STEP_NAMES.index(args.start_at) > STEP_NAMES.index(args.stop_after):
        raise ValueError("--start-at must come before or match --stop-after.")

    artifacts_root = Path(args.artifacts_root)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    steps = _select_steps(_build_steps(args), start_at=args.start_at, stop_after=args.stop_after)
    run_summary = {
        "status": "running",
        "python": args.python,
        "raw_dir": str(Path(args.raw_dir).resolve()),
        "artifacts_root": str(artifacts_root.resolve()),
        "started_at": _utc_now(),
        "steps": [],
    }
    summary_path = artifacts_root / "pipeline_run_summary.json"
    start_time = time.perf_counter()
    overall = tqdm(total=len(steps), desc="Pipeline", unit="step", dynamic_ncols=True, position=0)

    try:
        for index, step in enumerate(steps, start=1):
            tqdm.write(f"\n[{index}/{len(steps)}] {step.label}")
            tqdm.write(step.description)
            step_result = _run_step(step, position=1)
            run_summary["steps"].append(step_result)
            overall.update(1)
            overall.set_postfix_str(f"last={step.name} {step_result['elapsed_seconds']:.1f}s")
            _write_summary(run_summary, summary_path, status="running", started_at=run_summary["started_at"], total_seconds=time.perf_counter() - start_time)
    except subprocess.CalledProcessError as exc:
        run_summary["failed_step"] = step.name
        run_summary["failed_command"] = exc.cmd
        _write_summary(run_summary, summary_path, status="failed", started_at=run_summary["started_at"], total_seconds=time.perf_counter() - start_time)
        raise
    finally:
        overall.close()

    _write_summary(run_summary, summary_path, status="completed", started_at=run_summary["started_at"], total_seconds=time.perf_counter() - start_time)
    tqdm.write("\nPipeline completed successfully.")
    tqdm.write(f"Timing summary written to {summary_path}")


def _build_steps(args: argparse.Namespace) -> list[PipelineStep]:
    artifacts_root = Path(args.artifacts_root)
    load_dir = artifacts_root / "load"
    preprocess_dir = artifacts_root / "preprocess"
    pca_dir = artifacts_root / "pca_covariance"
    regression_dir = artifacts_root / "regression"
    classification_dir = artifacts_root / "classification"
    clustering_dir = artifacts_root / "clustering"
    report_dir = artifacts_root / "report"
    manifest_path = load_dir / "tcga_sample_manifest.parquet"

    return [
        PipelineStep(
            name="load",
            label="Load TCGA manifest",
            description="Build the merged manifest from RNA, clinical, and CDR inputs.",
            command=[
                args.python,
                "load_tcga.py",
                "--raw-dir",
                args.raw_dir,
                "--output-dir",
                str(load_dir),
            ],
        ),
        PipelineStep(
            name="preprocess",
            label="Preprocess expression data",
            description="Select representative tumors, clean the expression matrix, and create train/test splits.",
            command=[
                args.python,
                "preprocess.py",
                "--raw-dir",
                args.raw_dir,
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(preprocess_dir),
                "--top-genes",
                str(args.top_genes),
                "--graph-genes",
                str(args.graph_genes),
                "--test-size",
                str(args.test_size),
                "--transform",
                args.transform,
                "--variance-epsilon",
                str(args.variance_epsilon),
                "--seed",
                str(args.seed),
            ],
        ),
        PipelineStep(
            name="pca_covariance",
            label="PCA and covariance analysis",
            description="Run PCA, shrinkage covariance estimation, and graphical-lasso diagnostics.",
            command=[
                args.python,
                "pca_covariance.py",
                "--processed-dir",
                str(preprocess_dir),
                "--output-dir",
                str(pca_dir),
                "--pca-components",
                str(args.pca_components),
                "--stability-resamples",
                str(args.graph_stability_resamples),
                "--stability-fraction",
                str(args.pca_stability_fraction),
                "--pca-stability-components",
                str(args.pca_stability_components),
                "--seed",
                str(args.seed),
            ],
        ),
        PipelineStep(
            name="regression",
            label="Multivariate regression",
            description="Relate top principal components to clinical covariates.",
            command=[
                args.python,
                "regression.py",
                "--processed-dir",
                str(preprocess_dir),
                "--output-dir",
                str(regression_dir),
                "--num-pcs",
                str(args.regression_pcs),
                "--permutations",
                str(args.regression_permutations),
                "--seed",
                str(args.seed),
            ],
        ),
        PipelineStep(
            name="classification",
            label="Cancer-type classification",
            description="Benchmark supervised classifiers on the TCGA PC representation.",
            command=[
                args.python,
                "classification.py",
                "--processed-dir",
                str(preprocess_dir),
                "--output-dir",
                str(classification_dir),
                "--top-genes",
                str(args.top_genes),
                "--feature-space",
                args.classification_feature_space,
                "--pc-components",
                str(args.classification_pc_components),
                "--cv-folds",
                str(args.classification_cv_folds),
                "--n-jobs",
                str(args.classification_n_jobs),
                "--seed",
                str(args.seed),
            ],
        ),
        PipelineStep(
            name="clustering",
            label="Unsupervised clustering",
            description="Compare PCA-based clustering methods and subsample stability.",
            command=[
                args.python,
                "clustering.py",
                "--processed-dir",
                str(preprocess_dir),
                "--output-dir",
                str(clustering_dir),
                "--pc-components",
                str(args.clustering_pc_components),
                "--stability-repeats",
                str(args.clustering_stability_repeats),
                "--seed",
                str(args.seed),
            ],
        ),
        PipelineStep(
            name="report",
            label="Report assembly",
            description="Build final tables and editorial figures from all previous outputs.",
            command=[
                args.python,
                "plots_tables.py",
                "--processed-dir",
                str(preprocess_dir),
                "--load-dir",
                str(load_dir),
                "--pca-dir",
                str(pca_dir),
                "--regression-dir",
                str(regression_dir),
                "--classification-dir",
                str(classification_dir),
                "--clustering-dir",
                str(clustering_dir),
                "--output-dir",
                str(report_dir),
            ],
        ),
    ]


def _select_steps(steps: list[PipelineStep], *, start_at: str, stop_after: str) -> list[PipelineStep]:
    start_index = STEP_NAMES.index(start_at)
    stop_index = STEP_NAMES.index(stop_after)
    allowed = set(STEP_NAMES[start_index : stop_index + 1])
    return [step for step in steps if step.name in allowed]


def _run_step(step: PipelineStep, *, position: int) -> dict[str, object]:
    started_at = _utc_now()
    started_perf = time.perf_counter()
    progress = tqdm(
        total=None,
        desc=step.name,
        unit="s",
        dynamic_ncols=True,
        leave=True,
        position=position,
    )
    process = subprocess.Popen(step.command, cwd=PROJECT_ROOT)
    last_second = 0

    try:
        while True:
            return_code = process.poll()
            elapsed_seconds = time.perf_counter() - started_perf
            elapsed_whole = int(elapsed_seconds)
            if elapsed_whole > last_second:
                progress.update(elapsed_whole - last_second)
                last_second = elapsed_whole
            if return_code is not None:
                if elapsed_seconds > last_second:
                    progress.update(elapsed_seconds - last_second)
                progress.set_postfix_str(f"done {elapsed_seconds:.1f}s" if return_code == 0 else f"failed {elapsed_seconds:.1f}s")
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, step.command)
                break
            time.sleep(0.25)
    finally:
        progress.close()

    ended_at = _utc_now()
    return {
        "name": step.name,
        "label": step.label,
        "description": step.description,
        "command": step.command,
        "command_shell": shlex.join(step.command),
        "started_at": started_at,
        "completed_at": ended_at,
        "elapsed_seconds": round(time.perf_counter() - started_perf, 3),
        "status": "completed",
    }


def _write_summary(
    summary: dict[str, object],
    path: Path,
    *,
    status: str,
    started_at: str,
    total_seconds: float,
) -> None:
    payload = dict(summary)
    payload["status"] = status
    payload["started_at"] = started_at
    payload["completed_at"] = _utc_now()
    payload["total_seconds"] = round(total_seconds, 3)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
