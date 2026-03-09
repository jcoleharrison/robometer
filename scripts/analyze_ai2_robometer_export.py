#!/usr/bin/env python3
"""Analyze Robometer AI2 export JSONL files.

Computes per-episode and per-dataset correlation metrics from exported predictions,
including VOC-style correlation (Spearman between predicted progress and true
completion along trajectory prefixes).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr

from robometer.evals.eval_metrics_utils import compute_pearson, compute_spearman


@dataclass
class EpisodeMetric:
    dataset: str
    episode_index: int
    task_name: str | None
    instruction: str | None
    video_path: str | None
    num_points: int
    voc: float | None
    spearman: float | None
    pearson: float | None
    pred_final_progress: float | None
    true_final_completion: float | None
    error: str | None


def value_order_correlation(values: Iterable[float], true_values: Iterable[float]) -> float:
    """Instruction_GVL-compatible VOC helper (Spearman with NaN on degenerate inputs)."""
    v = np.asarray(list(values), dtype=float)
    t = np.asarray(list(true_values), dtype=float)

    if v.shape[0] != t.shape[0]:
        raise ValueError(f"Length mismatch: {v.shape[0]} vs {t.shape[0]}")
    if v.size < 2:
        return float("nan")
    if np.allclose(v, v[0]) or np.allclose(t, t[0]):
        return float("nan")

    corr = spearmanr(v, t)
    return float(corr.statistic if hasattr(corr, "statistic") else corr[0])


def _safe_float(x: Any) -> float | None:
    try:
        val = float(x)
        if math.isfinite(val):
            return val
        return None
    except Exception:
        return None


def _to_float_list(x: Any) -> list[float] | None:
    if not isinstance(x, list):
        return None
    out: list[float] = []
    for v in x:
        fv = _safe_float(v)
        if fv is None:
            return None
        out.append(fv)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Robometer AI2 export JSONL files and compute VOC/Spearman summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing *_predictions.jsonl from eval_ai2_robometer_export.py",
    )
    parser.add_argument(
        "--predictions-glob",
        default="*_predictions.jsonl",
        help="Glob to select prediction files inside input-dir.",
    )
    parser.add_argument(
        "--episode-analysis-name",
        default="analysis_per_episode.jsonl",
        help="Output filename for per-episode analysis records.",
    )
    parser.add_argument(
        "--summary-name",
        default="analysis_summary.json",
        help="Output filename for aggregate summary.",
    )
    parser.add_argument(
        "--dataset-csv-name",
        default="analysis_by_dataset.csv",
        help="Output filename for per-dataset metrics table.",
    )
    return parser.parse_args()


def _metric_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "count": 0,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": int(arr.size),
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    prediction_files = sorted(input_dir.glob(args.predictions_glob))
    if not prediction_files:
        raise FileNotFoundError(f"No files matched {args.predictions_glob} in {input_dir}")

    per_episode_records: list[dict[str, Any]] = []
    dataset_summaries: list[dict[str, Any]] = []

    global_voc: list[float] = []
    global_spearman: list[float] = []
    global_pearson: list[float] = []
    global_valid_records = 0
    global_error_records = 0
    global_metric_ready_records = 0

    for pred_file in prediction_files:
        dataset_name = pred_file.name.replace("_predictions.jsonl", "")

        total_records = 0
        valid_records = 0
        error_records = 0
        metric_ready_records = 0
        voc_values: list[float] = []
        spearman_values: list[float] = []
        pearson_values: list[float] = []

        with pred_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_records += 1

                rec = json.loads(line)
                rec_error = rec.get("error")
                if rec_error is not None:
                    error_records += 1
                    global_error_records += 1
                    metric = EpisodeMetric(
                        dataset=rec.get("dataset", dataset_name),
                        episode_index=int(rec.get("episode_index", -1)),
                        task_name=rec.get("task_name"),
                        instruction=rec.get("instruction"),
                        video_path=rec.get("video_path"),
                        num_points=0,
                        voc=None,
                        spearman=None,
                        pearson=None,
                        pred_final_progress=_safe_float(rec.get("pred_final_progress")),
                        true_final_completion=None,
                        error=str(rec_error),
                    )
                    per_episode_records.append(metric.__dict__)
                    continue

                valid_records += 1
                global_valid_records += 1

                pred_progress = _to_float_list(rec.get("pred_progress"))
                true_completion = _to_float_list(rec.get("true_completion"))

                voc_val: float | None = None
                spearman_val: float | None = None
                pearson_val: float | None = None
                num_points = 0
                true_final_completion: float | None = None
                metric_error: str | None = None

                if pred_progress is None or true_completion is None:
                    metric_error = "missing_or_invalid_progress_arrays"
                elif len(pred_progress) != len(true_completion):
                    metric_error = f"length_mismatch:{len(pred_progress)}_vs_{len(true_completion)}"
                elif len(pred_progress) < 2:
                    metric_error = "too_few_points"
                else:
                    num_points = len(pred_progress)
                    true_final_completion = true_completion[-1]

                    voc_raw = value_order_correlation(pred_progress, true_completion)
                    if math.isfinite(voc_raw):
                        voc_val = float(voc_raw)

                    sp_raw = compute_spearman(true_completion, pred_progress)
                    if math.isfinite(float(sp_raw)):
                        spearman_val = float(sp_raw)

                    pe_raw = compute_pearson(true_completion, pred_progress)
                    if math.isfinite(float(pe_raw)):
                        pearson_val = float(pe_raw)

                    if voc_val is not None or spearman_val is not None or pearson_val is not None:
                        metric_ready_records += 1
                        global_metric_ready_records += 1

                if voc_val is not None:
                    voc_values.append(voc_val)
                    global_voc.append(voc_val)
                if spearman_val is not None:
                    spearman_values.append(spearman_val)
                    global_spearman.append(spearman_val)
                if pearson_val is not None:
                    pearson_values.append(pearson_val)
                    global_pearson.append(pearson_val)

                metric = EpisodeMetric(
                    dataset=rec.get("dataset", dataset_name),
                    episode_index=int(rec.get("episode_index", -1)),
                    task_name=rec.get("task_name"),
                    instruction=rec.get("instruction"),
                    video_path=rec.get("video_path"),
                    num_points=num_points,
                    voc=voc_val,
                    spearman=spearman_val,
                    pearson=pearson_val,
                    pred_final_progress=_safe_float(rec.get("pred_final_progress")),
                    true_final_completion=true_final_completion,
                    error=metric_error,
                )
                per_episode_records.append(metric.__dict__)

        voc_stats = _metric_stats(voc_values)
        spearman_stats = _metric_stats(spearman_values)
        pearson_stats = _metric_stats(pearson_values)

        dataset_summary = {
            "dataset_name": dataset_name,
            "records_path": str(pred_file),
            "num_records": total_records,
            "valid_records": valid_records,
            "error_records": error_records,
            "metric_ready_records": metric_ready_records,
            "metrics": {
                "voc_mean": voc_stats["mean"],
                "voc_std": voc_stats["std"],
                "voc_min": voc_stats["min"],
                "voc_max": voc_stats["max"],
                "voc_valid_count": voc_stats["count"],
                "spearman_mean": spearman_stats["mean"],
                "spearman_std": spearman_stats["std"],
                "spearman_min": spearman_stats["min"],
                "spearman_max": spearman_stats["max"],
                "spearman_valid_count": spearman_stats["count"],
                "pearson_mean": pearson_stats["mean"],
                "pearson_std": pearson_stats["std"],
                "pearson_min": pearson_stats["min"],
                "pearson_max": pearson_stats["max"],
                "pearson_valid_count": pearson_stats["count"],
            },
        }
        dataset_summaries.append(dataset_summary)

    global_summary = {
        "analysis_time": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "num_datasets": len(dataset_summaries),
        "num_records": int(sum(d["num_records"] for d in dataset_summaries)),
        "valid_records": int(global_valid_records),
        "error_records": int(global_error_records),
        "metric_ready_records": int(global_metric_ready_records),
        "metrics": {
            "voc_mean": _metric_stats(global_voc)["mean"],
            "voc_std": _metric_stats(global_voc)["std"],
            "voc_min": _metric_stats(global_voc)["min"],
            "voc_max": _metric_stats(global_voc)["max"],
            "voc_valid_count": _metric_stats(global_voc)["count"],
            "spearman_mean": _metric_stats(global_spearman)["mean"],
            "spearman_std": _metric_stats(global_spearman)["std"],
            "spearman_min": _metric_stats(global_spearman)["min"],
            "spearman_max": _metric_stats(global_spearman)["max"],
            "spearman_valid_count": _metric_stats(global_spearman)["count"],
            "pearson_mean": _metric_stats(global_pearson)["mean"],
            "pearson_std": _metric_stats(global_pearson)["std"],
            "pearson_min": _metric_stats(global_pearson)["min"],
            "pearson_max": _metric_stats(global_pearson)["max"],
            "pearson_valid_count": _metric_stats(global_pearson)["count"],
        },
        "datasets": dataset_summaries,
    }

    per_episode_path = input_dir / args.episode_analysis_name
    with per_episode_path.open("w", encoding="utf-8") as f:
        for rec in per_episode_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary_path = input_dir / args.summary_name
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2)

    csv_path = input_dir / args.dataset_csv_name
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_name",
                "num_records",
                "valid_records",
                "error_records",
                "metric_ready_records",
                "voc_mean",
                "voc_std",
                "voc_valid_count",
                "spearman_mean",
                "spearman_std",
                "spearman_valid_count",
                "pearson_mean",
                "pearson_std",
                "pearson_valid_count",
            ],
        )
        writer.writeheader()
        for ds in dataset_summaries:
            m = ds["metrics"]
            writer.writerow(
                {
                    "dataset_name": ds["dataset_name"],
                    "num_records": ds["num_records"],
                    "valid_records": ds["valid_records"],
                    "error_records": ds["error_records"],
                    "metric_ready_records": ds["metric_ready_records"],
                    "voc_mean": m["voc_mean"],
                    "voc_std": m["voc_std"],
                    "voc_valid_count": m["voc_valid_count"],
                    "spearman_mean": m["spearman_mean"],
                    "spearman_std": m["spearman_std"],
                    "spearman_valid_count": m["spearman_valid_count"],
                    "pearson_mean": m["pearson_mean"],
                    "pearson_std": m["pearson_std"],
                    "pearson_valid_count": m["pearson_valid_count"],
                }
            )

    print("Wrote analysis files:")
    print(f"  - {per_episode_path}")
    print(f"  - {summary_path}")
    print(f"  - {csv_path}")


if __name__ == "__main__":
    main()
