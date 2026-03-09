#!/usr/bin/env python3
"""Run Robometer inference on AI2-style datasets and export JSONL records.

Example:
  uv run python scripts/eval_ai2_robometer_export.py \
    --model-path aliangdw/Robometer-4B \
    --datasets-root datasets \
    --dataset-names lerobot_annotations franka_data single_YAM bimanual_YAM_annotations \
    --use-full-video \
    --resume \
    --fsync-each-record \
    --max-input-frames 20
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import decord  # type: ignore
except Exception:
    decord = None
import numpy as np
import requests


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    subdir: str
    annotations_file: str | None


DATASET_SPECS: dict[str, DatasetSpec] = {
    "lerobot_annotations": DatasetSpec(
        dataset_name="lerobot_annotations",
        subdir="lerobot_annotations",
        annotations_file="annotations.json",
    ),
    "lerobot_failed": DatasetSpec(
        dataset_name="lerobot_failed",
        subdir="lerobot_failed",
        annotations_file=None,
    ),
    "franka_data": DatasetSpec(
        dataset_name="franka_data",
        subdir="franka_data",
        annotations_file="franka_annotations.json",
    ),
    "single_YAM": DatasetSpec(
        dataset_name="single_YAM",
        subdir="single_YAM",
        annotations_file="single_yam_annotations.json",
    ),
    "bimanual_YAM_annotations": DatasetSpec(
        dataset_name="bimanual_YAM_annotations",
        subdir="bimanual_YAM_annotations",
        annotations_file="yam_annotations.json",
    ),
}
DATASET_ALIASES: dict[str, str] = {
    "ai2_lerobot": "lerobot_annotations",
    "ai2_lerobot_failed": "lerobot_failed",
    "ai2_franka": "franka_data",
    "ai2_single_yam": "single_YAM",
    "ai2_bimanual_yam": "bimanual_YAM_annotations",
}
DEFAULT_DATASET_NAMES = [
    "lerobot_annotations",
    "lerobot_failed",
    "franka_data",
    "single_YAM",
    "bimanual_YAM_annotations",
]
DATASETS_ROOT_ENV_KEYS = [
    "DATASETS_ROOT",
    "BEAKER_DATASETS_ROOT",
    "BEAKER_DATASET_DIR",
    "BEAKER_INPUT_DIR",
]
OUTPUT_DIR_ENV_KEYS = [
    "OUTPUT_DIR",
    "BEAKER_OUTPUT_DIR",
]


def parse_args() -> argparse.Namespace:
    env_eval_server = os.environ.get("EVAL_SERVER_URL", "").strip() or None
    env_model_path = os.environ.get("MODEL_PATH", "").strip() or "aliangdw/Robometer-4B"
    parser = argparse.ArgumentParser(
        description="Export Robometer predictions for AI2-style datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inference-mode",
        choices=["auto", "local", "server"],
        default="auto",
        help="Inference backend: local model, HTTP server, or auto-select.",
    )
    parser.add_argument(
        "--model-path",
        default=env_model_path,
        help="HF/local model path for local inference mode.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for local mode (e.g., cuda:0 or cpu).",
    )
    parser.add_argument(
        "--eval-server-url",
        default=env_eval_server,
        help="Eval server base URL for server mode (e.g., http://localhost:8000).",
    )
    parser.add_argument(
        "--datasets-root",
        default=None,
        help="Root directory containing AI2 dataset folders.",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        default=list(DEFAULT_DATASET_NAMES),
        help=(
            "Dataset folder names under datasets root "
            "(lerobot_annotations lerobot_failed franka_data single_YAM bimanual_YAM_annotations); "
            "legacy ai2_* aliases also supported."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dir for *_predictions.jsonl files.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=240.0,
        help="HTTP timeout per episode request.",
    )
    parser.add_argument(
        "--request-retries",
        type=int,
        default=2,
        help="Retries for failed inference attempts (total attempts = retries + 1).",
    )
    parser.add_argument(
        "--max-input-frames",
        type=int,
        default=20,
        help="Max frames sent per episode to inference backend.",
    )
    parser.add_argument(
        "--use-full-video",
        action="store_true",
        help="If set, source completion is built over full video length.",
    )
    parser.set_defaults(wait_for_server=True)
    parser.add_argument(
        "--wait-for-server",
        dest="wait_for_server",
        action="store_true",
        help="Wait for eval server readiness before processing episodes.",
    )
    parser.add_argument(
        "--no-wait-for-server",
        dest="wait_for_server",
        action="store_false",
        help="Do not wait for eval server readiness.",
    )
    parser.add_argument(
        "--server-ready-timeout-s",
        type=float,
        default=300.0,
        help="Max time to wait for eval server health/readiness.",
    )
    parser.add_argument(
        "--server-ready-poll-s",
        type=float,
        default=2.0,
        help="Polling interval while waiting for eval server readiness.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip episode indices already present in output JSONL files.",
    )
    parser.add_argument(
        "--fsync-each-record",
        action="store_true",
        help="Flush and fsync after each JSONL write.",
    )
    return parser.parse_args()


def _resolve_dataset_names(dataset_names: list[str]) -> list[str]:
    resolved: list[str] = []
    for name in dataset_names:
        canonical = DATASET_ALIASES.get(name, name)
        if canonical not in DATASET_SPECS:
            raise ValueError(
                f"Unknown dataset name: {name}. Supported folder names: {sorted(DATASET_SPECS.keys())}. "
                f"Legacy aliases: {sorted(DATASET_ALIASES.keys())}"
            )
        resolved.append(canonical)
    return resolved


class LocalInferenceRunner:
    """Local Robometer inference helper (no external eval server needed)."""

    def __init__(self, model_path: str, device: str | None = None):
        import torch
        from robometer.data.dataset_types import ProgressSample, Trajectory
        from robometer.evals.eval_server import compute_batch_outputs
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator

        self.torch = torch
        self.ProgressSample = ProgressSample
        self.Trajectory = Trajectory
        self.compute_batch_outputs = compute_batch_outputs

        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        exp_config, tokenizer, processor, reward_model = load_model_from_hf(
            model_path=model_path,
            device=self.device,
        )
        self.exp_config = exp_config
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.reward_model.eval()
        self.batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

        loss_cfg = getattr(exp_config, "loss", None)
        progress_loss_type = str(getattr(loss_cfg, "progress_loss_type", "l2")).lower()
        self.is_discrete = progress_loss_type == "discrete"
        self.num_bins = int(
            getattr(loss_cfg, "progress_discrete_bins", None)
            or getattr(exp_config.model, "progress_discrete_bins", 10)
        )

    def infer_progress(self, frames: np.ndarray, task: str) -> tuple[list[float], list[float]]:
        sample = self.ProgressSample(
            trajectory=self.Trajectory(
                frames=frames,
                frames_shape=tuple(int(x) for x in frames.shape),
                task=task,
                id="0",
                metadata={"subsequence_length": int(frames.shape[0])},
                video_embeddings=None,
            ),
            sample_type="progress",
        )
        batch = self.batch_collator([sample])
        progress_inputs = batch["progress_inputs"]
        for key, value in progress_inputs.items():
            if isinstance(value, self.torch.Tensor):
                progress_inputs[key] = value.to(self.device)

        outputs = self.compute_batch_outputs(
            self.reward_model,
            self.tokenizer,
            progress_inputs,
            sample_type="progress",
            is_discrete_mode=self.is_discrete,
            num_bins=self.num_bins,
        )
        progress_pred = outputs.get("progress_pred", [[]])[0]
        success_probs = outputs.get("outputs_success", {}).get("success_probs", [[]])[0]
        pred_progress = [float(x) for x in progress_pred] if progress_pred else []
        pred_success = [float(x) for x in success_probs] if success_probs else []
        return pred_progress, pred_success


def _safe_float(x: Any, default: float) -> float:
    try:
        val = float(x)
        if np.isfinite(val):
            return val
    except Exception:
        pass
    return default


def _read_instruction(video_path: Path, fallback: str) -> str:
    instruction_path = video_path.with_name("instruction.txt")
    if instruction_path.exists():
        text = instruction_path.read_text(encoding="utf-8", errors="ignore")
        text = " ".join(text.split())
        if text:
            return text
    metadata_path = video_path.with_name("metadata.json")
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(metadata, dict):
                text = str(metadata.get("task") or metadata.get("instruction") or "").strip()
                if text:
                    return text
        except Exception:
            pass
    return fallback


def _derive_task_name(dataset_name: str, key: str) -> str:
    parts = PurePosixPath(key).parts
    if not parts:
        return "unknown_task"
    if dataset_name in {"lerobot_annotations", "lerobot_failed", "ai2_lerobot", "ai2_lerobot_failed"}:
        return parts[0]
    if dataset_name in {
        "franka_data",
        "single_YAM",
        "bimanual_YAM_annotations",
        "ai2_franka",
        "ai2_single_yam",
        "ai2_bimanual_yam",
    }:
        if len(parts) > 1:
            return parts[1]
    return parts[0]


def _load_dataset_entries(spec: DatasetSpec, datasets_root: Path) -> list[dict[str, Any]]:
    dataset_dir = datasets_root / spec.subdir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    entries: list[dict[str, Any]] = []
    annotations: dict[str, Any] = {}
    if spec.annotations_file:
        annotations_path = dataset_dir / spec.annotations_file
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
        with annotations_path.open("r", encoding="utf-8") as f:
            annotations = json.load(f)
        if not isinstance(annotations, dict):
            raise ValueError(f"Expected dict annotations in {annotations_path}")

    if annotations:
        for key in sorted(annotations.keys()):
            ann = annotations[key]
            if not isinstance(ann, dict):
                continue
            video_path = (dataset_dir / key).resolve()
            if not video_path.exists():
                continue
            task_text = str(ann.get("task", "")).strip()
            task_name = _derive_task_name(spec.dataset_name, key)
            instruction = _read_instruction(video_path, task_text or task_name.replace("_", " "))
            entries.append(
                {
                    "key": key,
                    "video_path": video_path,
                    "task_name": task_name,
                    "instruction": instruction,
                    "annotation": ann,
                }
            )
        return entries

    # No annotation map: enumerate mp4 files and use empty annotation dict.
    mp4_files = sorted(
        p
        for p in dataset_dir.rglob("*.mp4")
        if ".cache" not in p.parts and not any(part.startswith(".") for part in p.parts)
    )
    for video_path in mp4_files:
        rel = str(video_path.relative_to(dataset_dir).as_posix())
        task_name = _derive_task_name(spec.dataset_name, rel)
        instruction = _read_instruction(video_path, task_name.replace("_", " "))
        entries.append(
            {
                "key": rel,
                "video_path": video_path.resolve(),
                "task_name": task_name,
                "instruction": instruction,
                "annotation": {},
            }
        )
    return entries


def _read_video_metadata(video_path: Path) -> tuple[int, float]:
    if decord is not None:
        try:
            vr = decord.VideoReader(str(video_path), num_threads=1)
            total_frames = int(len(vr))
            try:
                fps = float(vr.get_avg_fps())
            except Exception:
                fps = 1.0
            del vr
            if total_frames > 0:
                return total_frames, fps if fps > 0 else 1.0
        except Exception:
            pass

    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if total_frames <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")
    if fps <= 0:
        fps = 1.0
    return total_frames, fps


def _read_frames(video_path: Path, frame_indices: list[int]) -> np.ndarray:
    if decord is not None:
        try:
            vr = decord.VideoReader(str(video_path), num_threads=1)
            frames = vr.get_batch(frame_indices).asnumpy()
            del vr
            return frames
        except Exception:
            pass

    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    out: list[np.ndarray] = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed reading frame {idx} from {video_path}")
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not out:
        return np.zeros((0, 1, 1, 3), dtype=np.uint8)
    return np.asarray(out, dtype=np.uint8)


def _completion_from_subtasks(
    subtasks: list[dict[str, Any]],
    total_frames: int,
    fps: float,
) -> list[float]:
    if total_frames <= 0:
        return []

    if not subtasks:
        if total_frames == 1:
            return [0.0]
        return [float(int(round((i / (total_frames - 1)) * 100.0))) for i in range(total_frames)]

    uses_seconds = "start_second" in subtasks[0] or "end_second" in subtasks[0]
    starts: list[float] = []
    ends: list[float] = []
    for s in subtasks:
        if uses_seconds:
            start = _safe_float(s.get("start_second"), 0.0)
            end = _safe_float(s.get("end_second"), start)
        else:
            start = _safe_float(s.get("start_frame"), 0.0)
            end = _safe_float(s.get("end_frame"), start)
        if end < start:
            start, end = end, start
        starts.append(start)
        ends.append(end)

    n = max(len(starts), 1)
    out: list[float] = []
    eps = 1e-6
    for frame_idx in range(total_frames):
        t = (frame_idx / fps) if uses_seconds else float(frame_idx)
        progress = 0.0
        assigned = False
        for i, (start, end) in enumerate(zip(starts, ends)):
            span = max(end - start, eps)
            seg_start_progress = (i / n) * 100.0
            seg_end_progress = ((i + 1) / n) * 100.0
            if t < start:
                progress = seg_start_progress
                assigned = True
                break
            if start <= t <= end:
                frac = (t - start) / span
                progress = seg_start_progress + frac * (seg_end_progress - seg_start_progress)
                assigned = True
                break
            progress = seg_end_progress
        if not assigned:
            progress = 100.0
        out.append(float(int(round(progress))))
    return out


def _last_annotated_frame(subtasks: list[dict[str, Any]], fps: float, total_frames: int) -> int:
    if not subtasks:
        return max(total_frames - 1, 0)
    if "end_second" in subtasks[0]:
        last_second = max(_safe_float(s.get("end_second"), 0.0) for s in subtasks)
        return int(max(0, min(total_frames - 1, round(last_second * fps))))
    last_frame = max(_safe_float(s.get("end_frame"), 0.0) for s in subtasks)
    return int(max(0, min(total_frames - 1, round(last_frame))))


def _choose_input_positions(total_source_frames: int, max_input_frames: int) -> list[int]:
    if total_source_frames <= 0:
        return []
    if max_input_frames <= 0 or total_source_frames <= max_input_frames:
        return list(range(total_source_frames))
    raw = np.linspace(0, total_source_frames - 1, max_input_frames, dtype=int).tolist()
    # Deduplicate while preserving order.
    dedup: list[int] = []
    seen: set[int] = set()
    for x in raw:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    return dedup


def _build_sample_payload(frames: np.ndarray, task: str) -> tuple[dict[str, Any], dict[str, Any]]:
    file_key = "sample_0_trajectory_frames"
    sample = {
        "sample_type": "progress",
        "trajectory": {
            "frames": {"__numpy_file__": file_key},
            "frames_shape": [int(x) for x in frames.shape],
            "task": task,
            "id": "0",
            "metadata": {"subsequence_length": int(frames.shape[0])},
            "video_embeddings": None,
        },
    }
    buf = io.BytesIO()
    np.save(buf, frames)
    buf.seek(0)
    files = {file_key: (f"{file_key}.npy", buf, "application/octet-stream")}
    data = {"sample_0": json.dumps(sample), "use_frame_steps": "false"}
    return files, data


def _run_server_inference(
    eval_server_url: str,
    frames: np.ndarray,
    task: str,
    timeout_s: float,
) -> tuple[list[float], list[float]]:
    files, data = _build_sample_payload(frames=frames, task=task)
    url = eval_server_url.rstrip("/") + "/evaluate_batch_npy"
    try:
        resp = requests.post(url, files=files, data=data, timeout=timeout_s)
        resp.raise_for_status()
        outputs = resp.json()
    finally:
        for _, file_tuple in files.items():
            try:
                file_tuple[1].close()
            except Exception:
                pass

    progress_pred = (
        outputs.get("outputs_progress", {})
        .get("progress_pred", [[]])[0]
    )
    success_probs = (
        outputs.get("outputs_success", {})
        .get("success_probs", [[]])[0]
    )
    pred_progress = [float(x) for x in progress_pred] if progress_pred else []
    pred_success = [float(x) for x in success_probs] if success_probs else []
    return pred_progress, pred_success


def _get_existing_episode_indices(jsonl_path: Path) -> set[int]:
    done: set[int] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            try:
                done.add(int(rec.get("episode_index")))
            except Exception:
                continue
    return done


def _write_record(f, record: dict[str, Any], fsync_each_record: bool) -> None:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if fsync_each_record:
        f.flush()
        os.fsync(f.fileno())


def _resolve_datasets_root(datasets_root_arg: str | None, dataset_names: list[str]) -> Path:
    if datasets_root_arg:
        p = Path(datasets_root_arg).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Datasets root not found: {p}")

    env_candidates = [os.environ.get(k, "").strip() for k in DATASETS_ROOT_ENV_KEYS]
    default_candidates = [
        "datasets",
        "dataset",
        "/data",
        "/datasets",
        "/mnt/data",
        "/mnt/datasets",
        "/input",
    ]
    candidates = [c for c in env_candidates + default_candidates if c]
    for cand in candidates:
        root = Path(cand).expanduser().resolve()
        if not root.exists():
            continue
        has_any = False
        for name in dataset_names:
            subdir = DATASET_SPECS[name].subdir
            if (root / subdir).exists():
                has_any = True
                break
        if has_any:
            return root
    raise FileNotFoundError(
        "Could not resolve datasets root. Set --datasets-root or one of "
        + ", ".join(DATASETS_ROOT_ENV_KEYS)
    )


def _resolve_output_dir(output_dir_arg: str | None, use_full_video: bool) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg).expanduser().resolve()

    for key in OUTPUT_DIR_ENV_KEYS:
        val = os.environ.get(key, "").strip()
        if val:
            return Path(val).expanduser().resolve()

    if Path("/output").exists():
        return (
            Path("/output")
            / (
                "robometer_ai2_server_export_"
                + ("full_video" if use_full_video else "trimmed_video")
                + "_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
            )
        ).resolve()

    return (
        Path(
            "baseline_eval_output/"
            + "robometer_ai2_server_export_"
            + ("full_video" if use_full_video else "trimmed_video")
            + "_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    ).resolve()


def _wait_for_eval_server(eval_server_url: str, timeout_s: float, poll_s: float) -> None:
    deadline = time.time() + max(1.0, float(timeout_s))
    health_urls = [
        eval_server_url.rstrip("/") + "/health",
        eval_server_url.rstrip("/") + "/docs",
    ]
    while time.time() < deadline:
        for url in health_urls:
            try:
                resp = requests.get(url, timeout=5.0)
                if resp.status_code < 500:
                    return
            except Exception:
                pass
        time.sleep(max(0.2, float(poll_s)))
    raise TimeoutError(f"Eval server did not become ready within {timeout_s}s: {eval_server_url}")


def _resolve_inference_mode(args: argparse.Namespace) -> str:
    mode = str(args.inference_mode).lower()
    if mode == "auto":
        if args.model_path:
            return "local"
        if args.eval_server_url:
            return "server"
        raise ValueError("Auto mode could not resolve backend. Set --model-path or --eval-server-url.")
    return mode


def main() -> None:
    args = parse_args()
    inference_mode = _resolve_inference_mode(args)
    if inference_mode == "local" and not args.model_path:
        raise ValueError("Local mode requires --model-path (or MODEL_PATH env var).")
    if inference_mode == "server" and not args.eval_server_url:
        raise ValueError("Server mode requires --eval-server-url (or EVAL_SERVER_URL env var).")

    resolved_dataset_names = _resolve_dataset_names(args.dataset_names)
    datasets_root = _resolve_datasets_root(args.datasets_root, resolved_dataset_names)

    output_dir = _resolve_output_dir(args.output_dir, args.use_full_video)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Datasets root: {datasets_root}")
    print(f"Output dir: {output_dir}")
    print(f"Inference mode: {inference_mode}")
    if inference_mode == "local":
        print(f"Model path: {args.model_path}")
    else:
        print(f"Eval server: {args.eval_server_url}")
    if inference_mode == "server" and args.wait_for_server:
        print("Waiting for eval server readiness...")
        _wait_for_eval_server(
            eval_server_url=args.eval_server_url,
            timeout_s=float(args.server_ready_timeout_s),
            poll_s=float(args.server_ready_poll_s),
        )
        print("Eval server is ready.")

    local_runner: LocalInferenceRunner | None = None
    if inference_mode == "local":
        print("Loading local model...")
        local_runner = LocalInferenceRunner(model_path=args.model_path, device=args.device)
        print(f"Local model loaded on device: {local_runner.device}")

    for dataset_name in resolved_dataset_names:
        spec = DATASET_SPECS[dataset_name]
        entries = _load_dataset_entries(spec=spec, datasets_root=datasets_root)
        output_path = output_dir / f"{dataset_name}_predictions.jsonl"
        done_indices = _get_existing_episode_indices(output_path) if args.resume else set()
        output_file_mode = "a" if args.resume else "w"
        print(f"\n[{dataset_name}] episodes={len(entries)} output={output_path}")
        if done_indices:
            print(f"[{dataset_name}] resume: skipping {len(done_indices)} existing episode indices")

        with output_path.open(output_file_mode, encoding="utf-8") as f:
            for episode_index, entry in enumerate(entries):
                if episode_index in done_indices:
                    continue

                record: dict[str, Any] = {
                    "dataset": dataset_name,
                    "episode_index": int(episode_index),
                    "prediction_time": datetime.now().isoformat(),
                }
                try:
                    video_path: Path = entry["video_path"]
                    annotation: dict[str, Any] = entry["annotation"]
                    task_name: str = entry["task_name"]
                    instruction: str = entry["instruction"]

                    total_frames, source_fps = _read_video_metadata(video_path)

                    subtasks = annotation.get("subtasks", []) or []
                    if not isinstance(subtasks, list):
                        subtasks = []
                    completion_all = _completion_from_subtasks(
                        subtasks=subtasks,
                        total_frames=total_frames,
                        fps=source_fps,
                    )

                    if args.use_full_video:
                        source_frame_indices = list(range(total_frames))
                    else:
                        last_annotated = _last_annotated_frame(
                            subtasks=subtasks,
                            fps=source_fps,
                            total_frames=total_frames,
                        )
                        source_frame_indices = list(range(last_annotated + 1))

                    source_completion = [float(completion_all[i]) for i in source_frame_indices]

                    keep_positions = _choose_input_positions(
                        total_source_frames=len(source_frame_indices),
                        max_input_frames=int(args.max_input_frames),
                    )
                    input_frame_indices = [int(source_frame_indices[p]) for p in keep_positions]
                    true_completion = [float(source_completion[p]) for p in keep_positions]
                    if input_frame_indices:
                        frames = _read_frames(video_path, input_frame_indices)
                    else:
                        frames = np.zeros((0, 1, 1, 3), dtype=np.uint8)

                    if frames.size == 0:
                        raise RuntimeError("No frames selected for inference")

                    last_exc: Exception | None = None
                    pred_progress: list[float] = []
                    pred_success_probs: list[float] = []
                    for attempt in range(int(args.request_retries) + 1):
                        try:
                            if inference_mode == "local":
                                assert local_runner is not None
                                pred_progress, pred_success_probs = local_runner.infer_progress(
                                    frames=frames,
                                    task=instruction,
                                )
                            else:
                                pred_progress, pred_success_probs = _run_server_inference(
                                    eval_server_url=args.eval_server_url,
                                    frames=frames,
                                    task=instruction,
                                    timeout_s=float(args.timeout_s),
                                )
                            last_exc = None
                            break
                        except Exception as e:
                            last_exc = e
                            if attempt < int(args.request_retries):
                                time.sleep(1.0 + attempt)
                    if last_exc is not None:
                        raise last_exc

                    record.update(
                        {
                            "task_name": task_name,
                            "instruction": instruction,
                            "video_path": str(video_path),
                            "source_total_frames": total_frames,
                            "source_fps": source_fps,
                            "source_num_frames_loaded": len(source_frame_indices),
                            "source_frame_indices": source_frame_indices,
                            "source_completion": source_completion,
                            "input_frame_indices": input_frame_indices,
                            "true_completion": true_completion,
                            "input_num_frames": len(input_frame_indices),
                            "inference_keep_positions": input_frame_indices,
                            "pred_progress": pred_progress,
                            "pred_success_probs": pred_success_probs,
                            "pred_num_frames": len(pred_progress),
                            "pred_final_progress": float(pred_progress[-1]) if pred_progress else None,
                            "pred_final_success_prob": float(pred_success_probs[-1]) if pred_success_probs else None,
                            "error": None,
                        }
                    )
                except Exception as e:
                    record.update({"error": str(e)})

                _write_record(f, record, fsync_each_record=bool(args.fsync_each_record))
                print(
                    f"[{dataset_name}] episode={episode_index} "
                    f"status={'ok' if record.get('error') is None else 'error'}"
                )

    print("\nDone.")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
