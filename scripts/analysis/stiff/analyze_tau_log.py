#!/usr/bin/env python3
"""Analyze torque log snapshots and plot selected joint torque traces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# Canonical joint order used by the locomotion task (0-based indices).
JOINT_NAME_ORDER: List[str] = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]
JOINT_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(JOINT_NAME_ORDER)}

DEFAULT_LOG_PATH = (
    Path(__file__).resolve().parents[2]
    / "source"
    / "unitree_rl_lab"
    / "unitree_rl_lab"
    / "tasks"
    / "locomotion"
    / "mdp"
    / "tau0.log"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a torque log file and plot moving-average torque for the requested joint IDs."
        )
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to the torque log produced by actions_multi.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output image path. When omitted, a PNG is written next to the log "
            "with the suffix '_stats'."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving the figure.",
    )
    parser.add_argument(
        "--ma-window",
        type=int,
        default=50,
        help="Moving-average window size in snapshots (use 1 to plot the raw torque).",
    )
    parser.add_argument(
        "--joint-id",
        type=int,
        action="append",
        default=None,
        help=(
            "Optional joint indices to track (0-based). Repeat to specify multiple joints."
        ),
    )
    parser.add_argument(
        "--joint-name",
        type=str,
        action="append",
        default=None,
        help="Optional joint names to track (resolved using the locomotion joint order).",
    )
    return parser.parse_args()


def _parse_vector_snapshot(text: str) -> np.ndarray:
    cleaned = text.replace("[", " ").replace("]", " ")
    values = np.fromstring(cleaned, sep=" ")
    if values.size == 0:
        raise ValueError("Empty torque snapshot encountered.")
    return values


def _iter_entries(lines: Iterable[str]) -> Iterable[Optional[np.ndarray]]:
    buffer: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "":
            if buffer:
                text = "".join(buffer)
                yield _parse_vector_snapshot(text)
                buffer.clear()
            continue
        if stripped == "None":
            if buffer:
                raise ValueError("Encountered 'None' marker while still buffering an entry.")
            yield None
            continue
        buffer.append(line)

    if buffer:
        text = "".join(buffer)
        yield _parse_vector_snapshot(text)


def load_snapshots(log_path: Path) -> List[Optional[np.ndarray]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    with log_path.open("r", encoding="utf-8") as file:
        entries = list(_iter_entries(file))

    if not entries:
        raise ValueError(f"No entries found in log file: {log_path}")

    return entries


def extract_joint_series(
    entries: Iterable[Optional[np.ndarray]], joint_ids: Sequence[int]
) -> dict[int, np.ndarray]:
    if not joint_ids:
        return {}

    joint_ids = list(dict.fromkeys(int(j) for j in joint_ids))
    joint_series: dict[int, List[float]] = {jid: [] for jid in joint_ids}
    expected_size: Optional[int] = None

    for entry in entries:
        if entry is None:
            for values in joint_series.values():
                values.append(np.nan)
            continue

        if entry.ndim != 1:
            entry = np.asarray(entry).reshape(-1)

        if expected_size is None:
            expected_size = entry.shape[0]
            for jid in joint_ids:
                if jid < 0:
                    raise ValueError(f"Joint ID must be non-negative, got {jid}")
                if jid >= expected_size:
                    raise ValueError(
                        f"Joint ID {jid} is out of range for torque vectors of length {expected_size}."
                    )
        elif entry.shape[0] != expected_size:
            raise ValueError(
                "Encountered torque vectors with differing lengths; ensure the log comes from a single configuration."
            )

        for jid in joint_ids:
            joint_series[jid].append(float(entry[jid]))

    return {jid: np.asarray(series, dtype=float) for jid, series in joint_series.items()}


def build_output_path(log_path: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    return log_path.with_name(log_path.stem + "_stats.png")


def moving_average(series: np.ndarray, window: int) -> np.ndarray:
    data = np.asarray(series, dtype=float)
    if window <= 1:
        return data.copy()

    weights = np.ones(window, dtype=float)
    valid_mask = np.isfinite(data).astype(float)
    data_filled = np.nan_to_num(data, nan=0.0, copy=True)

    numerators = np.convolve(data_filled, weights, mode="valid")
    denominators = np.convolve(valid_mask, weights, mode="valid")
    denominators = np.maximum(denominators, 1.0)

    ma_valid = numerators / denominators
    result = np.full_like(data, np.nan, dtype=float)
    result[window - 1 :] = ma_valid
    return result


def plot_joint_series(
    joint_ma_series: dict[int, np.ndarray],
    output_path: Path,
    show: bool,
    window: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it via 'pip install matplotlib' and retry."
        ) from exc
    if not joint_ma_series:
        raise ValueError("No joint series provided to plot.")

    length = max(series.shape[0] for series in joint_ma_series.values())
    indices = np.arange(length)

    plt.figure(figsize=(10, 5))
    for jid, series in joint_ma_series.items():
        plt.plot(indices, series, label=f"joint {jid} (MA window={window})")
    plt.xlabel("Snapshot index")
    plt.ylabel("Torque (Nm)")
    plt.yscale("symlog", linthresh=1e-3)
    # plt.yscale("linear")
    plt.title("Torque statistics over time (log scale)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)

    if show:
        plt.show()


def main() -> None:
    args = parse_args()
    try:
        snapshots = load_snapshots(args.log_path)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    if args.ma_window < 1:
        raise ValueError("--ma-window must be a positive integer")

    joint_ids: List[int] = [] if args.joint_id is None else list(args.joint_id)
    if args.joint_name:
        for name in args.joint_name:
            if name not in JOINT_NAME_TO_ID:
                available = ", ".join(JOINT_NAME_TO_ID.keys())
                raise ValueError(
                    f"Unknown joint name '{name}'. Available names: {available}"
                )
            joint_ids.append(JOINT_NAME_TO_ID[name])

    if not joint_ids:
        raise ValueError("Specify at least one --joint-id to plot.")

    joint_series = extract_joint_series(snapshots, joint_ids)
    joint_ma_series = {jid: moving_average(series, args.ma_window) for jid, series in joint_series.items()}
    output_path = build_output_path(args.log_path, args.output)
    plot_joint_series(joint_ma_series, output_path, args.show, args.ma_window)

    print(f"Processed {len(snapshots)} entries.")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
