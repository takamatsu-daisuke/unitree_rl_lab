#!/usr/bin/env python3
"""Plot tau0.log joint torques for a specific timestep as a bar chart."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unitree_rl_lab.scripts.analysis.stiff.analyze_stiff_log import JOINT_NAME_ORDER

DEFAULT_LOG_PATH = (
    Path(__file__).resolve().parents[3]
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
            "Display a bar chart of joint torques from tau0.log for a selected timestep."
        )
    )
    parser.add_argument(
        "timestep",
        type=int,
        help="0-based timestep to visualize (negative values index from the end).",
    )
    parser.add_argument(
        "scale",
        choices=("linear", "log"),
        help="Set the y-axis scale. 'log' uses a symmetrical log scale for signed torques.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"Path to tau0.log (default: {DEFAULT_LOG_PATH}).",
    )
    return parser.parse_args()


def _iter_vectors(lines: Iterable[str]) -> Iterable[np.ndarray]:
    buffer: List[str] = []
    bracket_balance = 0
    for raw_line in lines:
        if not buffer and raw_line.strip() == "":
            continue

        buffer.append(raw_line)
        bracket_balance += raw_line.count("[") - raw_line.count("]")
        if bracket_balance != 0:
            continue

        text = "".join(buffer).strip()
        buffer.clear()
        if not text:
            continue

        values = np.fromstring(text.replace("[", " ").replace("]", " "), sep=" ")
        if values.size == 0:
            raise ValueError("Encountered an empty torque entry in the log file.")
        yield values

    if buffer:
        raise ValueError("Trailing partial entry detected; log file may be truncated.")


def load_tau_vectors(log_path: Path) -> np.ndarray:
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    with log_path.open("r", encoding="utf-8") as handle:
        vectors = list(_iter_vectors(handle))

    if not vectors:
        raise ValueError(f"No torque entries found in log file: {log_path}")

    expected = len(JOINT_NAME_ORDER)
    for index, vec in enumerate(vectors):
        if vec.size != expected:
            raise ValueError(
                f"Entry {index} has {vec.size} elements, expected {expected}. Check the log formatting."
            )

    return np.vstack(vectors)


def resolve_timestep(data: np.ndarray, timestep: int) -> tuple[int, np.ndarray]:
    total = data.shape[0]
    idx = timestep if timestep >= 0 else total + timestep
    if idx < 0 or idx >= total:
        raise IndexError(
            f"Requested timestep {timestep} is out of range for {total} available entries."
        )
    return idx, data[idx]


def plot_torques(timestep: int, torques: np.ndarray, scale: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it via 'pip install matplotlib'."
        ) from exc

    joint_indices = np.arange(len(torques))

    plt.figure(figsize=(14, 6))
    plt.bar(joint_indices, torques, color="steelblue")
    plt.xticks(joint_indices, JOINT_NAME_ORDER, rotation=75, ha="right", fontsize=9)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Joint")
    plt.ylabel("Torque")
    plt.title(f"tau0.log torques at timestep {timestep}")

    if scale == "log":
        plt.yscale("symlog", linthresh=1e-3)
        plt.ylabel("Torque (symlog scale)")

    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    data = load_tau_vectors(args.log_path)
    idx, torques = resolve_timestep(data, args.timestep)
    plot_torques(idx, torques, args.scale)


if __name__ == "__main__":
    main()
