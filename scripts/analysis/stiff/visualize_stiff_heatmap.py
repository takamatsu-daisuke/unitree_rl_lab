#!/usr/bin/env python3
"""Render stiffness matrix snapshots as heatmaps or a simple animation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _ensure_repo_on_path() -> None:
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    candidates = [repo_root.parent, repo_root, repo_root / "source"]
    for path in candidates:
        path_str = str(path)
        if path_str not in sys.path and path.exists():
            sys.path.insert(0, path_str)


_ensure_repo_on_path()

from unitree_rl_lab.scripts.analysis.stiff.analyze_stiff_log import load_snapshots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize stiffness matrix snapshots as heatmaps (grid or animation)."
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Path to the stiffness log. Defaults to the analyzer's built-in location.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for saved images/animation. Defaults next to the log file.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Explicit snapshot indices to visualize (0-based).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting snapshot index (inclusive) when indices not provided.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending snapshot index (exclusive) when indices not provided.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride between snapshots when indices not provided.",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=16,
        help="Maximum number of snapshots to render in grid mode.",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Save an animation instead of a static grid."
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Animation frames-per-second when --animate is used.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="coolwarm",
        help="Matplotlib colormap name to use for heatmaps.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Lower limit for color scale. Defaults to data minimum.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Upper limit for color scale. Defaults to data maximum.",
    )
    parser.add_argument(
        "--abs-max",
        type=float,
        default=None,
        help="Use ±abs-max for symmetric color limits (overrides vmin/vmax).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure window after saving.",
    )
    return parser.parse_args()


def _resolve_log_path(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    repo_root = Path(__file__).resolve().parents[3]
    candidate = (
        repo_root
        / "source"
        / "unitree_rl_lab"
        / "unitree_rl_lab"
        / "tasks"
        / "locomotion"
        / "mdp"
        / "stiff_np_list0.log"
    )
    if not candidate.exists():
        raise FileNotFoundError("Could not locate default stiffness log; specify --log-path.")
    return candidate


def _pick_indices(
    total: int,
    indices: Optional[Sequence[int]],
    start: int,
    end: Optional[int],
    stride: int,
    max_count: Optional[int],
) -> List[int]:
    if indices:
        selection = sorted(set(i for i in indices if 0 <= i < total))
        if not selection:
            raise ValueError("No valid snapshot indices within range.")
        return selection

    stop = total if end is None else min(end, total)
    if start < 0 or start >= total:
        raise ValueError(f"--start ({start}) must be in [0, {total - 1}].")
    sequence = list(range(start, stop, max(stride, 1)))
    if not sequence:
        raise ValueError("No snapshot indices selected by range parameters.")
    if max_count is not None and len(sequence) > max_count:
        step = math.ceil(len(sequence) / max_count)
        sequence = sequence[::step][:max_count]
    return sequence


def _gather_arrays(entries: Iterable[Optional[np.ndarray]], indices: Sequence[int]) -> List[np.ndarray]:
    snapshot_map = {}
    for idx, entry in enumerate(entries):
        if entry is not None and idx in indices:
            snapshot_map[idx] = entry
    missing = [i for i in indices if i not in snapshot_map]
    if missing:
        raise ValueError(f"Requested snapshots contain None entries: {missing}")
    return [snapshot_map[i] for i in indices]


def _resolve_output_dir(base: Path, requested: Optional[Path]) -> Path:
    if requested is not None:
        return requested
    return base.with_suffix("") / "stiff_heatmaps"


def _determine_limits(arrays: Sequence[np.ndarray], vmin: Optional[float], vmax: Optional[float], abs_max: Optional[float]) -> tuple[float, float]:
    if abs_max is not None:
        limit = float(abs(abs_max))
        if limit == 0:
            limit = 1.0
        return -limit, limit
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)
    data_min = min(float(arr.min()) for arr in arrays)
    data_max = max(float(arr.max()) for arr in arrays)
    if vmin is None:
        vmin = data_min
    if vmax is None:
        vmax = data_max
    return float(vmin), float(vmax)


def _plot_grid(arrays: Sequence[np.ndarray], indices: Sequence[int], output_dir: Path, cmap: str, vmin: float, vmax: float, show: bool) -> Path:
    count = len(arrays)
    cols = min(4, count)
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for ax, arr, idx in zip(axes.flat, arrays, indices):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"snapshot {idx}", fontsize=9)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "stiff_heatmap_grid.png"
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _make_animation(arrays: Sequence[np.ndarray], indices: Sequence[int], output_dir: Path, cmap: str, vmin: float, vmax: float, fps: float, show: bool) -> Path:
    from matplotlib import animation

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(arrays[0], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    text = ax.text(0.02, 0.98, f"snapshot {indices[0]}", color="white", ha="left", va="top", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax)

    def update(frame: int) -> list:
        im.set_data(arrays[frame])
        text.set_text(f"snapshot {indices[frame]}")
        return [im, text]

    anim = animation.FuncAnimation(fig, update, frames=len(arrays), interval=1000.0 / max(fps, 1.0), blit=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "stiff_heatmap_animation.mp4"
    try:
        anim.save(out_path, fps=fps, dpi=150, writer="ffmpeg")
    except Exception:
        fallback = out_path.with_suffix(".gif")
        anim.save(fallback, fps=fps, dpi=150, writer="pillow")
        out_path = fallback

    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()

    log_path = _resolve_log_path(args.log_path)
    snapshots = load_snapshots(log_path)
    valid_total = len(snapshots)
    indices = _pick_indices(valid_total, args.indices, args.start, args.end, args.stride, None if args.animate else args.max_count)
    arrays = _gather_arrays(snapshots, indices)

    output_dir = _resolve_output_dir(log_path, args.output_dir)
    vmin, vmax = _determine_limits(arrays, args.vmin, args.vmax, args.abs_max)

    if args.animate:
        result = _make_animation(arrays, indices, output_dir, args.cmap, vmin, vmax, args.fps, args.show)
    else:
        result = _plot_grid(arrays, indices, output_dir, args.cmap, vmin, vmax, args.show)

    print(f"Saved visualization to {result}")


if __name__ == "__main__":
    main()
