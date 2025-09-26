#!/usr/bin/env python3
"""Generate an animation (GIF or MP4) from `stiff_np_list0.log`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm, Normalize, SymLogNorm


def iter_frames(log_path: Path) -> Iterator[np.ndarray]:
    """Yield matrices parsed from the log file one block at a time."""
    buffer: List[str] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                if buffer:
                    yield _block_to_array(buffer)
                    buffer.clear()
                continue
            buffer.append(stripped)
        if buffer:
            yield _block_to_array(buffer)


def _block_to_array(lines: List[str]) -> np.ndarray:
    """Convert a list of textual rows into a float matrix."""
    if not lines:
        raise ValueError("Encountered empty block while parsing log file")

    row_count = sum(1 for line in lines if line.lstrip().startswith("["))
    if row_count == 0:
        raise ValueError("Unable to determine row count for block")

    numeric_str = " ".join(line.replace("[", " ").replace("]", " ") for line in lines)
    values = np.fromstring(numeric_str, sep=" ", dtype=np.float32)
    if values.size == 0:
        raise ValueError("No numeric data found in block")
    if values.size % row_count != 0:
        raise ValueError(
            f"Inconsistent block: total entries {values.size} not divisible by row count {row_count}"
        )
    col_count = values.size // row_count
    return values.reshape((row_count, col_count))


def load_frames(
    log_path: Path, step: int, max_frames: int | None,
) -> tuple[List[np.ndarray], List[int]]:
    """Load and optionally down-sample frames from the log file."""
    frames: List[np.ndarray] = []
    source_indices: List[int] = []
    for idx, frame in enumerate(iter_frames(log_path)):
        if idx % step != 0:
            continue
        frames.append(frame)
        source_indices.append(idx)
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames, source_indices


def load_state_labels(state_path: Path) -> List[str]:
    """Read per-frame state labels from a plain text file."""
    states: List[str] = []
    with state_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            label = raw_line.strip()
            if not label:
                continue
            states.append(label)
    if not states:
        raise ValueError(f"State log {state_path} did not contain any labels")
    return states


def _compute_z_axis_limits(
    data_min: float,
    data_max: float,
    zscale: str,
    z_limit_exp: float | None,
) -> tuple[float, float]:
    if z_limit_exp is None:
        return float(data_min), float(data_max)

    limit = float(10.0 ** z_limit_exp)
    if zscale == "log":
        reciprocal = float(10.0 ** (-z_limit_exp))
        lower = min(limit, reciprocal)
        upper = max(limit, reciprocal)
        eps = np.finfo(float).tiny
        lower = max(lower, eps)
        if upper <= lower:
            upper = lower * 10.0
        return lower, upper

    if zscale == "linear" and data_min >= 0:
        return 0.0, limit

    return -limit, limit


def create_animation(
    frames: List[np.ndarray],
    output_path: Path,
    fps: int,
    dpi: int,
    elev: float,
    azim: float,
    cmap_name: str,
    zscale: str,
    symlog_linthresh: float,
    z_limit_exp: float | None,
    output_format: str,
    state_labels: Sequence[str] | None,
) -> None:
    if not frames:
        raise ValueError("No frames were loaded. Adjust --step/--max-frames values.")

    first = frames[0]
    n_rows, n_cols = first.shape
    x_indices = np.arange(n_cols)
    y_indices = np.arange(n_rows)
    x_grid, y_grid = np.meshgrid(x_indices, y_indices)
    xpos = x_grid.ravel()
    ypos = y_grid.ravel()
    zpos = np.zeros_like(xpos)
    dx = np.full_like(xpos, 0.8, dtype=float)
    dy = np.full_like(ypos, 0.8, dtype=float)

    data_min = min(float(frame.min()) for frame in frames)
    data_max = max(float(frame.max()) for frame in frames)
    axis_min, axis_max = _compute_z_axis_limits(data_min, data_max, zscale, z_limit_exp)
    if np.isclose(axis_min, axis_max):
        axis_max = axis_min + 1.0

    if zscale == "log":
        if data_min <= 0:
            raise ValueError(
                "Log scaling for the z-axis requires strictly positive values in the input data"
            )
        norm = LogNorm(vmin=axis_min, vmax=axis_max, clip=True)
    elif zscale == "symlog":
        if symlog_linthresh <= 0:
            raise ValueError("--symlog-linthresh must be > 0 when --zscale symlog is selected")
        norm = SymLogNorm(
            linthresh=symlog_linthresh,
            vmin=axis_min,
            vmax=axis_max,
            clip=True,
        )
    else:
        norm = Normalize(vmin=axis_min, vmax=axis_max, clip=True)
    cmap = plt.get_cmap(cmap_name)

    fig = plt.figure(figsize=(8, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    if zscale == "log":
        ax.set_zscale("log")
    elif zscale == "symlog":
        ax.set_zscale("symlog", linthresh=symlog_linthresh)

    def draw_frame(frame_index: int) -> None:
        ax.cla()
        data = frames[frame_index]
        dz = data.ravel()
        colors = cmap(norm(dz))
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)
        ax.view_init(elev=elev, azim=azim)
        if zscale == "log":
            ax.set_zscale("log")
        elif zscale == "symlog":
            ax.set_zscale("symlog", linthresh=symlog_linthresh)
        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        ax.set_zlim(axis_min, axis_max)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_zlabel("Value")
        ax.set_title(f"Frame {frame_index + 1}/{len(frames)}")
        if state_labels is not None:
            label = state_labels[frame_index]
            ax.text2D(
                0.97,
                0.93,
                f"設置状態: {label}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 3.0},
            )

    anim = FuncAnimation(fig, draw_frame, frames=len(frames), interval=1000 / fps, repeat=False)
    if output_format == "gif":
        writer = PillowWriter(fps=fps)
    elif output_format == "mp4":
        try:
            writer = FFMpegWriter(fps=fps)
        except (FileNotFoundError, RuntimeError) as exc:
            raise RuntimeError(
                "FFmpeg is required to export MP4 files. Install FFmpeg and retry, or "
                "choose --format gif."
            ) from exc
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=writer)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_input = Path(
        "unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/stiff_np_list0.log"
    )
    default_output = Path("unitree_rl_lab/script/analysis/stiff_np_list0.mp4")
    default_state_log = Path(
        "unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/stand0.log"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Create an animation (GIF or MP4) that shows the time evolution of a 3D bar chart "
            "based on stiff_np_list0.log"
        )
    )
    parser.add_argument("--input", type=Path, default=default_input, help="Path to the log file")
    parser.add_argument(
        "--output", type=Path, default=default_output, help="Target animation path"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Use every Nth frame (>=1) to control the animation length",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10000,
        help="Maximum number of frames to load (omit or set <=0 for all frames)",
    )
    parser.add_argument("--fps", type=int, default=4, help="Animation speed in frames per second")
    parser.add_argument("--dpi", type=int, default=100, help="Figure resolution in DPI")
    parser.add_argument("--elev", type=float, default=35.0, help="Elevation angle for the 3D view")
    parser.add_argument("--azim", type=float, default=45.0, help="Azimuth angle for the 3D view")
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name for bar coloring",
    )
    parser.add_argument(
        "--zscale",
        choices=("linear", "log", "symlog"),
        default="symlog",
        help="Scaling for the z-axis",
    )
    parser.add_argument(
        "--symlog-linthresh",
        type=float,
        default=1.0,
        help="Linear range half-width when using symlog scaling",
    )
    parser.add_argument(
        "--state-log",
        type=Path,
        default=default_state_log,
        help="Optional path to per-frame state labels (one entry per line)",
    )
    parser.add_argument(
        "--no-state-log",
        action="store_true",
        help="Disable state overlay even when a state log path is provided",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("gif", "mp4"),
        default="mp4",
        help="Animation container to export",
    )
    parser.add_argument(
        "-e",
        "--z-limit-exp",
        type=float,
        default=5.0,
        help=(
            "Exponent e used to clamp the z-axis. For linear/symlog scaling the range becomes "
            "[-10**e, 10**e]; for log scaling it becomes [10**(-e), 10**e]."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.step < 1:
        print("--step must be >= 1", file=sys.stderr)
        sys.exit(1)
    max_frames = None if args.max_frames is None or args.max_frames <= 0 else args.max_frames

    frames, frame_indices = load_frames(args.input, args.step, max_frames)
    print(f"Loaded {len(frames)} frame(s) using step={args.step} from {args.input}")
    state_labels: Sequence[str] | None = None
    state_log_path = None if args.no_state_log else args.state_log
    if state_log_path is not None:
        all_states = load_state_labels(state_log_path)
        if frame_indices:
            max_index = frame_indices[-1]
            if max_index >= len(all_states):
                raise ValueError(
                    f"State log {state_log_path} has only {len(all_states)} entries, "
                    f"but frame index {max_index} was requested."
                )
            state_labels = [all_states[idx] for idx in frame_indices]
        else:
            state_labels = []
    output_path = args.output
    expected_suffix = f".{args.format}"
    if output_path.suffix.lower() != expected_suffix:
        print(
            f"[INFO] Output suffix {output_path.suffix or '<none>'} does not match --format {args.format}; "
            f"saving as {output_path.with_suffix(expected_suffix)}",
            file=sys.stderr,
        )
        output_path = output_path.with_suffix(expected_suffix)
    create_animation(
        frames,
        output_path,
        args.fps,
        args.dpi,
        args.elev,
        args.azim,
        args.cmap,
        args.zscale,
        args.symlog_linthresh,
        args.z_limit_exp,
        args.format,
        state_labels,
    )
    print(f"Saved animation to {output_path}")


if __name__ == "__main__":
    main()
