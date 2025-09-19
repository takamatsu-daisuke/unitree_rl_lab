"""Spawn-safe variant of the RSL-RL training entry point."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Sequence

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


def _ensure_namespace_packages() -> list[str]:
    """Ensure namespace packages used by isaaclab are discoverable."""

    potential_paths: list[Path] = []

    # Project source tree (for editable installs or direct source usage).
    project_source = Path(__file__).resolve().parents[2] / "source"
    if project_source.exists():
        potential_paths.append(project_source)

    # Isaac Lab source tree (contains isaaclab_tasks namespace package).
    try:
        import isaaclab  # noqa: F401
    except ModuleNotFoundError:
        isaaclab_source: Path | None = None
    else:
        isaaclab_source = Path(sys.modules["isaaclab"].__file__).resolve().parent / "source"  # type: ignore[attr-defined]
        if isaaclab_source.exists():
            potential_paths.append(isaaclab_source)

    # Deduplicate while preserving order.
    unique_paths: list[str] = []
    for path in potential_paths:
        path_str = str(path)
        if path_str not in unique_paths:
            unique_paths.append(path_str)

    # Prepend to sys.path so imports succeed in the parent process.
    for path_str in reversed(unique_paths):
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    # Propagate to child processes spawned via multiprocessing.
    if unique_paths:
        existing = os.environ.get("PYTHONPATH", "")
        entries = [entry for entry in existing.split(os.pathsep) if entry]
        for path_str in unique_paths:
            if path_str not in entries:
                entries.insert(0, path_str)
        os.environ["PYTHONPATH"] = os.pathsep.join(entries)

    return unique_paths


_ensure_namespace_packages()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL (spawn-safe entrypoint).")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
    parser.add_argument(
        "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
    )
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _parse_cli(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = _build_arg_parser()
    args_cli, hydra_args = parser.parse_known_args(argv)
    return args_cli, hydra_args


def _build_hydra_main(args_cli: argparse.Namespace, app_launcher: AppLauncher) -> Callable[[], None]:
    """Construct the Hydra-decorated main function after CLI parsing."""
    import importlib.metadata as metadata
    import inspect
    import os
    import platform
    import shutil
    from datetime import datetime

    import gymnasium as gym
    import torch
    from packaging import version
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_pickle, dump_yaml
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import unitree_rl_lab.tasks  # noqa: F401
    from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

    # Check minimum supported RSL-RL version when running distributed
    RSL_RL_VERSION = "2.3.1"
    installed_version = metadata.version("rsl-rl-lib")
    if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
        if platform.system() == "Windows":
            cmd = [r".\\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
        else:
            cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
        print(
            f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
            f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
            f"\n\n\t{' '.join(cmd)}\n"
        )
        sys.exit(1)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def _main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
        """Train with RSL-RL agent (spawn-safe entrypoint)."""
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg.max_iterations = (
            args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
        )

        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        if args_cli.distributed:
            env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
            agent_cfg.device = f"cuda:{app_launcher.local_rank}"

            seed = agent_cfg.seed + app_launcher.local_rank
            env_cfg.seed = seed
            agent_cfg.seed = seed

        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Exact experiment name requested from command line: {log_dir}")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        runner.add_git_repo_to_log(__file__)
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)

        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
        export_deploy_cfg(env.unwrapped, log_dir)
        shutil.copy(
            inspect.getfile(env_cfg.__class__),
            os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
        )

        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        env.close()

    return _main


def _run() -> None:
    args_cli, hydra_args = _parse_cli()

    if args_cli.video:
        args_cli.enable_cameras = True

    _ensure_namespace_packages()

    sys.argv = [sys.argv[0]] + hydra_args

    original_sys_path = list(sys.path)

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    added_paths = [path for path in sys.path if path not in original_sys_path]
    if added_paths:
        existing_py_path = os.environ.get("PYTHONPATH", "")
        if existing_py_path:
            current_entries = [entry for entry in existing_py_path.split(os.pathsep) if entry]
        else:
            current_entries = []

        merged_entries: list[str] = []
        for entry in added_paths + current_entries:
            if entry and entry not in merged_entries:
                merged_entries.append(entry)

        os.environ["PYTHONPATH"] = os.pathsep.join(merged_entries)

    main_fn = _build_hydra_main(args_cli, app_launcher)

    try:
        main_fn()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    _run()
