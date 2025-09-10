import os
import math
import multiprocessing as mp
from typing import Optional, Tuple, List

import numpy as np
import torch

from isaaclab.envs.mdp.actions import actions_cfg as il_actions_cfg
from isaaclab.envs.mdp.actions import joint_actions as il_joint_actions
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

# Local modules (compiled pybind + helpers)
from . import asr_pybindings  # type: ignore
from . import asura  # helper to set joint angles by name
from . import stiff_utils


#
# Worker-side persistent state
#
_WORKER: dict = {}

def _resolve_asr_path(path: str) -> str:
    """Return absolute path for ASR file.

    If `path` is relative, resolve relative to this module directory.
    """
    if os.path.isabs(path):
        return path
    base_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(base_dir, path))


def _init_worker(asr_path: str, joint_names: List[str]) -> None:
    """Initializer for worker processes.

    Builds and caches heavy, constant objects in the worker:
    - ASR robot model, tasks, link IDs
    - LIPM regulator and related setup
    - Target joint name order for reordering
    """
    # Robot
    robot = asr_pybindings.Robot(asr_path)
    robot.asrRobotReadFilePybind(asr_path)
    robot._init_joint_matrices()  # ensure stiffness/viscosity matrices exist

    # Tasks
    COG = 0
    dof_joint = int(robot.joint_num)
    dof_double = max(dof_joint - 6, 1)

    task_single = asr_pybindings.TaskCore(dof_joint)
    asr_pybindings.asrTaskMarkerInitPybind(task_single, COG, asr_pybindings.MarkerType.COG, dof_joint)
    asr_pybindings.asrTaskMarkerOnPybind(task_single, COG)

    task_double = asr_pybindings.TaskCore(dof_double)
    asr_pybindings.asrTaskMarkerInitPybind(task_double, COG, asr_pybindings.MarkerType.COG, dof_double)
    asr_pybindings.asrTaskMarkerOnPybind(task_double, COG)

    # LIPM-based COG viscoelasticity (same for both tasks)
    lipm = asr_pybindings.LTISys()
    w2 = lipm.kLIPModelCreate3DPybind(0.88, 0.0)
    gain_np = lipm.kLIPModelSetRegulatorPybind(w2, 0.5, 0.5)
    gain = asr_pybindings.kMat.from_numpy(gain_np)
    K_Z = 500.0
    asr_pybindings.asrRobotCalcCOGVEFromGainPybind(robot, gain, K_Z, task_single, COG)
    asr_pybindings.asrRobotCalcCOGVEFromGainPybind(robot, gain, K_Z, task_double, COG)
    asr_pybindings.asrTaskCountStatusPybind(task_single)
    asr_pybindings.asrTaskCountStatusPybind(task_double)

    # Link IDs for contacts
    pivot_id_left = robot.asrRobotLinkFindByNamePybind("left_ankle_roll_link")
    pivot_id_right = robot.asrRobotLinkFindByNamePybind("right_ankle_roll_link")
    if pivot_id_left == -1 or pivot_id_right == -1:
        raise ValueError(
            f"Failed to find left/right ankle roll links in ASR robot. "
            f"Found left_id={pivot_id_left}, right_id={pivot_id_right}. "
            f"Check cfg.asr_robot_filename and link names."
        )

    _WORKER["robot"] = robot
    _WORKER["task_single"] = task_single
    _WORKER["task_double"] = task_double
    _WORKER["pivot_left"] = pivot_id_left
    _WORKER["pivot_right"] = pivot_id_right
    _WORKER["joint_names"] = list(joint_names)
    _WORKER["worker_name"] = mp.current_process().name
    _WORKER["pid"] = os.getpid()
    _WORKER["last_printed_step"] = -1


def _compute_stiff_visco(joint_angles_dict: dict[str, float], left: bool, right: bool, step_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute stiffness/viscosity matrices for a single env snapshot.

    Assumes worker initializer has populated _WORKER with robot/tasks.
    """
    robot = _WORKER["robot"]
    task_single = _WORKER["task_single"]
    task_double = _WORKER["task_double"]
    pivot_left = _WORKER["pivot_left"]
    pivot_right = _WORKER["pivot_right"]
    joint_names = _WORKER["joint_names"]
    # print step progress once per worker per step
    last = _WORKER.get("last_printed_step", -1)
    if step_index != last:
        _WORKER["last_printed_step"] = step_index
        wid = _WORKER.get("worker_name")
        pid = _WORKER.get("pid")
        print(f"[DDC Worker {wid or '?'} pid={pid}] step={step_index}", flush=True)

    # Update robot state from angles (input radians, ASR expects degrees)
    asura.set_joint_angles_by_name(robot, joint_angles_dict)
    robot.asrRobotUpdateStatePybind()
    robot.asrRobotFwdKinematicsPybind()

    # Support condition branches mirror actions.py
    if left and right:
        asr_pybindings.asrRobotTaskFwdKinematicsPybind(robot, task_double)
        asr_pybindings.asrRobotCalcInertiaPybind(robot)
        inertia = asr_pybindings.kMat(robot.joint_num, robot.joint_num)
        asr_pybindings.asrRobotCalcInertiaSinglePybind(robot, pivot_left, inertia)
        P = 5.0
        M = inertia.to_numpy()
        M = 0.5 * (M + M.T)
        eps = 1e-6
        ref_k = asr_pybindings.kMat.from_numpy(M * P + np.eye(M.shape[0]) * eps)
        ref_d = asr_pybindings.kMat.from_numpy(M * (2.0 * np.sqrt(P)) + np.eye(M.shape[0]) * (2.0 * np.sqrt(P) * eps))
        asr_pybindings.asrRVCCalcDCCDoublePybind(robot, pivot_left, pivot_right, task_double, ref_k, ref_d, 1e-6)

    if left and not right:
        asr_pybindings.asrRobotTaskFwdKinematicsPybind(robot, task_single)
        asr_pybindings.asrRobotCalcInertiaPybind(robot)
        inertia = asr_pybindings.kMat(robot.joint_num, robot.joint_num)
        asr_pybindings.asrRobotCalcInertiaSinglePybind(robot, pivot_left, inertia)
        P = 5.0
        M = inertia.to_numpy()
        M = 0.5 * (M + M.T)
        eps = 1e-6
        ref_k = asr_pybindings.kMat.from_numpy(M * P + np.eye(M.shape[0]) * eps)
        ref_d = asr_pybindings.kMat.from_numpy(M * (2.0 * np.sqrt(P)) + np.eye(M.shape[0]) * (2.0 * np.sqrt(P) * eps))
        asr_pybindings.asrRVCCalcDCCSinglePybind(robot, pivot_left, task_single, ref_k, ref_d, 1e-6)

    if (not left) and right:
        asr_pybindings.asrRobotTaskFwdKinematicsPybind(robot, task_single)
        asr_pybindings.asrRobotCalcInertiaPybind(robot)
        inertia = asr_pybindings.kMat(robot.joint_num, robot.joint_num)
        asr_pybindings.asrRobotCalcInertiaSinglePybind(robot, pivot_right, inertia)
        P = 5.0
        M = inertia.to_numpy()
        M = 0.5 * (M + M.T)
        eps = 1e-6
        ref_k = asr_pybindings.kMat.from_numpy(M * P + np.eye(M.shape[0]) * eps)
        ref_d = asr_pybindings.kMat.from_numpy(M * (2.0 * np.sqrt(P)) + np.eye(M.shape[0]) * (2.0 * np.sqrt(P) * eps))
        asr_pybindings.asrRVCCalcDCCSinglePybind(robot, pivot_right, task_single, ref_k, ref_d, 1e-6)

    # no-contact branch does nothing

    # Reorder to target joint order expected by the env
    stiff_np = stiff_utils.get_reordered_joint_stiff(robot, target_names=joint_names)
    visco_np = stiff_utils.get_reordered_joint_visco(robot, target_names=joint_names)
    return stiff_np, visco_np


class JointPositionDDCMultiAction(il_joint_actions.JointPositionAction):
    """Multi-process variant of JointPositionAction computing DDC per env in parallel.

    Heavy ASR computations are offloaded to a process pool; fixed data lives
    in worker processes and is reused across steps.
    """

    def __init__(self, cfg: il_actions_cfg.JointActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Resolve ASR model path: prefer cfg if provided, otherwise use the same default as actions.py
        default_asr_path = (
            r"/home/daisuke/rl_lab/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/ascii/g1_29dof_rev1.asr"
        )
        cfg_path = getattr(cfg, "asr_robot_filename", None)
        asr_path = cfg_path if (cfg_path and isinstance(cfg_path, str)) else default_asr_path
        asr_path = _resolve_asr_path(asr_path)
        if not os.path.exists(asr_path):
            raise FileNotFoundError(
                f"ASR model not found: '{asr_path}'. If relative, it is resolved relative to '{os.path.dirname(__file__)}'."
            )
        self._asr_path = asr_path

        # Prepare joint-name mapping (Isaac -> ASR link names)
        self.joint_names = list(self._joint_names)
        self.joint_names = [
            name.replace("waist_pitch_joint", "torso_joint").replace("_joint", "_link")
            for name in self.joint_names
        ]

        # Allocate buffers for stiffness/viscosity (use env joint count)
        dof = len(self.joint_names)
        self.stiff = torch.zeros(self.num_envs, dof, dof, device=self.device)
        self.visco = torch.zeros(self.num_envs, dof, dof, device=self.device)

        # Process pool created lazily
        self._pool: Optional[mp.pool.Pool] = None
        self._num_workers: Optional[int] = None
        self._step_index: int = 0

    # ----- pool management -----
    def _ensure_pool(self) -> None:
        if self._pool is not None:
            return
        # Worker count: up to num_envs but not exceeding CPU count
        cfg_workers = getattr(self.cfg, "num_workers", None)
        if cfg_workers is not None and cfg_workers > 0:
            procs = min(cfg_workers, self.num_envs)
        else:
            procs = min(os.cpu_count() or 1, self.num_envs)
        self._num_workers = procs
        start_method = getattr(self.cfg, "mp_start_method", "spawn")
        try:
            ctx = mp.get_context(start_method)
        except ValueError:
            # Fallback if invalid method provided
            ctx = mp.get_context("spawn")
            start_method = "spawn"
        self._pool = ctx.Pool(
            processes=procs,
            initializer=_init_worker,
            initargs=(self._asr_path, self.joint_names),
        )

    def close_pool(self) -> None:
        pool = getattr(self, "_pool", None)
        if pool is not None:
            try:
                pool.close()
                pool.join()
            finally:
                self._pool = None

    def __del__(self) -> None:
        try:
            self.close_pool()
        except Exception:
            pass

    # ----- main step -----
    def apply_actions(self) -> None:
        q_des = self.processed_actions
        q = self._asset.data.joint_pos[:, self._joint_ids]
        qd = self._asset.data.joint_vel[:, self._joint_ids]

        # Build per-env joint-angle dicts (radians)
        q_cpu = q.detach().cpu()
        joint_angle_dicts_rad = [
            {self.joint_names[j]: float(q_cpu[i, j].item()) for j in range(q_cpu.shape[1])}
            for i in range(q_cpu.shape[0])
        ]

        # Contact states from env
        left_contact = torch.zeros(self.num_envs, device=self.device)
        right_contact = torch.zeros(self.num_envs, device=self.device)
        try:
            contact_sensor = self._env.scene.sensors["contact_forces"]
            left_ids, _ = contact_sensor.find_bodies([r".*left.*ankle.*roll.*"])  # type: ignore[arg-type]
            right_ids, _ = contact_sensor.find_bodies([r".*right.*ankle.*roll.*"])  # type: ignore[arg-type]
            if len(left_ids) == 0 or len(right_ids) == 0:
                raise ValueError("Failed to find left/right ankle roll links using regex.")
            if getattr(contact_sensor.cfg, "track_air_time", False):
                ctime = contact_sensor.data.current_contact_time
                if len(left_ids) > 0:
                    left_contact = (ctime[:, left_ids] > 0.0).any(dim=1).float()
                if len(right_ids) > 0:
                    right_contact = (ctime[:, right_ids] > 0.0).any(dim=1).float()
            else:
                raise ValueError("Contact sensor does not have 'track_air_time' enabled in its config.")
        except Exception:
            raise RuntimeError(
                "Failed to get contact forces from env. Ensure that contact sensor is added to the scene and configured properly."
            )

        left_in_contact = left_contact > 0.5
        right_in_contact = right_contact > 0.5

        # Parallel compute per-env stiffness/viscosity
        self._ensure_pool()
        tasks = []
        step_idx = self._step_index
        for i in range(self.num_envs):
            tasks.append((
                joint_angle_dicts_rad[i],
                bool(left_in_contact[i].item()),
                bool(right_in_contact[i].item()),
                step_idx,
            ))
            print(i)

        # Use starmap to pass multiple args without lambdas (pickle-safe)
        results: List[tuple[np.ndarray, np.ndarray]] = self._pool.starmap(_compute_stiff_visco, tasks)

        # Fill tensors
        for i, (stiff_np, visco_np) in enumerate(results):
            self.stiff[i] = torch.from_numpy(stiff_np).to(device=self.device, dtype=self.stiff.dtype)
            self.visco[i] = torch.from_numpy(visco_np).to(device=self.device, dtype=self.visco.dtype)

        # Compute torque via batched PD with environment-specific gains
        pos_err = (q_des - q).unsqueeze(1)
        vel = qd.unsqueeze(1)
        tau = torch.bmm(pos_err, self.stiff).squeeze(1) - torch.bmm(vel, self.visco).squeeze(1)
        tau = torch.clamp(tau, min=-80.0, max=80.0)

        self._asset.set_joint_effort_target(tau, joint_ids=self._joint_ids)
        # increment step counter after successful application
        self._step_index += 1

@configclass
class JointPositionDDCMultiActionCfg(il_actions_cfg.JointActionCfg):
    """Config for multi-process DDC joint action.

    Same semantics as JointActionCfg with optional ASR robot filename.
    """

    class_type: type[il_joint_actions.JointAction] = JointPositionDDCMultiAction

    # PD gains
    kp: float | dict[str, float] = 50.0
    kd: float | dict[str, float] = 1.0

    # treat action as absolute desired pos around defaults
    use_default_offset: bool = True

    # optional torque clipping on the final torque (per joint)
    tau_clip: dict[str, Tuple[float, float]] | None = None

    # ASR: path to robot model file used by worker initializer
    asr_robot_filename: Optional[str] = None

    # Number of worker processes (default: min(cpu_count, num_envs))
    num_workers: Optional[int] = None

    # Multiprocessing start method (default: 'spawn' — recommended for GPU/PhysX)
    mp_start_method: str = "spawn"
