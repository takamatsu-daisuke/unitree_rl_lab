import atexit
import math
import os
from dataclasses import dataclass
from multiprocessing.pool import Pool
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.multiprocessing as torch_mp

from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

from . import asr_pybindings, stiff_utils
from .actions import JointPositionDDCAction, JointPositionDDCActionCfg


@dataclass
class _WorkerContext:
    """Container for per-process reusable ASR objects."""

    robot: "asr_pybindings.Robot"
    task_single: "asr_pybindings.TaskCore"
    task_double: "asr_pybindings.TaskCore"
    pivot_id_left: int
    pivot_id_right: int
    joint_link_ids: List[int]
    joint_names: List[str]
    perm_to_env: np.ndarray
    perm_ix: Tuple[np.ndarray, np.ndarray]


_WORKER_CONTEXT: Optional[_WorkerContext] = None


def _resolve_asr_path(filename: Optional[str]) -> str:
    """Resolve ASR model path relative to this package if needed."""
    if filename is None:
        base = Path(__file__).resolve().parent
        return str(base / "ascii" / "g1_29dof_rev1.asr")
    path = Path(filename)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / filename
    return str(path)


def _init_worker(asr_path: str, joint_names: Sequence[str]) -> None:
    """Initializer executed in each worker process."""
    global _WORKER_CONTEXT

    robot = asr_pybindings.Robot(asr_path)
    robot.asrRobotReadFilePybind(asr_path)
    robot._init_joint_matrices()

    cog_index = 0
    dof_joint = int(robot.joint_num)
    dof_double = max(dof_joint - 6, 1)

    task_single = asr_pybindings.TaskCore(dof_joint)
    asr_pybindings.asrTaskMarkerInitPybind(task_single, cog_index, asr_pybindings.MarkerType.COG, dof_joint)
    asr_pybindings.asrTaskMarkerOnPybind(task_single, cog_index)

    task_double = asr_pybindings.TaskCore(dof_double)
    asr_pybindings.asrTaskMarkerInitPybind(task_double, cog_index, asr_pybindings.MarkerType.COG, dof_double)
    asr_pybindings.asrTaskMarkerOnPybind(task_double, cog_index)

    lipm = asr_pybindings.LTISys()
    w2 = lipm.kLIPModelCreate3DPybind(0.88, 0.0)
    gain_np = lipm.kLIPModelSetRegulatorPybind(w2, 0.5, 0.5)
    gain = asr_pybindings.kMat.from_numpy(gain_np)
    k_z = 500.0
    asr_pybindings.asrRobotCalcCOGVEFromGainPybind(robot, gain, k_z, task_single, cog_index)
    asr_pybindings.asrRobotCalcCOGVEFromGainPybind(robot, gain, k_z, task_double, cog_index)
    asr_pybindings.asrTaskCountStatusPybind(task_single)
    asr_pybindings.asrTaskCountStatusPybind(task_double)

    pivot_id_left = robot.asrRobotLinkFindByNamePybind("left_ankle_roll_link")
    pivot_id_right = robot.asrRobotLinkFindByNamePybind("right_ankle_roll_link")
    if pivot_id_left == -1 or pivot_id_right == -1:
        raise ValueError(
            "Failed to find ankle roll links in ASR robot for multiprocessing workers."
        )

    joint_names_list = list(joint_names)

    joint_link_ids: List[int] = []
    for name in joint_names_list:
        link_id = robot.asrRobotLinkFindByNamePybind(name)
        if link_id == -1:
            raise ValueError(f"Joint '{name}' not found in ASR model {asr_path}")
        joint_link_ids.append(link_id)

    try:
        asr_joint_names_native = stiff_utils.get_current_joint_order_names(robot)
        perm_to_env = stiff_utils.build_perm_from_lists(asr_joint_names_native, joint_names_list)
    except stiff_utils.ReorderError as err:
        raise RuntimeError(
            "Failed to build joint-name permutation between ASR robot and environment joints in worker."
        ) from err
    perm_ix = np.ix_(perm_to_env, perm_to_env)

    _WORKER_CONTEXT = _WorkerContext(
        robot=robot,
        task_single=task_single,
        task_double=task_double,
        pivot_id_left=pivot_id_left,
        pivot_id_right=pivot_id_right,
        joint_link_ids=joint_link_ids,
        joint_names=joint_names_list,
        perm_to_env=perm_to_env,
        perm_ix=perm_ix,
    )


def _compute_stiffness(payload: Tuple[int, Sequence[float], bool, bool]) -> Tuple[int, np.ndarray, np.ndarray]:
    """Worker entry point to compute stiffness/viscosity matrices."""
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker context not initialized")

    env_id, joint_angles_rad, left, right = payload
    ctx = _WORKER_CONTEXT

    joint_angles_deg = [math.degrees(angle) for angle in joint_angles_rad]
    for link_id, angle_deg in zip(ctx.joint_link_ids, joint_angles_deg):
        ctx.robot.asrRobotLinkSetAngPybind(link_id, angle_deg)
    ctx.robot.asrRobotUpdateStatePybind()
    ctx.robot.asrRobotFwdKinematicsPybind()

    eps = 1e-6
    gain_scale = 50.0

    if left and right:
        asr_pybindings.asrRobotTaskFwdKinematicsPybind(ctx.robot, ctx.task_double)
        asr_pybindings.asrRobotCalcInertiaPybind(ctx.robot)
        inertia = asr_pybindings.kMat(ctx.robot.joint_num, ctx.robot.joint_num)
        asr_pybindings.asrRobotCalcInertiaSinglePybind(ctx.robot, ctx.pivot_id_left, inertia)
        M = inertia.to_numpy()
        M = 0.5 * (M + M.T)
        ref_k = asr_pybindings.kMat.from_numpy(M * gain_scale + np.eye(M.shape[0]) * eps)
        ref_d = asr_pybindings.kMat.from_numpy(
            M * (2.0 * np.sqrt(gain_scale)) + np.eye(M.shape[0]) * (2.0 * np.sqrt(gain_scale) * eps)
        )
        asr_pybindings.asrRVCCalcDCCDoublePybind(
            ctx.robot, ctx.pivot_id_left, ctx.pivot_id_right, ctx.task_double, ref_k, ref_d, eps
        )
    elif left and not right:
        asr_pybindings.asrRobotTaskFwdKinematicsPybind(ctx.robot, ctx.task_single)
        asr_pybindings.asrRobotCalcInertiaPybind(ctx.robot)
        inertia = asr_pybindings.kMat(ctx.robot.joint_num, ctx.robot.joint_num)
        asr_pybindings.asrRobotCalcInertiaSinglePybind(ctx.robot, ctx.pivot_id_left, inertia)
        M = inertia.to_numpy()
        M = 0.5 * (M + M.T)
        ref_k = asr_pybindings.kMat.from_numpy(M * gain_scale + np.eye(M.shape[0]) * eps)
        ref_d = asr_pybindings.kMat.from_numpy(
            M * (2.0 * np.sqrt(gain_scale)) + np.eye(M.shape[0]) * (2.0 * np.sqrt(gain_scale) * eps)
        )
        asr_pybindings.asrRVCCalcDCCSinglePybind(ctx.robot, ctx.pivot_id_left, ctx.task_single, ref_k, ref_d, eps)
    elif right and not left:
        asr_pybindings.asrRobotTaskFwdKinematicsPybind(ctx.robot, ctx.task_single)
        asr_pybindings.asrRobotCalcInertiaPybind(ctx.robot)
        inertia = asr_pybindings.kMat(ctx.robot.joint_num, ctx.robot.joint_num)
        asr_pybindings.asrRobotCalcInertiaSinglePybind(ctx.robot, ctx.pivot_id_right, inertia)
        M = inertia.to_numpy()
        M = 0.5 * (M + M.T)
        ref_k = asr_pybindings.kMat.from_numpy(M * gain_scale + np.eye(M.shape[0]) * eps)
        ref_d = asr_pybindings.kMat.from_numpy(
            M * (2.0 * np.sqrt(gain_scale)) + np.eye(M.shape[0]) * (2.0 * np.sqrt(gain_scale) * eps)
        )
        asr_pybindings.asrRVCCalcDCCSinglePybind(ctx.robot, ctx.pivot_id_right, ctx.task_single, ref_k, ref_d, eps)
    # If both feet are in the air, keep the previous stiffness/viscosity matrices.

    stiff_native = np.asarray(ctx.robot._joint_stiff)
    visco_native = np.asarray(ctx.robot._joint_visco)
    stiff_np = stiff_native[ctx.perm_ix].copy()
    visco_np = visco_native[ctx.perm_ix].copy()

    if not __debug__:
        stiff_legacy = stiff_utils.get_reordered_joint_stiff(
            ctx.robot, target_names=ctx.joint_names, copy=True
        )
        visco_legacy = stiff_utils.get_reordered_joint_visco(
            ctx.robot, target_names=ctx.joint_names, copy=True
        )
        stiff_cached = stiff_native[ctx.perm_ix]
        visco_cached = visco_native[ctx.perm_ix]
        if np.allclose(stiff_legacy, stiff_cached):
            raise RuntimeError("Cached stiffness permutation mismatch with legacy helper (worker).")
        if not np.allclose(visco_legacy, visco_cached):
            raise RuntimeError("Cached viscosity permutation mismatch with legacy helper (worker).")

    return env_id, stiff_np, visco_np


class JointPositionDDCMultiAction(JointPositionDDCAction):
    """Multiprocessing variant that offloads DDC stiffness computations."""

    def __init__(self, cfg: JointPositionDDCActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._resolved_asr_path = _resolve_asr_path(cfg.asr_robot_filename)
        cfg_workers = getattr(cfg, "num_workers", None)
        cfg_chunksize = getattr(cfg, "chunksize", 32)

        self._num_workers = cfg_workers or os.cpu_count() or 1
        self._num_workers = 32#max(1, min(self._num_workers, self.num_envs))
        self._chunksize = max(1, cfg_chunksize)
        self._pool: Optional[Pool] = None

        if self._num_workers > 1:
            ctx = torch_mp.get_context("fork")
            self._pool = ctx.Pool(
                processes=self._num_workers,
                initializer=_init_worker,
                initargs=(self._resolved_asr_path, self.joint_names),
            )
            atexit.register(self._shutdown_pool)
        else:
            self._num_workers = 1  # Explicitly mark single-process fallback.

    def _shutdown_pool(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.close()
            pool.join()

    def __del__(self) -> None:
        self._shutdown_pool()

    def apply_actions(self) -> None:
        if self._pool is None:
            super().apply_actions()
            return

        q_des = self.processed_actions
        q = self._asset.data.joint_pos[:, self._joint_ids]
        qd = self._asset.data.joint_vel[:, self._joint_ids]

        q_cpu = q.detach().cpu()
        joint_angle_array = q_cpu.contiguous().numpy()

        left_contact = torch.zeros(self.num_envs, device=self.device)
        right_contact = torch.zeros(self.num_envs, device=self.device)
        try:
            contact_sensor = self._contact_sensor
            left_ids = self._left_contact_body_ids
            right_ids = self._right_contact_body_ids
            if getattr(contact_sensor.cfg, "track_air_time", False):
                ctime = contact_sensor.data.current_contact_time
                if len(left_ids) > 0:
                    left_contact = (ctime[:, left_ids] > 0.0).any(dim=1).float()
                if len(right_ids) > 0:
                    right_contact = (ctime[:, right_ids] > 0.0).any(dim=1).float()
            else:
                raise ValueError("Contact sensor does not have 'track_air_time' enabled in its config.")
        except Exception as exc:
            raise RuntimeError("Failed to get contact forces from env. Ensure contact sensor is configured.") from exc

        left_in_contact = left_contact > 0.5
        right_in_contact = right_contact > 0.5

        jobs: List[Tuple[int, Sequence[float], bool, bool]] = []
        left_flags = left_in_contact.cpu().numpy()
        right_flags = right_in_contact.cpu().numpy()
        for i in range(self.num_envs):
            jobs.append(
                (
                    i,
                    joint_angle_array[i],
                    bool(left_flags[i]),
                    bool(right_flags[i]),
                )
            )

        results = self._pool.map(_compute_stiffness, jobs, chunksize=self._chunksize)

        stiff_np_list: List[Optional[np.ndarray]] = [None] * self.num_envs
        visco_np_list: List[Optional[np.ndarray]] = [None] * self.num_envs
        for env_id, stiff_np, visco_np in results:
            stiff_np_list[env_id] = stiff_np
            visco_np_list[env_id] = visco_np

        if any(item is None for item in stiff_np_list) or any(item is None for item in visco_np_list):
            raise RuntimeError("Missing stiffness/viscosity result for one or more environments.")

        stiff_arrays = cast(List[np.ndarray], stiff_np_list)
        visco_arrays = cast(List[np.ndarray], visco_np_list)

        stiff_stack = torch.as_tensor(
            np.asarray(stiff_arrays),
            device=self.device,
            dtype=self.stiff.dtype,
        )
        visco_stack = torch.as_tensor(
            np.asarray(visco_arrays),
            device=self.device,
            dtype=self.visco.dtype,
        )
        self.stiff.copy_(stiff_stack)
        self.visco.copy_(visco_stack)

        pos_err = (q_des - q).unsqueeze(1)
        vel = qd.unsqueeze(1)
        tau = torch.bmm(pos_err, self.stiff).squeeze(1) - torch.bmm(vel, self.visco).squeeze(1)
        tau = torch.clamp(tau, min=-80.0, max=80.0)

        self._asset.set_joint_effort_target(tau, joint_ids=self._joint_ids)


@configclass
class JointPositionDDCMultiActionCfg(JointPositionDDCActionCfg):
    """Config for the multiprocessing-backed DDC action."""

    class_type: type[JointPositionDDCMultiAction] = JointPositionDDCMultiAction
    num_workers: Optional[int] = None
    chunksize: int = 32
