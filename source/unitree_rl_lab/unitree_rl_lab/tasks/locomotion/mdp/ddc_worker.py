"""ASR-backed stiffness computation helpers for multiprocessing workers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import asr_pybindings, stiff_utils


@dataclass
class DDCWorkerConfig:
    """Configuration passed to each worker process."""

    asr_path: str
    joint_names: List[str]


@dataclass
class _WorkerContext:
    """Cache of ASR-related objects reused inside a worker."""

    robot: "asr_pybindings.Robot"
    task_single: "asr_pybindings.TaskCore"
    task_double: "asr_pybindings.TaskCore"
    pivot_id_left: int
    pivot_id_right: int
    joint_link_ids: List[int]
    joint_names: List[str]


_WORKER_CONTEXT: Optional[_WorkerContext] = None


def init_worker(cfg: DDCWorkerConfig) -> None:
    """Initializer executed in each worker process."""
    global _WORKER_CONTEXT

    robot = asr_pybindings.Robot(cfg.asr_path)
    robot.asrRobotReadFilePybind(cfg.asr_path)
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
        raise ValueError("Failed to find ankle roll links in ASR robot for multiprocessing workers.")

    joint_link_ids: List[int] = []
    for name in cfg.joint_names:
        link_id = robot.asrRobotLinkFindByNamePybind(name)
        if link_id == -1:
            raise ValueError(f"Joint '{name}' not found in ASR model {cfg.asr_path}")
        joint_link_ids.append(link_id)

    _WORKER_CONTEXT = _WorkerContext(
        robot=robot,
        task_single=task_single,
        task_double=task_double,
        pivot_id_left=pivot_id_left,
        pivot_id_right=pivot_id_right,
        joint_link_ids=joint_link_ids,
        joint_names=list(cfg.joint_names),
    )


def compute_stiffness(payload: Tuple[int, Sequence[float], bool, bool]) -> Tuple[int, np.ndarray, np.ndarray]:
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
    gain_scale = 5.0

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

    stiff_np = stiff_utils.get_reordered_joint_stiff(ctx.robot, target_names=ctx.joint_names)
    visco_np = stiff_utils.get_reordered_joint_visco(ctx.robot, target_names=ctx.joint_names)
    return env_id, stiff_np, visco_np
