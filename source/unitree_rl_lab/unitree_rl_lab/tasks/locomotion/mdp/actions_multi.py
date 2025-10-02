import atexit
import os
from dataclasses import dataclass
from multiprocessing.pool import Pool
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, cast
import math

import numpy as np
import torch
import torch.multiprocessing as torch_mp

from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

from . import asr_pybindings, stiff_utils
from .actions import JointPositionDDCAction, JointPositionDDCActionCfg


_DOUBLE_SUPPORT_STIFF_GAIN = 32.0
_DOUBLE_SUPPORT_VISCO_GAIN = math.sqrt(_DOUBLE_SUPPORT_STIFF_GAIN) * 2.0
_DEBUG_PRINT = False

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
    eps: float


_WORKER_CONTEXT: Optional[_WorkerContext] = None
_COG_GAIN_LOG_PATH = Path(__file__).resolve().parent / "cog_gain.log"
_COG_GAIN_PRINTED = True


def _diag_kmat(size: int, *, value: float = 20) -> "asr_pybindings.kMat":
    """Return a diagonal matrix with `value` on the diagonal wrapped in `kMat`."""

    return asr_pybindings.kMat.from_numpy(np.eye(size, dtype=np.float64) * value)


def _quat_to_rpy(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert XYZW quaternions to roll, pitch, yaw (radians)."""

    quat = np.asarray(quat_xyzw, dtype=np.float64)
    x = quat[..., 0]
    y = quat[..., 1]
    z = quat[..., 2]
    w = quat[..., 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack((roll, pitch, yaw), axis=-1)


def _format_matrix_full(matrix: np.ndarray) -> str:
    """Return a non-summarized string representation of the matrix."""

    if hasattr(matrix, "to_numpy"):
        matrix_np = matrix.to_numpy()
    else:
        matrix_np = np.asarray(matrix)
    return np.array2string(matrix_np, threshold=matrix_np.size)


def _resolve_asr_path(filename: Optional[str]) -> str:
    """Resolve ASR model path relative to this package if needed."""
    if filename is None:
        base = Path(__file__).resolve().parent
        return str(base / "ascii" / "g1_29dof_rev1.asr")
    path = Path(filename)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / filename
    return str(path)


def _init_worker(asr_path: str, joint_names: Sequence[str], eps: float, cog_kz: float) -> None:
    """Initializer executed in each worker process."""
    global _WORKER_CONTEXT, _COG_GAIN_PRINTED

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
    k_z = 500
    asr_pybindings.asrRobotCalcCOGVEFromGainPybind(robot, gain, k_z, task_single, cog_index)
    asr_pybindings.asrRobotCalcCOGVEFromGainPybind(robot, gain, k_z, task_double, cog_index)
    asr_pybindings.asrTaskCountStatusPybind(task_single)
    asr_pybindings.asrTaskCountStatusPybind(task_double)

    if _DEBUG_PRINT:
        gain_np = gain.to_numpy()
        gain_str = _format_matrix_full(gain_np)
        with _COG_GAIN_LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write("asrRobotCalcCOGVEFromGainPybind gain matrix:\n")
            log_file.write(gain_str)
            log_file.write("\n\n")
        cog_stiff_single = asr_pybindings.asrTaskMarkerGetStiffPybind(task_single, cog_index)
        log_path_cog_single = Path(__file__).resolve().parent / "cog_marker_single.log"
        with log_path_cog_single.open("a", encoding="utf-8") as log_file:
            log_file.write("COG marker stiffness (single support):\n")
            log_file.write(_format_matrix_full(cog_stiff_single))
            log_file.write("\n\n")
        cog_stiff_double = asr_pybindings.asrTaskMarkerGetStiffPybind(task_double, cog_index)
        log_path_cog_double = Path(__file__).resolve().parent / "cog_marker_double.log"
        with log_path_cog_double.open("a", encoding="utf-8") as log_file:
            log_file.write("COG marker stiffness (double support):\n")
            log_file.write(_format_matrix_full(cog_stiff_double))
            log_file.write("\n\n")
        if not _COG_GAIN_PRINTED:
            print("asrRobotCalcCOGVEFromGainPybind gain matrix:\n" + gain_str)
            _COG_GAIN_PRINTED = True

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
        eps=eps,
    )


def _compute_stiffness(
    payload: Tuple[
        int,
        Sequence[float],
        bool,
        bool,
        Sequence[float],
        Sequence[float],
    ]
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Worker entry point to compute stiffness/viscosity matrices."""
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker context not initialized")

    env_id, joint_angles_rad, left, right, root_pos, root_rpy = payload
    ctx = _WORKER_CONTEXT
    joint_count = int(ctx.robot.joint_num)

    for link_id, angle in zip(ctx.joint_link_ids, joint_angles_rad):
        ctx.robot.asrRobotLinkSetAngPybind(link_id, angle)
    ctx.robot.asrRobotSetRootPosePybind(root_pos, root_rpy)
    ctx.robot.asrRobotUpdateStatePybind()
    ctx.robot.asrRobotFwdKinematicsPybind()

    if left and right:
        asr_pybindings.asrRobotTaskFwdKinematicsPybind(ctx.robot, ctx.task_double)
        inertia = asr_pybindings.asrRobotCalcInertiaPybind(ctx.robot)
        if _DEBUG_PRINT and env_id == 0:
            log_path_inertia = Path(__file__).resolve().parent / "inertia0.log"
            with log_path_inertia.open("a", encoding="utf-8") as log_file:
                log_file.write(_format_matrix_full(inertia))
                log_file.write("\n\n")
        projected_inertia = ref_k = ref_d = jacob_map = jacob_task = None
        try:
            projected_inertia = asr_pybindings.kMat(joint_count, joint_count)
            asr_pybindings.asrRobotCalcInertiaSinglePybind(
                ctx.robot,
                ctx.pivot_id_left,
                projected_inertia,
            )
            if _DEBUG_PRINT and env_id == 0:
                log_path_proj = Path(__file__).resolve().parent / "proj_inertia0.log"
                with log_path_proj.open("a", encoding="utf-8") as log_file:
                    log_file.write(_format_matrix_full(projected_inertia.to_numpy()[ctx.perm_ix]))
                    log_file.write("\n\n")
            if joint_count > 6:
                reduced_dof = joint_count - 6
                jacob_map = asr_pybindings.kMat(joint_count, reduced_dof)
                asr_pybindings.asrJacobDoubleToJointPybind(
                    ctx.robot,
                    ctx.pivot_id_left,
                    ctx.pivot_id_right,
                    jacob_map,
                )
                jacob_task = asr_pybindings.kMat(ctx.task_double._active_dof, reduced_dof)
                asr_pybindings.asrJacobTaskDoublePybind(
                    ctx.robot,
                    ctx.pivot_id_left,
                    ctx.pivot_id_right,
                    jacob_map,
                    ctx.task_double,
                    jacob_task,
                )
                if _DEBUG_PRINT and env_id == 0:
                    log_path_jac = Path(__file__).resolve().parent / "jacob_double0.log"
                    with log_path_jac.open("a", encoding="utf-8") as log_file:
                        log_file.write(_format_matrix_full(jacob_task.to_numpy()))
                        log_file.write("\n\n")
            ref_k = projected_inertia.clone()
            ref_k.scale_inplace(_DOUBLE_SUPPORT_STIFF_GAIN)
            ref_d = projected_inertia.clone()
            ref_d.scale_inplace(_DOUBLE_SUPPORT_VISCO_GAIN)
            asr_pybindings.asrRVCCalcDCCDoublePybind(
                ctx.robot,
                ctx.pivot_id_left,
                ctx.pivot_id_right,
                ctx.task_double,
                ref_k,
                ref_d,
            )
            stiff_native = np.asarray(ctx.robot._joint_stiff)
            visco_native = np.asarray(ctx.robot._joint_visco)
        finally:
            # Drop references so the underlying kMat allocations are freed after each step.
            ref_k = None
            ref_d = None
            projected_inertia = None
            jacob_map = None
            jacob_task = None
    elif left and not right:
        asr_pybindings.asrRobotTaskFwdKinematicsPybind(ctx.robot, ctx.task_single)
        inertia = asr_pybindings.asrRobotCalcInertiaPybind(ctx.robot)
        if _DEBUG_PRINT and env_id == 0:
            log_path_inertia = Path(__file__).resolve().parent / "inertia0.log"
            with log_path_inertia.open("a", encoding="utf-8") as log_file:
                log_file.write(_format_matrix_full(inertia))
                log_file.write("\n\n")
        projected_inertia = ref_k = ref_d = jacob_task = None
        try:
            projected_inertia = asr_pybindings.kMat(joint_count, joint_count)
            asr_pybindings.asrRobotCalcInertiaSinglePybind(
                ctx.robot,
                ctx.pivot_id_left,
                projected_inertia,
            )
            if _DEBUG_PRINT and env_id == 0:
                log_path_proj = Path(__file__).resolve().parent / "proj_inertia0.log"
                with log_path_proj.open("a", encoding="utf-8") as log_file:
                    log_file.write(_format_matrix_full(projected_inertia.to_numpy()[ctx.perm_ix]))
                    log_file.write("\n\n")
            if joint_count > 0 and ctx.task_single._active_dof > 0:
                jacob_task = asr_pybindings.kMat(ctx.task_single._active_dof, joint_count)
                asr_pybindings.asrJacobTaskSinglePybind(
                    ctx.robot,
                    ctx.pivot_id_left,
                    ctx.task_single,
                    jacob_task,
                )
                if _DEBUG_PRINT and env_id == 0:
                    log_path_jac = Path(__file__).resolve().parent / "jacob_single_left0.log"
                    with log_path_jac.open("a", encoding="utf-8") as log_file:
                        log_file.write(_format_matrix_full(jacob_task.to_numpy()))
                        log_file.write("\n\n")
            ref_k = projected_inertia.clone()
            ref_k.scale_inplace(_DOUBLE_SUPPORT_STIFF_GAIN)
            ref_d = projected_inertia.clone()
            ref_d.scale_inplace(_DOUBLE_SUPPORT_VISCO_GAIN)
            asr_pybindings.asrRVCCalcDCCSinglePybind(
                ctx.robot,
                ctx.pivot_id_left,
                ctx.task_single,
                ref_k,
                ref_d,
            )
            stiff_native = np.asarray(ctx.robot._joint_stiff)
            visco_native = np.asarray(ctx.robot._joint_visco)
        finally:
            ref_k = None
            ref_d = None
            projected_inertia = None
            jacob_task = None
    elif right and not left:
        asr_pybindings.asrRobotTaskFwdKinematicsPybind(ctx.robot, ctx.task_single)
        inertia = asr_pybindings.asrRobotCalcInertiaPybind(ctx.robot)
        if _DEBUG_PRINT and env_id == 0:
            log_path_inertia = Path(__file__).resolve().parent / "inertia0.log"
            with log_path_inertia.open("a", encoding="utf-8") as log_file:
                log_file.write(_format_matrix_full(inertia))
                log_file.write("\n\n")
        projected_inertia = ref_k = ref_d = jacob_task = None
        try:
            projected_inertia = asr_pybindings.kMat(joint_count, joint_count)
            asr_pybindings.asrRobotCalcInertiaSinglePybind(
                ctx.robot,
                ctx.pivot_id_right,
                projected_inertia,
            )
            if _DEBUG_PRINT and env_id == 0:
                log_path_proj = Path(__file__).resolve().parent / "proj_inertia0.log"
                with log_path_proj.open("a", encoding="utf-8") as log_file:
                    log_file.write(_format_matrix_full(projected_inertia.to_numpy()[ctx.perm_ix]))
                    log_file.write("\n\n")
            if joint_count > 0 and ctx.task_single._active_dof > 0:
                jacob_task = asr_pybindings.kMat(ctx.task_single._active_dof, joint_count)
                asr_pybindings.asrJacobTaskSinglePybind(
                    ctx.robot,
                    ctx.pivot_id_right,
                    ctx.task_single,
                    jacob_task,
                )
                if _DEBUG_PRINT and env_id == 0:
                    log_path_jac = Path(__file__).resolve().parent / "jacob_single_right0.log"
                    with log_path_jac.open("a", encoding="utf-8") as log_file:
                        log_file.write(_format_matrix_full(jacob_task.to_numpy()))
                        log_file.write("\n\n")
            ref_k = projected_inertia.clone()
            ref_k.scale_inplace(_DOUBLE_SUPPORT_STIFF_GAIN)
            ref_d = projected_inertia.clone()
            ref_d.scale_inplace(_DOUBLE_SUPPORT_VISCO_GAIN)
            asr_pybindings.asrRVCCalcDCCSinglePybind(
                ctx.robot,
                ctx.pivot_id_right,
                ctx.task_single,
                ref_k,
                ref_d,
            )
            stiff_native = np.asarray(ctx.robot._joint_stiff)
            visco_native = np.asarray(ctx.robot._joint_visco)
        finally:
            ref_k = None
            ref_d = None
            projected_inertia = None
            jacob_task = None
    else:
        if _DEBUG_PRINT:
            projected_debug = None
            try:
                asr_pybindings.asrRobotTaskFwdKinematicsPybind(ctx.robot, ctx.task_single)
                inertia = asr_pybindings.asrRobotCalcInertiaPybind(ctx.robot)
                if env_id == 0:
                    log_path_inertia = Path(__file__).resolve().parent / "inertia0.log"
                    with log_path_inertia.open("a", encoding="utf-8") as log_file:
                        log_file.write(_format_matrix_full(inertia))
                        log_file.write("\n\n")
                projected_debug = asr_pybindings.kMat(joint_count, joint_count)
                asr_pybindings.asrRobotCalcInertiaSinglePybind(
                    ctx.robot,
                    ctx.pivot_id_right,
                    projected_debug,
                )
                if env_id == 0:
                    log_path_proj = Path(__file__).resolve().parent / "proj_inertia0.log"
                    with log_path_proj.open("a", encoding="utf-8") as log_file:
                        log_file.write(_format_matrix_full(projected_debug.to_numpy()[ctx.perm_ix]))
                        log_file.write("\n\n")
            finally:
                projected_debug = None
        stiff_native = np.eye(len(ctx.joint_names), dtype=np.float64) * 150
        visco_native = np.eye(len(ctx.joint_names), dtype=np.float64) * 5.0

    stiff_np = stiff_native[ctx.perm_ix].copy()
    visco_np = visco_native[ctx.perm_ix].copy()

    return env_id, stiff_np, visco_np


class JointPositionDDCMultiAction(JointPositionDDCAction):
    """Multiprocessing variant that offloads DDC stiffness computations."""

    def __init__(self, cfg: JointPositionDDCActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._resolved_asr_path = _resolve_asr_path(cfg.asr_robot_filename)
        cfg_workers = getattr(cfg, "num_workers", None)
        cfg_chunksize = getattr(cfg, "chunksize", 32)

        self._num_workers = 32
        self._chunksize = max(1, cfg_chunksize)
        self._pool: Optional[Pool] = None
        self._elapsed_time_s = 0.0

        if self._num_workers > 1:
            ctx = torch_mp.get_context("fork")
            self._pool = ctx.Pool(
                processes=self._num_workers,
                initializer=_init_worker,
                initargs=(
                    self._resolved_asr_path,
                    self.joint_names,
                    self._ddc_eps,
                    self._ddc_cog_kz,
                ),
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
            raise RuntimeError("Failed to apply actions in single-process mode.")
            return

        # sim_time = self._env.sim.current_time
        # print(f"[JointPositionDDCMultiAction] elapsed {sim_time:.6f}s")

        q_des = self.processed_actions
        q = self._asset.data.joint_pos[:, self._joint_ids]
        qd = self._asset.data.joint_vel[:, self._joint_ids]

        q_cpu = q.detach().cpu()
        joint_angle_array = q_cpu.contiguous().numpy()

        root_pos_w = np.asarray(
            self._asset.data.root_pos_w.detach().cpu().numpy(), dtype=np.float64
        )
        root_quat_w = np.asarray(
            self._asset.data.root_quat_w.detach().cpu().numpy(), dtype=np.float64
        )
        root_rpy = _quat_to_rpy(root_quat_w)

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

        jobs: List[Tuple[int, Sequence[float], bool, bool, Sequence[float], Sequence[float]]] = []
        left_flags = left_in_contact.cpu().numpy()
        right_flags = right_in_contact.cpu().numpy()
        for i in range(self.num_envs):
            root_xyz = tuple(root_pos_w[i].tolist())
            root_rpy_i = tuple(root_rpy[i].tolist())
            jobs.append(
                (
                    i,
                    joint_angle_array[i],
                    bool(left_flags[i]),
                    bool(right_flags[i]),
                    root_xyz,
                    root_rpy_i,
                )
            )

        results = self._pool.map(_compute_stiffness, jobs, chunksize=self._chunksize)

        stiff_np_list: List[Optional[np.ndarray]] = [None] * self.num_envs
        visco_np_list: List[Optional[np.ndarray]] = [None] * self.num_envs
        for env_id, stiff_np, visco_np in results:
            stiff_np_list[env_id] = stiff_np
            visco_np_list[env_id] = visco_np

        if _DEBUG_PRINT:
            log_path_stiff = Path(__file__).resolve().parent / "stiff_np_list0.log"
            with log_path_stiff.open("a", encoding="utf-8") as log_file:
                log_file.write(str(stiff_np_list[0]))
                log_file.write("\n\n")
            log_path_visco = Path(__file__).resolve().parent / "visco_np_list0.log"
            with log_path_visco.open("a", encoding="utf-8") as log_file:
                log_file.write(str(visco_np_list[0]))
                log_file.write("\n\n")
            if any(item is None for item in stiff_np_list) or any(item is None for item in visco_np_list):
                raise RuntimeError("Missing stiffness/viscosity result for one or more environments.")

            log_path_stand = Path(__file__).resolve().parent / "stand0.log"
            with log_path_stand.open("a", encoding="utf-8") as log_file:
                if left_in_contact[0] and right_in_contact[0]:
                    log_file.write("Double\n")
                elif left_in_contact[0] and not right_in_contact[0]:
                    log_file.write("Left\n")
                elif not left_in_contact[0] and right_in_contact[0]:
                    log_file.write("Right\n")
                else:
                    log_file.write("Air\n")

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
        if _DEBUG_PRINT:
            log_path_tau = Path(__file__).resolve().parent / "tau0.log"
            with log_path_tau.open("a", encoding="utf-8") as log_file:
                log_file.write(str(tau[0].cpu().numpy()))
                log_file.write("\n\n")
            log_path_q = Path(__file__).resolve().parent / "q0.log"
            with log_path_q.open("a", encoding="utf-8") as log_file:
                log_file.write(str(q[0].cpu().numpy()))
                log_file.write("\n\n")
            log_path_qd = Path(__file__).resolve().parent / "qd0.log"
            with log_path_qd.open("a", encoding="utf-8") as log_file:
                log_file.write(str(qd[0].cpu().numpy()))
                log_file.write("\n\n")
            log_path_q_des = Path(__file__).resolve().parent / "q_des0.log"
            with log_path_q_des.open("a", encoding="utf-8") as log_file:
                log_file.write(str(q_des[0].cpu().numpy()))
                log_file.write("\n\n")
        tau = torch.clamp(tau, min=-80.0, max=80.0)

        self._asset.set_joint_effort_target(tau, joint_ids=self._joint_ids)


@configclass
class JointPositionDDCMultiActionCfg(JointPositionDDCActionCfg):
    """Config for the multiprocessing-backed DDC action."""

    class_type: type[JointPositionDDCMultiAction] = JointPositionDDCMultiAction
    num_workers: Optional[int] = None
    chunksize: int = 32
