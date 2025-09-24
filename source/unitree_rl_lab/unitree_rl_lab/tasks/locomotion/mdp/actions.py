import torch
import numpy as np
from typing import List, Optional, Tuple

from isaaclab.envs.mdp.actions import actions_cfg as il_actions_cfg
from isaaclab.envs.mdp.actions import joint_actions as il_joint_actions
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.utils import configclass

from . import asr_pybindings # derived from ~/src/kross, ~/src/asura, ~/src/asura_pybind
from . import stiff_utils
from . import asura

def debug_print(msg1, msg2=None):
    return 
    print(msg1)
    if msg2:
        print(msg2)


def _identity_kmat(size: int) -> "asr_pybindings.kMat":
    """Return an identity matrix wrapped in `kMat`."""

    return asr_pybindings.kMat.from_numpy(np.eye(size, dtype=np.float64))

class JointPositionDDCAction(il_joint_actions.JointPositionAction):
    def __init__(self, cfg: il_actions_cfg.JointActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        ASR_PATH = r"/home/daisuke/rl_lab/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/ascii/g1_29dof_rev1.asr"
        # initialize robot in asr_pybindings
        self.asr_robot = asr_pybindings.Robot(ASR_PATH)
        self.asr_robot.asrRobotReadFilePybind(ASR_PATH)
        self.asr_robot._init_joint_matrices() # should not be part of asr_pybindings. should be in high level wrapper. TODO

        # initialize tasks in asr_pybindings
        COG = 0
        dof_joint = int(self.asr_robot.joint_num)
        dof_double = max(dof_joint - 6, 1)

        ## single support task
        self.asr_task_single = asr_pybindings.TaskCore(dof_joint)
        asr_pybindings.asrTaskMarkerInitPybind(self.asr_task_single, COG, asr_pybindings.MarkerType.COG, dof_joint)
        asr_pybindings.asrTaskMarkerOnPybind(self.asr_task_single, COG)

        ## double support task
        self.asr_task_double = asr_pybindings.TaskCore(dof_double)
        asr_pybindings.asrTaskMarkerInitPybind(self.asr_task_double, COG, asr_pybindings.MarkerType.COG, dof_double)
        asr_pybindings.asrTaskMarkerOnPybind(self.asr_task_double, COG)

        ## LIPM-based COG viscoelasticity for both tasks
        self.asr_lipm = asr_pybindings.LTISys()
        w2 = self.asr_lipm.kLIPModelCreate3DPybind(0.6, 0.0)
        gain_np = self.asr_lipm.kLIPModelSetRegulatorPybind(w2, 0.5, 0.5)
        gain = asr_pybindings.kMat.from_numpy(gain_np)
        K_Z = 5.0
        asr_pybindings.asrRobotCalcCOGVEFromGainPybind(self.asr_robot, gain, K_Z, self.asr_task_single, COG)
        asr_pybindings.asrRobotCalcCOGVEFromGainPybind(self.asr_robot, gain, K_Z, self.asr_task_double, COG)
        asr_pybindings.asrTaskCountStatusPybind(self.asr_task_single)
        asr_pybindings.asrTaskCountStatusPybind(self.asr_task_double)

        # find left/right ankle roll link ids for ground contact
        self.pivot_id_left = self.asr_robot.asrRobotLinkFindByNamePybind("left_ankle_roll_link")
        self.pivot_id_right = self.asr_robot.asrRobotLinkFindByNamePybind("right_ankle_roll_link")
        if self.pivot_id_left == -1 or self.pivot_id_right == -1:
            raise ValueError(
                f"Failed to find left/right ankle roll links in ASR robot. "
                f"Found left_id={self.pivot_id_left}, right_id={self.pivot_id_right}. "
                f"Check cfg.asr_robot_filename and link names."
            )
        
        # variables to record stiff and visco 
        self.stiff = torch.zeros(self.num_envs, dof_joint, dof_joint, device=self.device)
        self.visco = torch.zeros(self.num_envs, dof_joint, dof_joint, device=self.device)

        # DDC tuning constants (aligned with ASURA reference controllers)
        self._ddc_cog_kz = K_Z
        self._ddc_eps = 1e-6

        # constants for optimization
        self.joint_names = list(self._joint_names)
        self.joint_names = [name.replace('waist_pitch_joint', 'torso_joint').replace('_joint', '_link') for name in self.joint_names] # convert to asr naming
        self._joint_names_np = np.array(self.joint_names, dtype=object)
        debug_print("type of self._joint_names:", self._joint_names)

        try:
            self._asr_joint_names_native = stiff_utils.get_current_joint_order_names(self.asr_robot)
            self._asr_perm_to_env = stiff_utils.build_perm_from_lists(
                self._asr_joint_names_native, self.joint_names
            )
        except stiff_utils.ReorderError as err:
            raise RuntimeError(
                "Failed to build joint-name permutation between ASR robot and environment joints."
            ) from err
        self._asr_perm_ix = np.ix_(self._asr_perm_to_env, self._asr_perm_to_env)

        # cache contact sensor and body indices so we avoid repeated regex lookups each step
        self._contact_sensor = self._get_contact_sensor()
        (
            self._left_contact_body_ids,
            self._right_contact_body_ids,
        ) = self._resolve_contact_body_ids(self._contact_sensor)

        
    def apply_actions(self):
        q_des = self.processed_actions

        q = self._asset.data.joint_pos[:, self._joint_ids]
        qd = self._asset.data.joint_vel[:, self._joint_ids]
        debug_print("self._joint_ids:", self._joint_ids)

        q_cpu = q.detach().cpu()
        q_cpu_np = q_cpu.contiguous().numpy()
        # print(len(self.joint_names), self.joint_names)
        self.joint_angle_dicts_rad = [dict(zip(self._joint_names_np, row)) for row in q_cpu_np]

        # get Contact State from env
        left_contact = torch.zeros(self.num_envs, device=self.device)
        right_contact = torch.zeros(self.num_envs, device=self.device)
        try:
            contact_sensor = self._contact_sensor
            left_ids = self._left_contact_body_ids
            right_ids = self._right_contact_body_ids
            # Compute per-env contact as 1.0 if any of the matched bodies are in contact (current_contact_time > 0)
            if getattr(contact_sensor.cfg, "track_air_time", False):
                ctime = contact_sensor.data.current_contact_time  # (N, B)
                if len(left_ids) > 0:
                    left_contact = (ctime[:, left_ids] > 0.0).any(dim=1).float()
                if len(right_ids) > 0:
                    right_contact = (ctime[:, right_ids] > 0.0).any(dim=1).float()
            else:
                raise ValueError("Contact sensor does not have 'track_air_time' enabled in its config.")
        except Exception:
            raise RuntimeError("Failed to get contact forces from env. Ensure that contact sensor is added to the scene and configured properly.")
        left_in_contact = left_contact > 0.5
        right_in_contact = right_contact > 0.5

        for i in range(self.num_envs):
            left = bool(left_in_contact[i].item())
            right = bool(right_in_contact[i].item())
            # debug_print(f"Step {self.step_count} Env {i}: left_contact={left} right_contact={right}")
            asura.set_joint_angles_by_name(self.asr_robot, self.joint_angle_dicts_rad[i])
            self.asr_robot.asrRobotUpdateStatePybind()
            self.asr_robot.asrRobotFwdKinematicsPybind()

            if left and right:
                asr_pybindings.asrRobotTaskFwdKinematicsPybind(self.asr_robot, self.asr_task_double)
                asr_pybindings.asrRobotCalcInertiaPybind(self.asr_robot)
                size = max(int(self.asr_robot.joint_num) - 6, 1)
                ref_k = _identity_kmat(size)
                ref_d = _identity_kmat(size)
                asr_pybindings.asrRVCCalcDCCDoublePybind(
                    self.asr_robot,
                    self.pivot_id_left,
                    self.pivot_id_right,
                    self.asr_task_double,
                    ref_k,
                    ref_d,
                    self._ddc_eps,
                )
            
            if left and not right:
                asr_pybindings.asrRobotTaskFwdKinematicsPybind(self.asr_robot, self.asr_task_single)
                asr_pybindings.asrRobotCalcInertiaPybind(self.asr_robot)
                size = int(self.asr_robot.joint_num)
                ref_k = _identity_kmat(size)
                ref_d = _identity_kmat(size)
                asr_pybindings.asrRVCCalcDCCSinglePybind(
                    self.asr_robot,
                    self.pivot_id_left,
                    self.asr_task_single,
                    ref_k,
                    ref_d,
                    self._ddc_eps,
                )

            if not left and right:
                asr_pybindings.asrRobotTaskFwdKinematicsPybind(self.asr_robot, self.asr_task_single)
                asr_pybindings.asrRobotCalcInertiaPybind(self.asr_robot)
                size = int(self.asr_robot.joint_num)
                ref_k = _identity_kmat(size)
                ref_d = _identity_kmat(size)
                asr_pybindings.asrRVCCalcDCCSinglePybind(
                    self.asr_robot,
                    self.pivot_id_right,
                    self.asr_task_single,
                    ref_k,
                    ref_d,
                    self._ddc_eps,
                )
            if not left and not right:
                pass

            stiff_native = np.asarray(self.asr_robot._joint_stiff)
            visco_native = np.asarray(self.asr_robot._joint_visco)
            stiff_np = stiff_native[self._asr_perm_ix].copy()
            visco_np = visco_native[self._asr_perm_ix].copy()
            stiff_tensor = torch.from_numpy(stiff_np).to(device=self.device, dtype=self.stiff.dtype)
            visco_tensor = torch.from_numpy(visco_np).to(device=self.device, dtype=self.visco.dtype)
            self.stiff[i].copy_(stiff_tensor)
            self.visco[i].copy_(visco_tensor)
            
            if not __debug__:
                # Debug-only verification: confirm cached permutation matches legacy helpers.
                stiff_native = np.asarray(self.asr_robot._joint_stiff)
                visco_native = np.asarray(self.asr_robot._joint_visco)
                stiff_legacy = stiff_utils.get_reordered_joint_stiff(
                    self.asr_robot, target_names=self.joint_names, copy=True
                )
                visco_legacy = stiff_utils.get_reordered_joint_visco(
                    self.asr_robot, target_names=self.joint_names, copy=True
                )
                stiff_cached = stiff_native[self._asr_perm_ix]
                visco_cached = visco_native[self._asr_perm_ix]
                if np.allclose(stiff_legacy, stiff_cached):
                    raise RuntimeError("Cached stiffness permutation mismatch with legacy helper.")
                if not np.allclose(visco_legacy, visco_cached):
                    raise RuntimeError("Cached viscosity permutation mismatch with legacy helper.")


        pos_err = (q_des - q).unsqueeze(1)
        vel = qd.unsqueeze(1)
        tau = torch.bmm(pos_err, self.stiff).squeeze(1) - torch.bmm(vel, self.visco).squeeze(1)
        # Clip torques to [-80, 80] as requested
        tau = torch.clamp(tau, min=-80.0, max=80.0)

        self._asset.set_joint_effort_target(tau, joint_ids=self._joint_ids)

    def _get_contact_sensor(self):
        try:
            return self._env.scene.sensors["contact_forces"]
        except KeyError as exc:
            raise RuntimeError(
                "Contact sensor 'contact_forces' is missing from the scene. Ensure it is configured before initializing the action."
            ) from exc

    def _resolve_contact_body_ids(self, contact_sensor) -> Tuple[List[int], List[int]]:
        left_ids, _ = contact_sensor.find_bodies([r".*left.*ankle.*roll.*"])
        right_ids, _ = contact_sensor.find_bodies([r".*right.*ankle.*roll.*"])
        if len(left_ids) == 0 or len(right_ids) == 0:
            raise ValueError("Failed to find left/right ankle roll links using regex during initialization.")
        return left_ids, right_ids

@configclass
class JointPositionDDCActionCfg(il_actions_cfg.JointActionCfg):
    """Config for joint position-to-torque (PD) action.

    - `kp`, `kd`: global gains (float) or per-joint via dict of regex->float.
    - `use_default_offset`: if True, offsets are set to articulation default positions (absolute position commands).
    - `scale`/`offset`/`clip` follow the same semantics as other JointActionCfgs.
    - Optional `tau_clip`: dict of regex->(min, max) to clip output torque per joint.
    """

    class_type: type[il_joint_actions.JointAction] = JointPositionDDCAction

    # PD gains
    kp: float | dict[str, float] = 50.0
    kd: float | dict[str, float] = 1.0

    # treat action as absolute desired pos around defaults
    use_default_offset: bool = True

    # optional torque clipping on the final torque (per joint)
    tau_clip: dict[str, Tuple[float, float]] | None = None

    # ASR: path to robot model file required by asr_pybindings.Robot
    asr_robot_filename: Optional[str] = None
