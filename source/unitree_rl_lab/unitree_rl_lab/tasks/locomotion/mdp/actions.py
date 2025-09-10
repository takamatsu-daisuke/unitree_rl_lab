import torch
import numpy as np
from typing import Optional, Tuple

from isaaclab.envs.mdp.actions import actions_cfg as il_actions_cfg
from isaaclab.envs.mdp.actions import joint_actions as il_joint_actions
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.utils import configclass

from . import asr_pybindings # derived from ~/src/kross, ~/src/asura, ~/src/asura_pybind
from . import stiff_utils
from . import asura

def debug_print(msg1, msg2=""):
    return 
    print(msg1)
    if msg2:
        print(msg2)

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
        w2 = self.asr_lipm.kLIPModelCreate3DPybind(0.88, 0.0)
        gain_np = self.asr_lipm.kLIPModelSetRegulatorPybind(w2, 0.5, 0.5)
        gain = asr_pybindings.kMat.from_numpy(gain_np)
        K_Z = 500.0
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

        # constants for optimization
        self.joint_names = list(self._joint_names)
        self.joint_names = [name.replace('waist_pitch_joint', 'torso_joint').replace('_joint', '_link') for name in self.joint_names] # convert to asr naming
        debug_print("type of self._joint_names:", self._joint_names)

    def apply_actions(self):
        q_des = self.processed_actions

        q = self._asset.data.joint_pos[:, self._joint_ids]
        qd = self._asset.data.joint_vel[:, self._joint_ids]
        debug_print("self._joint_ids:", self._joint_ids)

        q_cpu = q.detach().cpu()
        # print(len(self.joint_names), self.joint_names)
        self.joint_angle_dicts_rad = [
            {self.joint_names[j]: float(q_cpu[i, j].item()) for j in range(q_cpu.shape[1])}
            for i in range(q_cpu.shape[0])
        ]

        # get Contact State from env
        left_contact = torch.zeros(self.num_envs, device=self.device)
        right_contact = torch.zeros(self.num_envs, device=self.device)
        try:
            contact_sensor = self._env.scene.sensors["contact_forces"]
            # Resolve body indices for left/right using regex patterns.
            left_ids, _ = contact_sensor.find_bodies([r".*left.*ankle.*roll.*"])  # type: ignore[arg-type]
            right_ids, _ = contact_sensor.find_bodies([r".*right.*ankle.*roll.*"])  # type: ignore[arg-type]
            if len(left_ids) == 0 or  len(right_ids) == 0:
                raise ValueError("Failed to find left/right ankle roll links using regex.")
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
                inertia = asr_pybindings.kMat(self.asr_robot.joint_num, self.asr_robot.joint_num)
                asr_pybindings.asrRobotCalcInertiaSinglePybind(self.asr_robot, self.pivot_id_left, inertia)
                P = 5.0 # TODO 
                M = inertia.to_numpy()
                # Symmetrize and add small diagonal regularization to avoid singular/ill-conditioned cases
                M = 0.5 * (M + M.T) # TODO 
                eps = 1e-6
                ref_k = asr_pybindings.kMat.from_numpy(M * P + np.eye(M.shape[0]) * eps) # TODO 
                ref_d = asr_pybindings.kMat.from_numpy(M * (2.0*np.sqrt(P)) + np.eye(M.shape[0]) * (2.0*np.sqrt(P)*eps)) # TODO 
                asr_pybindings.asrRVCCalcDCCDoublePybind(
                    self.asr_robot, self.pivot_id_left, self.pivot_id_right, self.asr_task_double, ref_k, ref_d, 1e-6
                )
            
            if left and not right:
                asr_pybindings.asrRobotTaskFwdKinematicsPybind(self.asr_robot, self.asr_task_single)
                asr_pybindings.asrRobotCalcInertiaPybind(self.asr_robot)
                inertia = asr_pybindings.kMat(self.asr_robot.joint_num, self.asr_robot.joint_num)
                asr_pybindings.asrRobotCalcInertiaSinglePybind(self.asr_robot, self.pivot_id_left, inertia)
                P = 5.0
                M = inertia.to_numpy()
                M = 0.5 * (M + M.T)
                eps = 1e-6
                ref_k = asr_pybindings.kMat.from_numpy(M * P + np.eye(M.shape[0]) * eps)
                ref_d = asr_pybindings.kMat.from_numpy(M * (2.0*np.sqrt(P)) + np.eye(M.shape[0]) * (2.0*np.sqrt(P)*eps))
                asr_pybindings.asrRVCCalcDCCSinglePybind(
                    self.asr_robot, self.pivot_id_left, self.asr_task_single, ref_k, ref_d, 1e-6
                )

            if not left and right:
                asr_pybindings.asrRobotTaskFwdKinematicsPybind(self.asr_robot, self.asr_task_single)
                asr_pybindings.asrRobotCalcInertiaPybind(self.asr_robot)
                inertia = asr_pybindings.kMat(self.asr_robot.joint_num, self.asr_robot.joint_num)
                asr_pybindings.asrRobotCalcInertiaSinglePybind(self.asr_robot, self.pivot_id_right, inertia)
                P = 5.0
                M = inertia.to_numpy()
                M = 0.5 * (M + M.T)
                eps = 1e-6
                ref_k = asr_pybindings.kMat.from_numpy(M * P + np.eye(M.shape[0]) * eps)
                ref_d = asr_pybindings.kMat.from_numpy(M * (2.0*np.sqrt(P)) + np.eye(M.shape[0]) * (2.0*np.sqrt(P)*eps))
                asr_pybindings.asrRVCCalcDCCSinglePybind(
                    self.asr_robot, self.pivot_id_right, self.asr_task_single, ref_k, ref_d, 1e-6
                )
            if not left and not right:
                pass

            stiff_np = stiff_utils.get_reordered_joint_stiff(self.asr_robot, target_names=self.joint_names)
            visco_np = stiff_utils.get_reordered_joint_visco(self.asr_robot, target_names=self.joint_names)
            self.stiff[i] = torch.from_numpy(stiff_np).to(device=self.device, dtype=self.stiff.dtype)
            self.visco[i] = torch.from_numpy(visco_np).to(device=self.device, dtype=self.visco.dtype)

        pos_err = (q_des - q).unsqueeze(1)
        vel = qd.unsqueeze(1)
        tau = torch.bmm(pos_err, self.stiff).squeeze(1) - torch.bmm(vel, self.visco).squeeze(1)
        # Clip torques to [-80, 80] as requested
        tau = torch.clamp(tau, min=-80.0, max=80.0)

        self._asset.set_joint_effort_target(tau, joint_ids=self._joint_ids)

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
