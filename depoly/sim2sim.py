import time
import os
import pathlib
import argparse

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
from sim2sim_helper import parse_training_env_yaml, indices_mapping_lab_to_mjc

# Get the absolute path of the project root directory
SIM2SIM_ROOT = pathlib.Path(__file__).parent.absolute()

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def limit_effort(effort, effort_limit):
    """Limit the effort to the specified limit."""
    return np.clip(effort, -effort_limit, effort_limit)


def get_sorted_mjc_joint_names(m):
    """Get sorted list of MuJoCo joint names, excluding the floating base."""
    mjc_joint_names = []
    for i in range(m.njnt):
        # Skip floating base
        if i == 0:
            continue
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        mjc_joint_names.append(joint_name)
    return mjc_joint_names


def load_config(config_file):
    """Load and parse the configuration file."""
    file_path = os.path.join(SIM2SIM_ROOT, config_file)
    with open(file_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_scale_values(observations_info, actions_info):
    """Get scale values from observations and actions info, defaulting to 1.0 if null."""
    scales = {
        'ang_vel_scale': observations_info["base_ang_vel"]["scale"],
        'dof_pos_scale': observations_info["joint_pos"]["scale"],
        'dof_vel_scale': observations_info["joint_vel"]["scale"],
        'action_scale': actions_info["joint_pos"]["scale"]
    }
    return {k: 1.0 if v is None else v for k, v in scales.items()}


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, default="g1_lab.yaml", help="config file name in the config folder"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_file)
    policy_path = config["policy_path"]
    xml_path = config["xml_path"]
    ordered_joints_lab = config["ordered_joints_lab"]
    training_env_yaml = config["env_path"]
    simulation_duration = config["simulation_duration"]
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    cmd = np.array(config["cmd_init"], dtype=np.float32)

    # Parse training environment yaml
    actuators_info, joint_pos_info, sim, observations_info, actions_info = parse_training_env_yaml(
        os.path.join(SIM2SIM_ROOT, training_env_yaml)
    )
    control_decimation = int(sim["decimation"])

    # Load robot model
    m = mujoco.MjModel.from_xml_path(os.path.join(SIM2SIM_ROOT, xml_path))
    d = mujoco.MjData(m)
    mjc_joint_names = get_sorted_mjc_joint_names(m)

    # Get scale values
    scales = get_scale_values(observations_info, actions_info)
    ang_vel_scale = scales['ang_vel_scale']
    dof_pos_scale = scales['dof_pos_scale']
    dof_vel_scale = scales['dof_vel_scale']
    action_scale = scales['action_scale']

    # Get joint indices and parameters
    joint_ids, kps, kds, effort_limits, default_angles, armatures = indices_mapping_lab_to_mjc(
        ordered_joints_lab, mjc_joint_names, actuators_info, joint_pos_info
    )

    # Initialize robot state
    target_dof_pos = default_angles.copy()
    m.opt.timestep = float(sim["dt"])
    m.dof_armature[6:] = armatures
    d.qpos[7:] = target_dof_pos
    mujoco.mj_forward(m, d)

    # Load policy
    policy = torch.jit.load(os.path.join(SIM2SIM_ROOT, policy_path))

    # Initialize context variables
    action_lab = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    # Run simulation
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # Update viewer
            viewer.sync()

            # Compute control
            tau = kps * (target_dof_pos - d.qpos[7:]) + kds * (-d.qvel[6:])
            tau = limit_effort(tau, effort_limits)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            counter += 1

            if counter % control_decimation == 0:

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]
                base_vel = d.qvel[:3]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                obs[:3] = base_vel
                obs[3:6] = omega
                obs[6:9] = gravity_orientation
                obs[9:12] = cmd 
                # joint pos
                obs[12 : 12 + num_actions] = qj[joint_ids]
                # joint vel
                obs[12 + num_actions : 12 + 2 * num_actions] = dqj[joint_ids]
                #  actions
                obs[12 + 2 * num_actions : 12 + 3 * num_actions] = action_lab

                # Policy inference
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action_lab = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos_lab = action_lab * action_scale + default_angles[joint_ids]
                target_dof_pos[joint_ids] = target_dof_pos_lab

            # Rudimentary time keeping
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
