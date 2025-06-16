import os
import pathlib
import re
from typing import Dict, Any, Tuple, Union

import numpy as np
import yaml


def get_matching_value(param_dict: Union[Dict[str, float], float], joint_name: str, default_value: float = 0.0) -> float:
    """
    Helper function to get the first matching value from a parameter dictionary.
    
    Args:
        param_dict: Dictionary mapping joint names/patterns to values, or a single float value
        joint_name: Name of the joint to find a match for
        default_value: Default value to return if no match is found
        
    Returns:
        float: The matching value or default value
    """
    if not isinstance(param_dict, dict):
        return float(param_dict)
        
    # First try exact match
    if joint_name in param_dict:
        return float(param_dict[joint_name])
        
    # Then try pattern matching
    for pattern, value in param_dict.items():
        if re.match(pattern, joint_name):
            return float(value)
            
    return default_value


def parse_training_env_yaml(file_path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Parse joint information from env.yaml file.
    
    Args:
        file_path: Path to the env.yaml file
        
    Returns:
        Tuple containing:
        - Dictionary with joint names as keys and actuator info as values
        - Dictionary of joint positions
        - Simulation parameters
        - Observation parameters
        - Action parameters
    """
    # Get absolute path
    file_path = os.path.join(pathlib.Path(__file__).parent.absolute(), file_path)
    
    with open(file_path, 'r') as f:
        config = yaml.unsafe_load(f)
    
    # Extract configuration sections
    joint_pos_info = config['scene']['robot']['init_state']['joint_pos']
    actuators = config['scene']['robot']['actuators']
    sim = {
        'dt': config['sim']['dt'],
        'decimation': config['decimation'],
    }
    observations_info = config['observations']['policy']
    actions_info = config['actions']
    
    # Process actuators
    actuators_info = {}
    for actuator_name, actuator_config in actuators.items():
        joint_patterns = actuator_config.get('joint_names_expr', [])
        effort_limit = actuator_config.get('effort_limit', 0.0)
        stiffness = actuator_config.get('stiffness', {})
        damping = actuator_config.get('damping', {})
        armature = actuator_config.get('armature', {})
        
        for pattern in joint_patterns:
            joint_name = pattern
            if True:  # TODO: Remove this condition if not needed
                if re.match(pattern, joint_name):
                    actuator_info = {
                        'name': joint_name,
                        'stiffness': get_matching_value(stiffness, joint_name),
                        'damping': get_matching_value(damping, joint_name),
                        'armature': get_matching_value(armature, joint_name),
                        'effort_limits': get_matching_value(effort_limit, joint_name),
                        'position_limits': [-float('inf'), float('inf')],
                    }
                    actuators_info[joint_name] = actuator_info
    
    return actuators_info, joint_pos_info, sim, observations_info, actions_info


def indices_mapping_lab_to_mjc(
    ordered_joints_lab: list,
    joint_names_mjc: list,
    actuators: Dict[str, Dict[str, Any]],
    joint_pos: Dict[str, float]
) -> Tuple[list, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create mapping between lab and mujoco joint indices and extract joint parameters.
    
    Args:
        ordered_joints_lab: List of joint names in lab order
        joint_names_mjc: List of joint names in mujoco order
        actuators: Dictionary of actuator configurations
        joint_pos: Dictionary of joint positions
        
    Returns:
        Tuple containing:
        - List of joint indices
        - Array of stiffness values
        - Array of damping values
        - Array of effort limits
        - Array of default angles
        - Array of armature values
    """
    assert len(ordered_joints_lab) == len(joint_names_mjc), \
        f"Length of ordered_joints_lab: {len(ordered_joints_lab)} != Length of joint_names_mjc: {len(joint_names_mjc)}"
    
    joint_indices = []
    kds = np.zeros(len(ordered_joints_lab))
    kps = np.zeros(len(ordered_joints_lab))
    default_angles = np.zeros(len(ordered_joints_lab))
    effort_limits = np.zeros(len(ordered_joints_lab))
    armatures = np.zeros(len(ordered_joints_lab))
    
    for joint_name in ordered_joints_lab:
        # Find matching pattern in actuators
        matching_pattern = None
        for pattern in actuators.keys():
            if re.match(pattern, joint_name):
                matching_pattern = pattern
                break
        
        if matching_pattern is None:
            raise ValueError(f"No matching pattern found for joint '{joint_name}' in environment configuration")
            
        joint_info = actuators[matching_pattern]
        mujoco_index = joint_names_mjc.index(joint_name)
        joint_indices.append(mujoco_index)
        
        # Set joint parameters
        kds[mujoco_index] = joint_info["damping"]
        kps[mujoco_index] = joint_info["stiffness"]
        effort_limits[mujoco_index] = joint_info["effort_limits"]
        armatures[mujoco_index] = joint_info["armature"]
        default_angles[mujoco_index] = 0.0
        
        # Find matching joint position
        for pattern in joint_pos.keys():
            if re.match(pattern, joint_name):
                default_angles[mujoco_index] = joint_pos[pattern]
                break
        
    return joint_indices, kps, kds, effort_limits, default_angles, armatures
