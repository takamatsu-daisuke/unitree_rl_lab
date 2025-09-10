from typing import Dict
import math

def set_joint_angles_by_name(robot, joint_angles_dict: Dict[str, float]) -> None:
    """
    Set joint angles by name.

    Args:
        robot: Robot instance (e.g., asr_pybindings.Robot)
        joint_angles_dict: Dictionary mapping joint names to angles (in radians)

    Raises:
        ValueError: If a joint name is not found

    Example:
        >>> # robot = asr_pybindings.Robot("robot.asr")
        >>> # robot.asrRobotReadFilePybind("robot.asr")
        >>> set_joint_angles_by_name(robot, {
        ...     "right_shoulder": 0.5,
        ...     "right_elbow": 1.0,
        ...     "left_shoulder": -0.5,
        ...     "left_elbow": -1.0
        ... })
    """
    # Convert keys from Isaac naming (.._joint) to ASURA naming (.._link) if needed
    for joint_name, angle in joint_angles_dict.items():
        link_id = robot.asrRobotLinkFindByNamePybind(joint_name)
        if link_id == -1:
            raise ValueError(f"Joint/link '{joint_name}' not found in ASURA robot")
        robot.asrRobotLinkSetAngPybind(link_id, math.degrees(angle))  # ASURA uses degrees, convert from radians