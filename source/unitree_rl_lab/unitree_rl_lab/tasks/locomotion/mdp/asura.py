from typing import Dict, List

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
        robot.asrRobotLinkSetAngPybind(link_id, angle)


def _infer_joint_labels(robot, joint_count: int) -> List[str]:
    """Infer joint labels from the ASURA robot if possible."""
    labels = [f"joint_{i}" for i in range(joint_count)]

    link_num = getattr(robot, "link_num", None)
    get_joint_id = getattr(robot, "asrRobotLinkJointIDPybind", None)
    get_link_dof = getattr(robot, "asrRobotLinkDOFPybind", None)
    get_link_name = getattr(robot, "asrRobotLinkNamePybind", None)

    if not isinstance(link_num, int) or link_num <= 0:
        return labels
    if not callable(get_joint_id) or not callable(get_link_dof) or not callable(get_link_name):
        return labels

    for link_id in range(link_num):
        try:
            start_idx = get_joint_id(link_id)
            dof = get_link_dof(link_id)
            name = get_link_name(link_id)
        except Exception:
            return labels

        if start_idx is None or dof is None:
            continue
        if start_idx < 0 or dof <= 0:
            continue

        for offset in range(dof):
            joint_idx = start_idx + offset
            if joint_idx >= joint_count:
                break

            suffix = "" if offset == 0 else f"_{offset}"
            if name:
                labels[joint_idx] = f"{name}{suffix}"

    return labels


def print_joint_angles(robot) -> None:
    """Print all joint angles for the given ASURA robot instance."""
    try:
        angles = robot.asrRobotExportJointAngVecPybind()
    except AttributeError as exc:
        raise AttributeError("Robot instance does not expose joint angle export") from exc

    try:
        angle_values = angles.tolist()
    except AttributeError:
        try:
            angle_values = list(angles)
        except TypeError:
            print("joint angles (rad):", angles)
            return

    labels = _infer_joint_labels(robot, len(angle_values))

    for idx, angle in enumerate(angle_values):
        label = labels[idx]
        print(f"{label}: {angle:.6f} rad")
