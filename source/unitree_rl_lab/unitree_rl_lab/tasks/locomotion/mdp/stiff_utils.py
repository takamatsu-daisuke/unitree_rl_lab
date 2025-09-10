"""
Utilities for reordering joint-space stiffness/viscosity matrices
based on joint-name orderings.

Typical usage
-------------
- You have a native joint stiffness/viscosity matrix `K_native` (NxN) and want it in
  your desired joint-name order `names_target`. Given the current order of
  names for `K_native` as `names_current`, call:

    >>> K_reordered = reorder_joint_matrix(K_native, names_current, target_names=names_target)

Notes
-----
- Reordering is done as K' = P K P^T where P is a permutation matrix implied
  by the `perm` created from names.
- This returns a new numpy array; it does not modify the input in-place.
"""

from typing import List, Optional
import numpy as np


class ReorderError(ValueError):
    """Raised when reordering inputs are inconsistent or incomplete."""


def _check_square_matrix(K: np.ndarray) -> None:
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ReorderError(f"Expected a square matrix, got shape {K.shape}")

def _check_name_list(names: List[str], n: int, label: str) -> None:
    if len(names) != n:
        raise ReorderError(f"{label} length {len(names)} does not match matrix size {n}")
    if len(set(names)) != len(names):
        raise ReorderError(f"{label} contains duplicate names")

def build_perm_from_lists(current_names: List[str], target_names: List[str], *, strict: bool = True) -> np.ndarray:
    """
    Build a permutation vector such that rows/cols of a matrix in `current_names`
    order become `target_names` order by indexing with perm (i.e., K' = K[perm][:, perm]).

    Args:
        current_names: Joint names in the current/native order.
        target_names: Joint names in the desired order.
        strict: If True, require exact name match between the lists.

    Returns:
        perm: numpy int array of indices mapping target -> current.
    """
    n = len(current_names)
    _check_name_list(current_names, n, "current_names")
    _check_name_list(target_names, n, "target_names")

    name_to_idx = {name: i for i, name in enumerate(current_names)}

    try:
        perm = np.array([name_to_idx[name] for name in target_names], dtype=int)
    except KeyError as e:
        raise ReorderError(f"Name '{e.args[0]}' from target_names not found in current_names") from None

    if strict:
        # Ensure it's a proper permutation of [0..n-1]
        if len(set(perm.tolist())) != n:
            raise ReorderError("target_names produce a non-bijective permutation")

    return perm

def reorder_joint_matrix(
    K: np.ndarray,
    current_names: List[str],
    *,
    target_names: Optional[List[str]] = None,
    strict: bool = True,
    copy: bool = True,
) -> np.ndarray:
    """
    Reorder a joint-space square matrix to a desired joint-name order.

    Computes K' = P K P^T where P is the permutation from names.
    Returns a new numpy array; the input is not modified.

    Args:
        K: Square (N x N) matrix in the current/native joint order.
        current_names: Names in the current/native joint order (length N).
        target_names: Names in the desired order (length N).
        strict: If True, require a full bijection (no missing/extra names).
        copy: If True, ensure the returned array is a new array with own data.

    Returns:
        K' reordered to the target order as a new numpy array.
    """
    _check_square_matrix(K)
    n = K.shape[0]
    _check_name_list(current_names, n, "current_names")

    if target_names is None:
        raise ReorderError("target_names must be provided")
    perm = build_perm_from_lists(current_names, target_names, strict=strict)

    K_np = np.array(K, copy=False)
    K_reordered = K_np[np.ix_(perm, perm)]
    if copy:
        K_reordered = np.array(K_reordered, copy=True)
    return K_reordered

def _axis_suffixes_for_type(joint_type: int) -> List[str]:
    """Return axis suffixes for a joint type.

    Policy: Only Revolute joints produce a name entry.
    - Revolute: return [""] (single DOF; no axis suffix appended)
    - Fix (0-DOF): return [] to indicate it should be skipped
    - Others: raise ReorderError (unsupported under suffix-less policy)
    """
    # Keep numbers in sync with asr_link.h enum
    ASR_JOINT_FIX = 0
    ASR_JOINT_REVOL = 1
    ASR_JOINT_PRISM = 2
    ASR_JOINT_SCREW = 3
    ASR_JOINT_SPHERE = 4
    ASR_JOINT_FREE2D = 5
    ASR_JOINT_FREE = 6

    names = {
        ASR_JOINT_FIX: "FIX",
        ASR_JOINT_REVOL: "REVOL",
        ASR_JOINT_PRISM: "PRISM",
        ASR_JOINT_SCREW: "SCREW",
        ASR_JOINT_SPHERE: "SPHERE",
        ASR_JOINT_FREE2D: "FREE2D",
        ASR_JOINT_FREE: "FREE",
    }

    if joint_type == ASR_JOINT_REVOL:
        # No axis suffix for revolute (single DOF)
        return [""]
    if joint_type == ASR_JOINT_FIX or joint_type == ASR_JOINT_FREE:
        # Zero DOF joint; exclude from naming
        return []

    # All non-revolute joints are not supported under suffix-less policy
    jt_name = names.get(joint_type, f"UNKNOWN({joint_type})")
    raise ReorderError(
        f"Non-revolute/non-fix joint type not supported for name generation: {jt_name} ({joint_type}). "
        "Axis suffixes are disabled."
    )

def get_current_joint_order_names(robot, *, with_axes: bool = True) -> List[str]:
    """
    Construct the current joint-order name list that matches `robot._joint_stiff`.

    This uses link joint IDs, DOFs, and types to place names in the correct
    indices (subtracting the base offset). Only Revolute joints are named;
    FIX (0-DOF) joints are skipped. Axis suffixes are not added (a revolute
    joint yields just the link name).

    Returns:
        List[str] of length `robot.joint_num`, where index i corresponds to
        row/col i of `_joint_stiff`.
    """
    n = getattr(robot, "joint_num")
    link_n = getattr(robot, "link_num")
    names: List[str] = [None] * n  # type: ignore

    # Base offset equals the first non-root link's joint ID in generalized coords
    # Fallback: compute the minimum joint ID among non-root links
    try:
        base_offset = robot.asrRobotLinkJointIDPybind(1)
    except Exception:
        base_offset = None

    if base_offset is None or base_offset < 0:
        ids = []
        for lid in range(1, link_n):
            try:
                jid = robot.asrRobotLinkJointIDPybind(lid)
                if jid >= 0:
                    ids.append(jid)
            except Exception:
                continue
        if not ids:
            raise ReorderError("Failed to determine base joint ID offset from robot")
        base_offset = min(ids)

    for lid in range(1, link_n):
        name = robot.asrRobotLinkNamePybind(lid)
        jtype = robot.asrRobotLinkJointTypePybind(lid)
        start = robot.asrRobotLinkJointIDPybind(lid) - base_offset
        suffixes = _axis_suffixes_for_type(jtype)
        if not suffixes:
            continue  # 0-DOF (FIX) or unknown
        for d, sfx in enumerate(suffixes):
            idx = start + d
            if idx < 0 or idx >= n:
                raise ReorderError(f"Index out of bounds while mapping names: {idx} not in [0,{n})")
            label = name if (not with_axes or sfx == "") else f"{name}:{sfx}"
            if names[idx] is not None:
                raise ReorderError(f"Duplicate assignment to index {idx} for '{label}'")
            names[idx] = label

    # Final check and fill if anything missing
    if any(v is None for v in names):
        missing = [i for i, v in enumerate(names) if v is None]
        raise ReorderError(f"Could not infer names for indices: {missing}")

    return names  # type: ignore

def get_reordered_joint_matrix(
    robot,
    *,
    target_names: Optional[List[str]] = None,
    strict: bool = True,
    copy: bool = True,
    kind: str = "stiff",
):
    """
    Generic helper to fetch a robot's joint matrix (`_joint_stiff` or `_joint_visco`)
    and return it reordered to the specified joint-name order.

    Args:
        robot: Robot instance exposing `_joint_stiff`/`_joint_visco` (NumPy-viewable kMat).
        current_names: Inferred internally from the robot's link structure.
        target_names: Names in the desired order (length N).
        strict: Enforce bijection and completeness of mapping.
        copy: Return a new numpy array detached from underlying storage.
        kind: One of {"stiff", "visco"}.

    Returns:
        Reordered matrix as numpy.ndarray.

    Raises:
        ReorderError: If the selected matrix is None, or mapping invalid.
    """
    attr = "_joint_stiff" if kind == "stiff" else "_joint_visco" if kind == "visco" else None
    if attr is None:
        raise ReorderError(f"Unknown joint matrix kind: {kind}")

    K_obj = getattr(robot, attr, None)
    if K_obj is None:
        what = "stiffness" if attr == "_joint_stiff" else "viscosity"
        raise ReorderError(
            f"robot.{attr} is None. Compute or initialize {what} first "
            "(e.g., call robot._init_joint_matrices())."
        )

    K_native = np.array(K_obj, copy=True)
    current_names = get_current_joint_order_names(robot)
    return reorder_joint_matrix(
        K_native,
        current_names,
        target_names=target_names,
        strict=strict,
        copy=copy,
    )

def get_reordered_joint_stiff(
    robot,
    *,
    target_names: Optional[List[str]] = None,
    strict: bool = True,
    copy: bool = True,
):
    """
    Convenience wrapper for stiffness matrix reordering.

    Equivalent to `get_reordered_joint_matrix(..., kind='stiff')`.
    """
    return get_reordered_joint_matrix(
        robot,
        target_names=target_names,
        strict=strict,
        copy=copy,
        kind="stiff",
    )

def get_reordered_joint_visco(
    robot,
    *,
    target_names: Optional[List[str]] = None,
    strict: bool = True,
    copy: bool = True,
):
    """
    Convenience wrapper for viscosity matrix reordering.

    Equivalent to `get_reordered_joint_matrix(..., kind='visco')`.
    """
    return get_reordered_joint_matrix(
        robot,
        target_names=target_names,
        strict=strict,
        copy=copy,
        kind="visco",
    )
