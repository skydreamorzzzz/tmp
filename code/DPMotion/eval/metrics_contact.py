"""
基于关节位置的 CFrame / Contact 指标。

与 mesh 版本（metrics_cframe_mesh.py 需要 SMPL 6890 顶点）不同，
本模块直接使用 22 关节 3D 坐标，以关节球体半径近似碰撞检测。

指标定义：
  - CFrame (%):  在所有帧中，存在任意跨人关节对距离 < threshold 的帧占比
  - Contact:     平均每帧的接触关节对数量
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def joint_contact_metrics(
    joints: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    *,
    contact_threshold: float = 0.10,
) -> Dict[str, float]:
    """
    基于关节距离的 CFrame 和 Contact。

    Args:
        joints: (N, T, 22, 3)
        pairs: 人物对列表 [(i, j), ...]
        contact_threshold: 两关节中心距 < 此值视为接触 (米)。
            22 关节无 mesh 包络，默认 0.10m (10cm) 比 mesh 版 0.005m 宽松。

    Returns:
        cframe:  接触帧占比 (%)
        contact: 平均每帧接触关节对数
    """
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 4 or joints.shape[2] != 22 or joints.shape[3] != 3:
        raise ValueError(f"expected (N,T,22,3), got {joints.shape}")
    if not pairs:
        return {"cframe": float("nan"), "contact": float("nan")}

    T = int(joints.shape[1])
    J = int(joints.shape[2])

    contact_per_frame = np.zeros(T, dtype=np.int64)

    for (pi, pj) in pairs:
        ji = joints[pi]  # (T, 22, 3)
        jj = joints[pj]  # (T, 22, 3)
        diff = ji[:, :, None, :] - jj[:, None, :, :]  # (T, 22, 22, 3)
        dist = np.linalg.norm(diff, axis=-1)           # (T, 22, 22)
        in_contact = dist < contact_threshold           # (T, 22, 22)
        contact_per_frame += in_contact.sum(axis=(1, 2))

    frames_with_contact = int((contact_per_frame > 0).sum())
    cframe_pct = frames_with_contact / float(T) * 100.0 if T > 0 else 0.0
    contact_mean = float(contact_per_frame.mean()) if T > 0 else 0.0

    return {
        "cframe": float(cframe_pct),
        "contact": float(contact_mean),
    }


def compute_contact_metrics(
    joints: np.ndarray,
    *,
    interaction_pairs: Sequence[Tuple[int, int]] = None,
    non_interaction_pairs: Sequence[Tuple[int, int]] = None,
    contact_threshold: float = 0.10,
) -> Dict[str, float]:
    """
    分交互对 / 非交互对计算 CFrame 和 Contact。

    Args:
        joints: (N, T, 22, 3)
        interaction_pairs: 交互人物对
        non_interaction_pairs: 非交互人物对
        contact_threshold: 接触阈值 (米)

    Returns:
        cframe_all, contact_all:     全部对
        cframe_inter, contact_inter: 交互对
        cframe_non, contact_non:     非交互对
    """
    N = int(joints.shape[0])
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    if interaction_pairs is None and non_interaction_pairs is None:
        interaction_pairs = all_pairs
        non_interaction_pairs = []

    m_all = joint_contact_metrics(joints, all_pairs, contact_threshold=contact_threshold)
    m_inter = joint_contact_metrics(
        joints, interaction_pairs or [], contact_threshold=contact_threshold
    )
    m_non = joint_contact_metrics(
        joints, non_interaction_pairs or [], contact_threshold=contact_threshold
    )

    return {
        "cframe_all": m_all["cframe"],
        "contact_all": m_all["contact"],
        "cframe_inter": m_inter["cframe"],
        "contact_inter": m_inter["contact"],
        "cframe_non": m_non["cframe"],
        "contact_non": m_non["contact"],
    }
