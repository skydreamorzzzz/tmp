from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def _build_bone_indices(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    # HumanML3D 22 joints parent list
    parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
    child, par = [], []
    for i, p in enumerate(parents):
        if int(p) != -1:
            child.append(i)
            par.append(int(p))
    return (
        torch.tensor(child, device=device, dtype=torch.long),
        torch.tensor(par, device=device, dtype=torch.long),
    )


def segment_distance_batch(p1, q1, p2, q2):
    """
    向量化计算两组线段对之间的最短距离（广播友好）。
    线段1: p1 -> q1
    线段2: p2 -> q2
    输入张量应支持广播机制，例如 shape 为 (..., 3)。
    """
    # 形状约定（示例）：
    # p1,q1: (T,B,1,3), p2,q2: (T,1,B,3)
    # 广播后线段对为 (T,B,B)
    d1 = q1 - p1  # (T,B,1,3)
    d2 = q2 - p2  # (T,1,B,3)
    r = p1 - p2  # (T,B,B,3)

    a = torch.sum(d1 * d1, dim=-1)  # (T,B,1)
    e = torch.sum(d2 * d2, dim=-1)  # (T,1,B)
    b = torch.sum(d1 * d2, dim=-1)  # (T,B,B)
    c = torch.sum(d1 * r, dim=-1)  # (T,B,B)
    f = torch.sum(d2 * r, dim=-1)  # (T,B,B)

    # broadcast a,e to (T,B,B)
    a2 = a.expand_as(b)
    e2 = e.expand_as(b)

    epsilon = 1e-6
    denom = a2 * e2 - b * b  # (T,B,B)
    non_parallel = denom.abs() > epsilon

    # 无限直线最近点参数
    s = torch.where(non_parallel, (b * f - c * e2) / denom, torch.zeros_like(denom))
    t = torch.where(non_parallel, (a2 * f - b * c) / denom, torch.zeros_like(denom))

    # 平行近似：取 s=0, t=f/e
    parallel = ~non_parallel
    s = torch.where(parallel, torch.zeros_like(s), s)
    t = torch.where(parallel & (e2.abs() > epsilon), f / (e2 + 1e-8), t)

    # clamp 到线段
    s = torch.clamp(s, 0.0, 1.0)
    t = torch.where(e2.abs() > epsilon, (b * s + f) / e2, torch.zeros_like(s))
    t = torch.clamp(t, 0.0, 1.0)

    s = torch.where(a2.abs() > epsilon, (b * t - c) / a2, torch.zeros_like(t))
    s = torch.clamp(s, 0.0, 1.0)

    c1 = p1 + s.unsqueeze(-1) * d1
    c2 = p2 + t.unsqueeze(-1) * d2
    return torch.norm(c1 - c2, dim=-1)


def _compute_penebone_for_pairs(
    bone_starts: torch.Tensor,
    bone_ends: torch.Tensor,
    pairs: list,
    threshold: float,
    device: torch.device,
) -> Dict[str, float]:
    """
    计算指定人物对的 PeneBone 指标
    
    根据论文定义：PeneBone = sum over all collision depths for each sample
    即对每个人物对，计算所有帧、所有骨骼对的穿透深度总和
    """
    if not pairs:
        return {"penebone": 0.0, "rate": float("nan"), "depth_sum": 0.0, "count": 0.0, "total": 0.0}
    
    sum_depth = torch.tensor(0.0, device=device)
    count_pen = torch.tensor(0.0, device=device)
    total = 0
    
    for (i, j) in pairs:
        p1_s = bone_starts[i].unsqueeze(2)  # (T,B,1,3)
        p1_e = bone_ends[i].unsqueeze(2)
        p2_s = bone_starts[j].unsqueeze(1)  # (T,1,B,3)
        p2_e = bone_ends[j].unsqueeze(1)
        
        dist = segment_distance_batch(p1_s, p1_e, p2_s, p2_e)  # (T,B,B)
        pen = torch.clamp(threshold - dist, min=0.0)
        sum_depth = sum_depth + pen.sum()
        count_pen = count_pen + (pen > 0).sum()
        total += int(pen.numel())
    
    if total == 0:
        return {"penebone": 0.0, "rate": float("nan"), "depth_sum": 0.0, "count": 0.0, "total": 0.0}
    
    # PeneBone = 穿透深度总和 (论文定义)
    # Penetration Rate = 穿透的 bone-bone 对数 / 总骨骼对数
    return {
        "penebone": float(sum_depth.item()),  # 总和，不是平均
        "rate": float((count_pen / float(total)).item()),
        "depth_sum": float(sum_depth.item()),
        "count": float(count_pen.item()),
        "total": float(total),
    }


def compute_penebone_metrics(
    joints: np.ndarray | torch.Tensor,
    *,
    joint_radius: float = 0.02,
    device: str = "cpu",
    interaction_pairs: list = None,
    non_interaction_pairs: list = None,
) -> Dict[str, float]:
    """
    返回 PeneBone 指标，分为交互对和非交互对

    Args:
    joints: (N,T,22,3)
        joint_radius: 骨骼半径
        device: 计算设备
        interaction_pairs: 交互人物对列表 [(i,j), ...]
        non_interaction_pairs: 非交互人物对列表 [(i,j), ...]
    
    Returns:
        - pene_depth_mean: 平均穿透深度（所有对）
        - pene_rate: 穿透率（所有对）
        - pene_inter_*: 交互对的指标
        - pene_non_*: 非交互对的指标
    """
    if isinstance(joints, np.ndarray):
        x = torch.from_numpy(joints).to(device=device, dtype=torch.float32)
    else:
        x = joints.to(device=device, dtype=torch.float32)
    assert x.ndim == 4 and x.shape[-2:] == (22, 3), f"expected (N,T,22,3), got {tuple(x.shape)}"

    N, T = int(x.shape[0]), int(x.shape[1])
    dev = x.device

    child_idx, parent_idx = _build_bone_indices(dev)
    bone_starts = x[:, :, parent_idx, :]
    bone_ends = x[:, :, child_idx, :]
    B = int(bone_starts.shape[2])
    threshold = float(2.0 * joint_radius)
    
    # 如果没有提供分类，使用所有对
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    if interaction_pairs is None and non_interaction_pairs is None:
        # 旧的行为：计算所有对
        interaction_pairs = all_pairs
        non_interaction_pairs = []
    
    # 计算交互对的 PeneBone
    inter_metrics = _compute_penebone_for_pairs(
        bone_starts, bone_ends, interaction_pairs or [], threshold, dev
    )
    
    # 计算非交互对的 PeneBone
    non_metrics = _compute_penebone_for_pairs(
        bone_starts, bone_ends, non_interaction_pairs or [], threshold, dev
    )
    
    # 合并计算所有对的总体指标
    total_depth_sum = inter_metrics["depth_sum"] + non_metrics["depth_sum"]
    total_count = inter_metrics["count"] + non_metrics["count"]
    total_pairs = inter_metrics["total"] + non_metrics["total"]
    
    if total_pairs == 0:
        pene_rate = float("nan")
    else:
        pene_rate = total_count / total_pairs
    
    return {
        # 总体指标 (论文定义: PeneBone = 穿透深度总和)
        "penebone": float(total_depth_sum),  # 总和 (m)
        "pene_rate": float(pene_rate),       # 穿透率
        "pene_count": float(total_count),
        "pene_total_pairs": float(total_pairs),
        "pene_num_bones": float(B),
        # 交互对指标
        "penebone_inter": float(inter_metrics["penebone"]),
        "pene_inter_rate": float(inter_metrics["rate"]),
        # 非交互对指标
        "penebone_non": float(non_metrics["penebone"]),
        "pene_non_rate": float(non_metrics["rate"]),
    }


def compute_penebone_for_all_pairs(
    joints: np.ndarray | torch.Tensor,
    all_pairs: list,
    *,
    joint_radius: float = 0.02,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    简化版本：计算所有人员对的 PeneBone，不区分交互关系
    
    Args:
        joints: (N,T,22,3)
        all_pairs: 所有人物对列表 [(i,j), ...]
        joint_radius: 骨骼半径
        device: 计算设备
    
    Returns:
        - pene_depth_sum: 穿透深度总和 (论文定义)
        - pene_rate: 穿透率
    """
    if isinstance(joints, np.ndarray):
        x = torch.from_numpy(joints).to(device=device, dtype=torch.float32)
    else:
        x = joints.to(device=device, dtype=torch.float32)
    
    if x.ndim != 4 or x.shape[-2:] != (22, 3):
        raise ValueError(f"expected (N,T,22,3), got {tuple(x.shape)}")

    dev = x.device
    child_idx, parent_idx = _build_bone_indices(dev)
    bone_starts = x[:, :, parent_idx, :]
    bone_ends = x[:, :, child_idx, :]
    threshold = float(2.0 * joint_radius)
    
    metrics = _compute_penebone_for_pairs(
        bone_starts, bone_ends, all_pairs, threshold, dev
    )
    
    return {
        "pene_depth_sum": float(metrics["depth_sum"]),
        "pene_rate": float(metrics["rate"]),
    }


