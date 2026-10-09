from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def root_positions(joints: np.ndarray) -> np.ndarray:
    """
    joints: (N,T,22,3)
    returns: (N,T,3) 以 joint0 作为 pelvis/root
    """
    return joints[:, :, 0, :]


def mean_root_distance(root: np.ndarray, pairs: Sequence[Tuple[int, int]]) -> float:
    """
    root: (N,T,3)
    对给定 pair 集合，计算所有帧的 root 3D 距离均值。
    """
    if not pairs:
        return float("nan")
    idx_i = np.array([p[0] for p in pairs], dtype=np.int64)
    idx_j = np.array([p[1] for p in pairs], dtype=np.int64)
    ri = root[idx_i]  # (P,T,3)
    rj = root[idx_j]
    d = np.linalg.norm(ri - rj, axis=-1)  # (P,T)
    return float(d.mean())


def hip_distance_density_metrics(root: np.ndarray, group_pairs, person_pairs, non_pairs) -> Dict[str, float]:
    return {
        "hip_dist_group_mean": mean_root_distance(root, sorted(group_pairs)),
        "hip_dist_person_mean": mean_root_distance(root, sorted(person_pairs)),
        "hip_dist_non_mean": mean_root_distance(root, sorted(non_pairs)),
    }


def facing_directions(joints: np.ndarray) -> np.ndarray:
    """
    用肩膀与躯干估计人物朝向（forward）。

    joints: (N,T,22,3)
    returns: (N,T,3) 单位向量
    """
    # HumanML3D 22 joints: 16 left_shoulder, 17 right_shoulder, 0 pelvis, 12 neck
    left_sh = joints[:, :, 16, :]
    right_sh = joints[:, :, 17, :]
    pelvis = joints[:, :, 0, :]
    neck = joints[:, :, 12, :]

    lr = right_sh - left_sh  # left->right
    up = neck - pelvis
    lr_u = _safe_normalize(lr)
    up_u = _safe_normalize(up)
    fwd = np.cross(lr_u, up_u)  # right x up = forward (sign depends on coord, but相对一致)
    fwd = _safe_normalize(fwd)

    # fallback: 如果肩膀不可用，使用髋部
    bad = np.linalg.norm(fwd, axis=-1) < 1e-6
    if np.any(bad):
        lhip = joints[:, :, 1, :]
        rhip = joints[:, :, 2, :]
        lr2 = rhip - lhip
        lr2_u = _safe_normalize(lr2)
        fwd2 = _safe_normalize(np.cross(lr2_u, up_u))
        fwd[bad] = fwd2[bad]
    return fwd


def person_facing_orientation_score(
    joints: np.ndarray,
    person_links: Sequence[Tuple[int, str, Sequence[int]]],
) -> float:
    """
    Person 交互：person 是否朝向目标 group（面向 group centroid）。

    返回：平均朝向分数，范围 [0,1]（取 max(cos,0) 并对时间/交互求均值）。
    """
    if not person_links:
        return float("nan")
    root = root_positions(joints)  # (N,T,3)
    fwd = facing_directions(joints)  # (N,T,3)

    scores = []
    for p_idx, _gid, members in person_links:
        mem = list(members)
        if len(mem) == 0:
            continue
        target = root[mem].mean(axis=0)  # (T,3)
        to_tgt = _safe_normalize(target - root[p_idx])  # (T,3)
        cos = (fwd[p_idx] * to_tgt).sum(axis=-1)  # (T,)
        cos = np.clip(cos, 0.0, 1.0)
        scores.append(float(cos.mean()))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def motion_embedding_from_joints(joints: np.ndarray) -> np.ndarray:
    """
    给每个人构造一个轻量的“motion embedding”（不依赖外部评测网络），用于计算 people-level diversity。
    joints: (N,T,22,3)
    returns: (N,D)
    """
    N, T = joints.shape[0], joints.shape[1]
    root = root_positions(joints)  # (N,T,3)
    xz = root[:, :, [0, 2]]
    dxz = np.diff(xz, axis=1)  # (N,T-1,2)
    speed = np.linalg.norm(dxz, axis=-1)  # (N,T-1)
    path_len = speed.sum(axis=1)  # (N,)

    # joint velocity energy（选一些关键关节）
    key = [7, 8, 10, 11, 15, 20, 21]  # ankles/feet/head/wrists
    J = joints[:, :, key, :]  # (N,T,K,3)
    v = np.diff(J, axis=1)
    vmag = np.linalg.norm(v, axis=-1)  # (N,T-1,K)
    vmean = vmag.mean(axis=1)  # (N,K)

    # 统计量拼接
    emb = np.concatenate(
        [
            xz.mean(axis=1),  # (N,2)
            xz.std(axis=1),  # (N,2)
            path_len[:, None],  # (N,1)
            speed.mean(axis=1)[:, None],
            speed.std(axis=1)[:, None],
            vmean,  # (N,K)
            root[:, :, 1].mean(axis=1)[:, None],  # mean height
        ],
        axis=1,
    ).astype(np.float32)
    return emb


def diversity_motion_embedding(joints: np.ndarray) -> float:
    """
    所有人 motion embedding 的两两欧氏距离均值（N<=7 时直接全对计算）。
    """
    emb = motion_embedding_from_joints(joints)  # (N,D)
    N = emb.shape[0]
    if N < 2:
        return float("nan")
    dists = []
    for i in range(N):
        for j in range(i + 1, N):
            dists.append(float(np.linalg.norm(emb[i] - emb[j])))
    return float(np.mean(dists)) if dists else float("nan")


def interaction_diversity_variance(
    joints: np.ndarray,
    group_pairs: Sequence[Tuple[int, int]],
    person_links: Sequence[Tuple[int, str, Sequence[int]]],
    group_cframe: Optional[float] = None,
    person_orientation: Optional[float] = None,
) -> float:
    """
    交互多样性：对每个 interaction 取一个特征向量，再计算跨 interaction 的特征方差（trace of var）。

    这里的特征是“可解释”的几何/运动统计：
      [mean_dist, std_dist, mean_rel_speed, contact_rate, facing_score]

    - Group interaction: facing_score=0
    - Person interaction: contact_rate=0
    """
    root = root_positions(joints)  # (N,T,3)
    # root velocity
    v = np.diff(root, axis=1)  # (N,T-1,3)

    feats = []

    # Group interactions（按 pair）
    for (a, b) in group_pairs:
        d = np.linalg.norm(root[a] - root[b], axis=-1)  # (T,)
        va = v[a]
        vb = v[b]
        rel_speed = np.linalg.norm(va - vb, axis=-1)  # (T-1,)
        contact_rate = float(group_cframe) if group_cframe is not None else 0.0
        feats.append(
            [
                float(d.mean()),
                float(d.std()),
                float(rel_speed.mean()) if rel_speed.size > 0 else 0.0,
                contact_rate,
                0.0,
            ]
        )

    # Person interactions（person vs group centroid）
    fwd = facing_directions(joints)  # (N,T,3)
    for p_idx, _gid, members in person_links:
        mem = list(members)
        if not mem:
            continue
        tgt = root[mem].mean(axis=0)  # (T,3)
        d = np.linalg.norm(root[p_idx] - tgt, axis=-1)  # (T,)
        vtgt = np.diff(tgt, axis=0)  # (T-1,3)
        vp = v[p_idx]
        rel_speed = np.linalg.norm(vp - vtgt, axis=-1)  # (T-1,)
        to_tgt = _safe_normalize(tgt - root[p_idx])  # (T,3)
        cos = (fwd[p_idx] * to_tgt).sum(axis=-1)
        cos = np.clip(cos, 0.0, 1.0)
        facing_score = float(cos.mean())
        feats.append(
            [
                float(d.mean()),
                float(d.std()),
                float(rel_speed.mean()) if rel_speed.size > 0 else 0.0,
                0.0,
                facing_score,
            ]
        )

    if len(feats) < 2:
        return 0.0
    F = np.asarray(feats, dtype=np.float32)  # (K,D)
    var = F.var(axis=0)  # (D,)
    return float(var.sum())


