#!/usr/bin/env python3
"""Summarize matched guidance checks using the repository metric functions."""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DP = ROOT / "code/DPMotion"
sys.path.insert(0, str(DP))
os.chdir(DP)
from eval.metrics_penebone import compute_penebone_metrics
from eval.metrics_contact import compute_contact_metrics

OUT = ROOT / "artifacts/dpmotion_reproduction"
PAIR_DIR = OUT / "ablations/guidance_pairs"


def condition_metrics(x):
    roots = x[:, :, 0, :]
    center_delta = roots[:2, :, :][:, :, [0, 2]].mean(axis=0) - roots[2:, :, :][:, :, [0, 2]].mean(axis=0)
    center_d = np.linalg.norm(center_delta, axis=-1)
    cross = np.stack([np.linalg.norm(roots[i][:, [0, 2]] - roots[j][:, [0, 2]], axis=-1)
                      for i in (0, 1) for j in (2, 3)])
    within = {"group1_root_3d_mean_m": float(np.linalg.norm(roots[0] - roots[1], axis=-1).mean()),
              "group2_root_3d_mean_m": float(np.linalg.norm(roots[2] - roots[3], axis=-1).mean()),
              "group1_root_xz_mean_m": float(np.linalg.norm(roots[0][:, [0, 2]] - roots[1][:, [0, 2]], axis=-1).mean()),
              "group2_root_xz_mean_m": float(np.linalg.norm(roots[2][:, [0, 2]] - roots[3][:, [0, 2]], axis=-1).mean())}
    pene = compute_penebone_metrics(x, joint_radius=0.02, device="cpu",
        interaction_pairs=[(0, 1), (2, 3)], non_interaction_pairs=[(0, 2), (0, 3), (1, 2), (1, 3)])
    contact = compute_contact_metrics(x, interaction_pairs=[(0, 1), (2, 3)],
        non_interaction_pairs=[(0, 2), (0, 3), (1, 2), (1, 3)], contact_threshold=0.10)
    root_path = [float(np.linalg.norm(np.diff(roots[i][:, [0, 2]], axis=0), axis=-1).sum()) for i in range(4)]
    joint_speed = np.linalg.norm(np.diff(x, axis=1), axis=-1)
    return {
        "group_center_xz_distance_m": {"mean": float(center_d.mean()), "min": float(center_d.min()),
            "fraction_frames_below_3m": float((center_d < 3.0).mean())},
        "cross_group_root_xz_distance_m": {"mean": float(cross.mean()), "min": float(cross.min()),
            "fraction_pair_frames_below_3m": float((cross < 3.0).mean())},
        "within_group_root_distances": within,
        "root_xz_path_length_m_by_person": root_path,
        "mean_joint_xyz_speed_m_per_frame": float(joint_speed.mean()),
        "penebone_depth_sum_m": pene["penebone"],
        "penetration_rate_fraction": pene["pene_rate"],
        "interaction_penebone_depth_sum_m": pene["penebone_inter"],
        "non_interaction_penebone_depth_sum_m": pene["penebone_non"],
        "contact_frame_rate_pct": contact["cframe_all"],
        "contact_joint_pairs_per_frame": contact["contact_all"],
    }


def main():
    pairs = json.loads((PAIR_DIR / "pairs_manifest.json").read_text())["pairs"]
    results = []
    for pair in pairs:
        fid = pair["file_id"]
        off = np.load(PAIR_DIR / "off" / ("scene_{}_4p.npy".format(fid)))
        on = np.load(PAIR_DIR / "on" / ("scene_{}_4p.npy".format(fid)))
        if off.shape != (4, 150, 22, 3) or on.shape != off.shape:
            raise ValueError("bad shape for " + fid)
        if not pair["same_torch_random_draws_exactly"]:
            raise ValueError("random draw mismatch for " + fid)
        delta = np.abs(on - off)
        results.append({
            "file_id": fid,
            "retrieval": pair["retrieval"],
            "same_text_and_seed": True,
            "same_torch_random_draws_exactly": True,
            "off": condition_metrics(off),
            "on": condition_metrics(on),
            "paired_motion_change": {
                "mean_abs_joint_xyz_change_m": float(delta.mean()),
                "max_abs_joint_xyz_change_m": float(delta.max()),
                "mean_abs_change_by_person_m": [float(delta[i].mean()) for i in range(4)],
            },
        })

    numeric = ["group_center_xz_distance_m.mean", "cross_group_root_xz_distance_m.mean",
               "penebone_depth_sum_m", "penetration_rate_fraction", "contact_frame_rate_pct",
               "mean_joint_xyz_speed_m_per_frame"]
    summary = {"n_fixed_cases": len(results), "cases": results,
               "scope_note": "Four fixed retrieval examples chosen to span rank-6 fallback and below-threshold selection; descriptive mechanism check, not a random 120-scene estimate or a quality claim.",
               "paired_means_on_minus_off": {}}
    for path in numeric:
        key = path.split(".")[-1]
        def get(row, cond):
            value = row[cond]
            for component in path.split("."):
                value = value[component]
            return value
        summary["paired_means_on_minus_off"][path] = float(np.mean([get(r, "on") - get(r, "off") for r in results]))
    (PAIR_DIR / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary["paired_means_on_minus_off"], indent=2))
    for row in results:
        print(row["file_id"], "center", row["off"]["group_center_xz_distance_m"]["mean"],
              "->", row["on"]["group_center_xz_distance_m"]["mean"],
              "PeneBone", row["off"]["penebone_depth_sum_m"], "->", row["on"]["penebone_depth_sum_m"],
              "motion_delta", row["paired_motion_change"]["mean_abs_joint_xyz_change_m"])


if __name__ == "__main__":
    main()
