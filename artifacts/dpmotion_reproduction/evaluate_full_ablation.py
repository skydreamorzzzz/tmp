#!/usr/bin/env python3
"""Run the unmodified DPMotion evaluator and paired factorial contrasts."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DP = ROOT / "code/DPMotion"
sys.path.insert(0, str(DP))
os.chdir(DP)
import dpmotion_eval

CONDITIONS = ["A_no_rag_no_guidance", "B_rag_no_guidance",
              "C_no_rag_guidance", "D_rag_guidance"]
METRICS = ["penebone", "pene_rate", "cframe_all", "contact_all", "diversity"]


def bootstrap_delta(values, seed=20261008, reps=5000):
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    means = np.empty(reps)
    for i in range(reps):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "fraction_delta_positive": float((x > 0).mean()),
            "bootstrap_95pct_ci": [float(np.quantile(means, .025)), float(np.quantile(means, .975))]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    name = "full_2x2_smoke" if args.smoke else "full_2x2"
    root = ROOT / "artifacts/dpmotion_reproduction/ablations" / name
    metric_by_cond = {}
    for cond in CONDITIONS:
        out = root / ("repo_metrics_" + cond + ".json")
        summary = dpmotion_eval.evaluate_all(str(root / cond / "joints"), str(out), device="cpu")
        if not summary:
            raise RuntimeError("empty repository metrics for " + cond)
        metric_by_cond[cond] = summary

    result = {"conditions": {c: metric_by_cond[c]["metrics"] for c in CONDITIONS},
              "num_scenes": {c: metric_by_cond[c]["num_scenes"] for c in CONDITIONS},
              "per_scene_factorial_contrasts": {},
              "design": "A=no RAG/no guidance; B=RAG/no guidance; C=no RAG/guidance; D=RAG/guidance",
              "contrast_convention": "RAG main effect=((B-A)+(D-C))/2; guidance main effect=((C-A)+(D-B))/2; interaction=(D-C)-(B-A). Positive delta is numerically larger, not necessarily better."}
    indexed = {c: {r["scene"].replace("scene_", ""): r for r in metric_by_cond[c]["per_scene"]} for c in CONDITIONS}
    ids = sorted(set.intersection(*(set(indexed[c]) for c in CONDITIONS)))
    if any(len(indexed[c]) != len(ids) for c in CONDITIONS):
        raise RuntimeError("conditions do not have the same valid scene IDs")
    for metric in METRICS:
        a, b, c, d = (np.array([indexed[key][fid][metric] for fid in ids], dtype=float)
                      for key in CONDITIONS)
        rag_no = b - a
        rag_yes = d - c
        guide_no = c - a
        guide_yes = d - b
        result["per_scene_factorial_contrasts"][metric] = {
            "rag_effect_without_guidance": bootstrap_delta(rag_no, seed=10),
            "rag_effect_with_guidance": bootstrap_delta(rag_yes, seed=11),
            "rag_main_effect": bootstrap_delta((rag_no + rag_yes) / 2, seed=12),
            "guidance_effect_without_rag": bootstrap_delta(guide_no, seed=13),
            "guidance_effect_with_rag": bootstrap_delta(guide_yes, seed=14),
            "guidance_main_effect": bootstrap_delta((guide_no + guide_yes) / 2, seed=15),
            "rag_guidance_interaction": bootstrap_delta(rag_yes - rag_no, seed=16),
            "per_scene_rag_main_effect": ((rag_no + rag_yes) / 2).tolist(),
        }
    result["scene_ids"] = ids
    out_path = root / "repo_ablation_comparison.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("REPO_ABLATION_EVALUATION_COMPLETE", len(ids), str(out_path))


if __name__ == "__main__":
    main()
