#!/usr/bin/env python3
"""InterCLIP group-wise text-motion alignment for the full 2x2 ablation."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import clip

ROOT = Path(__file__).resolve().parents[2]
INTERGEN = ROOT / "code/InterGen-official"
CKPT = ROOT / "code/FreeMotion-Codes/eval_model/interclip.ckpt"
OUT = ROOT / "artifacts/dpmotion_reproduction/ablations"
sys.path.insert(0, str(INTERGEN))
os.chdir(INTERGEN)

clip_load = clip.load
def cached_clip_load(name, *args, **kwargs):
    kwargs.setdefault("download_root", str(ROOT / "assets/clip"))
    return clip_load(name, *args, **kwargs)
clip.load = cached_clip_load

from configs import get_config
from datasets.evaluator_models import InterCLIP

CONDITIONS = ["A_no_rag_no_guidance", "B_rag_no_guidance",
              "C_no_rag_guidance", "D_rag_guidance"]


def load_model(device):
    model = InterCLIP(get_config("configs/eval_model.yaml"))
    checkpoint = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    for key in list(state.keys()):
        if "model." in key:
            state[key.replace("model.", "")] = state.pop(key)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, checkpoint


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embed(model, motions, texts, device, batch_size):
    text_all, motion_all = [], []
    for start in range(0, len(texts), batch_size):
        stop = min(len(texts), start + batch_size)
        x = torch.from_numpy(motions[start:stop]).to(device=device, dtype=torch.float32)
        batch = {"motions": x, "mask": torch.ones(x.shape[:2], device=device),
                 "motion_lens": torch.full((len(x),), x.shape[1], device=device, dtype=torch.long),
                 "text": texts[start:stop]}
        with torch.no_grad():
            batch = model.encode_motion(batch)
            batch = model.encode_text(batch)
        text_all.append(batch["text_emb"].cpu().numpy())
        motion_all.append(batch["motion_emb"].cpu().numpy())
    return np.concatenate(text_all), np.concatenate(motion_all)


def score(text_emb, motion_emb):
    dist = np.linalg.norm(text_emb[:, None, :] - motion_emb[None, :, :], axis=-1)
    order = np.argsort(dist, axis=1)
    diagonal = np.arange(len(dist))
    ranks = np.empty(len(dist), dtype=np.int64)
    for i in range(len(dist)):
        ranks[i] = int(np.where(order[i] == i)[0][0]) + 1
    cosine = np.sum(text_emb * motion_emb, axis=-1) / (
        np.linalg.norm(text_emb, axis=-1) * np.linalg.norm(motion_emb, axis=-1) + 1e-12)
    return {"n": int(len(dist)), "matching_distance_mean": float(dist[diagonal, diagonal].mean()),
            "matching_distance_median": float(np.median(dist[diagonal, diagonal])),
            "paired_cosine_similarity_mean": float(cosine.mean()),
            "r_precision_top1": float((ranks <= 1).mean()),
            "r_precision_top2": float((ranks <= 2).mean()),
            "r_precision_top3": float((ranks <= 3).mean()),
            "mean_retrieval_rank": float(ranks.mean()),
            "rank_per_case": ranks.tolist(),
            "matched_distance_per_case": np.diag(dist).tolist(),
            "duplicate_text_count": int(len(text_emb) - len({tuple(x.tolist()) for x in text_emb}))}


def paired_summary(delta, seed):
    """Scene-paired bootstrap interval for a mean contrast."""
    delta = np.asarray(delta, dtype=np.float64)
    rng = np.random.RandomState(seed)
    means = np.empty(5000, dtype=np.float64)
    for i in range(len(means)):
        means[i] = rng.choice(delta, size=len(delta), replace=True).mean()
    return {"mean": float(delta.mean()), "median": float(np.median(delta)),
            "bootstrap_95pct_ci": [float(np.quantile(means, .025)),
                                    float(np.quantile(means, .975))],
            "fraction_treated_lower_distance": float((delta < 0).mean())}


def paired_rank_summary(treated_ranks, control_ranks, k, seed):
    """Paired change in per-scene recall@k under the condition-specific text set."""
    treated = np.asarray(treated_ranks) <= k
    control = np.asarray(control_ranks) <= k
    delta = treated.astype(np.float64) - control.astype(np.float64)
    rng = np.random.RandomState(seed)
    means = np.empty(5000, dtype=np.float64)
    for i in range(len(means)):
        means[i] = rng.choice(delta, size=len(delta), replace=True).mean()
    return {"mean_delta_recall_at_k": float(delta.mean()),
            "bootstrap_95pct_ci": [float(np.quantile(means, .025)),
                                    float(np.quantile(means, .975))],
            "fraction_scenes_improved": float((delta > 0).mean()),
            "fraction_scenes_regressed": float((delta < 0).mean()),
            "fraction_scenes_unchanged": float((delta == 0).mean())}


def run(args):
    out_root = OUT / ("full_2x2_smoke" if args.smoke else "full_2x2")
    model, checkpoint = load_model(args.device)
    result = {"evaluator": "FreeMotion-Codes InterCLIP; strict state_dict load",
              "checkpoint": str(CKPT.relative_to(ROOT)),
              "checkpoint_sha256": file_sha256(CKPT),
              "checkpoint_epoch": checkpoint.get("epoch"),
              "feature_protocol": "generated denormalized 262-D per-person global representation; pair inputs concatenate two people; 150 valid frames",
              "condition_scores": {}, "protocol_note": "Group2 alignment is the primary RAG-sensitive diagnostic. Text-motion R-precision is computed across this fixed set and is not the paper's reported multi-group score."}
    intermediates = {}
    for condition in CONDITIONS:
        rows_path = out_root / (condition + ".jsonl")
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        if any(row.get("status") != "ok" for row in rows):
            raise RuntimeError("failed generation rows in " + condition)
        rows = sorted(rows, key=lambda row: row["index"])
        if args.limit and len(rows) != args.limit:
            raise RuntimeError("expected {} rows for {} got {}".format(args.limit, condition, len(rows)))
        texts1, texts2, pairs1, pairs2 = [], [], [], []
        for row in rows:
            features = np.load(ROOT / row["feature_path"])
            if features.shape != (4, 150, 262) or not np.isfinite(features).all():
                raise ValueError("invalid features: " + row["feature_path"])
            # InterCLIP follows the official InterHuman two-person evaluator input.
            pairs1.append(np.concatenate([features[0], features[1]], axis=-1))
            pairs2.append(np.concatenate([features[2], features[3]], axis=-1))
            texts1.append(row["group1_text"])
            texts2.append(row["group2_text"])
        emb1_t, emb1_m = embed(model, np.stack(pairs1), texts1, args.device, args.batch_size)
        emb2_t, emb2_m = embed(model, np.stack(pairs2), texts2, args.device, args.batch_size)
        norm1m = emb1_m / np.clip(np.linalg.norm(emb1_m, axis=-1, keepdims=True), 1e-12, None)
        norm2m = emb2_m / np.clip(np.linalg.norm(emb2_m, axis=-1, keepdims=True), 1e-12, None)
        norm1t = emb1_t / np.clip(np.linalg.norm(emb1_t, axis=-1, keepdims=True), 1e-12, None)
        norm2t = emb2_t / np.clip(np.linalg.norm(emb2_t, axis=-1, keepdims=True), 1e-12, None)
        group_motion_distance = 1.0 - np.sum(norm1m * norm2m, axis=-1)
        group_text_distance = 1.0 - np.sum(norm1t * norm2t, axis=-1)
        result["condition_scores"][condition] = {
            "group1_test_text_alignment_control": score(emb1_t, emb1_m),
            "group2_conditioned_text_alignment": score(emb2_t, emb2_m),
            "within_scene_group_motion_embedding_cosine_distance": {
                "mean": float(group_motion_distance.mean()),
                "per_case": group_motion_distance.tolist(),
            },
            "within_scene_group_text_embedding_cosine_distance": {
                "mean": float(group_text_distance.mean()),
                "per_case": group_text_distance.tolist(),
            },
            "n_scenes": len(rows),
        }
        intermediates[condition + "_group1"] = {"text": emb1_t, "motion": emb1_m}
        intermediates[condition + "_group2"] = {"text": emb2_t, "motion": emb2_m}
        print("INTERCLIP_EVALUATED", condition, len(rows),
              result["condition_scores"][condition]["group2_conditioned_text_alignment"]["matching_distance_mean"],
              result["condition_scores"][condition]["group2_conditioned_text_alignment"]["r_precision_top1"], flush=True)

    # Paired effect sizes and per-scene differences for each factorial contrast.
    contrasts = {
        "RAG_effect_without_guidance": ("B_rag_no_guidance", "A_no_rag_no_guidance"),
        "RAG_effect_with_guidance": ("D_rag_guidance", "C_no_rag_guidance"),
        "guidance_effect_without_RAG": ("C_no_rag_guidance", "A_no_rag_no_guidance"),
        "guidance_effect_with_RAG": ("D_rag_guidance", "B_rag_no_guidance"),
    }
    result["paired_contrasts"] = {}
    for contrast_index, (name, (treated, control)) in enumerate(contrasts.items()):
        a = np.asarray(result["condition_scores"][treated]["group2_conditioned_text_alignment"]["matched_distance_per_case"])
        b = np.asarray(result["condition_scores"][control]["group2_conditioned_text_alignment"]["matched_distance_per_case"])
        delta = a - b
        result["paired_contrasts"][name] = {
            **paired_summary(delta, seed=20261008 + contrast_index),
            "distance_treated_minus_control_mean": float(delta.mean()),
            "distance_treated_minus_control_median": float(np.median(delta)),
            "per_case_distance_difference": delta.tolist(),
        }
    result["paired_group_motion_diversity_contrasts"] = {}
    for name, (treated, control) in contrasts.items():
        a = np.asarray(result["condition_scores"][treated]["within_scene_group_motion_embedding_cosine_distance"]["per_case"])
        b = np.asarray(result["condition_scores"][control]["within_scene_group_motion_embedding_cosine_distance"]["per_case"])
        delta = a - b
        result["paired_group_motion_diversity_contrasts"][name] = {
            "distance_treated_minus_control_mean": float(delta.mean()),
            "fraction_treated_more_different": float((delta > 0).mean()),
            "per_case_difference": delta.tolist(),
        }

    result["paired_r_precision_contrasts"] = {}
    for contrast_index, (name, (treated, control)) in enumerate(contrasts.items()):
        tscore = result["condition_scores"][treated]["group2_conditioned_text_alignment"]
        cscore = result["condition_scores"][control]["group2_conditioned_text_alignment"]
        if name.startswith("RAG_"):
            result["paired_r_precision_contrasts"][name] = {
                "note": "Per-scene rank changes against each condition's own 120-text/120-motion set; text queries differ under RAG.",
                **{"R@{}".format(k): paired_rank_summary(
                    tscore["rank_per_case"], cscore["rank_per_case"], k,
                    seed=20262000 + 100 * contrast_index + k)
                   for k in (1, 2, 3)},
            }
    (out_root / "interclip_alignment.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    np.savez_compressed(out_root / "interclip_features.npz", **{
        name + "_" + kind: arr for name, values in intermediates.items() for kind, arr in values.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if args.smoke and not args.limit:
        args.limit = 1
    run(args)


if __name__ == "__main__":
    main()
