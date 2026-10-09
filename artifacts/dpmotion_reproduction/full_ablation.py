#!/usr/bin/env python3
"""Full 2x2 RAG x guidance ablation on the fixed 120-scene split.

The upstream DPMotion repository is imported unchanged. This driver only saves
the full per-person 262-D output alongside the xyz array so InterCLIP can score
the generated two-person group against its conditioning text.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import subprocess
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
OUT = ART / "ablations"
sys.path.insert(0, str(ART))
import repro as base
from utils.utils import MotionNormalizer

CONDITIONS = {
    "A_no_rag_no_guidance": (False, False),
    "B_rag_no_guidance": (True, False),
    "C_no_rag_guidance": (False, True),
    "D_rag_guidance": (True, True),
}


def atomic_npy(path, x):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp.npy")
    np.save(temp, np.asarray(x, dtype=np.float32))
    os.replace(str(temp), str(path))


def valid(path, shape):
    try:
        x = np.load(path, mmap_mode="r")
        return x.shape == shape and np.isfinite(x).all()
    except Exception:
        return False


def generate_features(model, entry, group2_text, use_guidance, distance, weight):
    device = next(model.parameters()).device
    normalizer = MotionNormalizer()
    conditions = {}
    for group, text in [("Group1", entry["interaction_text"]), ("Group2", group2_text)]:
        enc = model.text_process({"text": [text]})
        conditions[group] = enc["cond"].detach()

    batch = OrderedDict()
    batch["motion_lens"] = torch.full((1, 1), 150, device=device, dtype=torch.long)
    batch["text"] = [entry["interaction_text"]]
    batch["motion_text_map"] = conditions
    batch["distance_weight"] = weight
    batch["angle_weight"] = 0.0
    batch["inter_graph"] = base.official_eval.build_4person_inter_graph(
        entry["interaction_text"], group2_text, distance)
    batch = model.text_process(batch)

    # The upstream sampler remains intact. For the no-guidance ablation the
    # runner's existing diagnostic hook returns a zero gradient at guidance
    # calls, preserving the exact sampler and RNG sequence.
    base._TRACE["enabled"] = bool(use_guidance)
    base._TRACE["events"] = []
    base._TRACE["current"] = None
    batch.update(model.decoder.forward_single(batch))
    sequences = np.concatenate([normalizer.backward(y.detach().cpu().numpy())
                                for y in batch["output"]], axis=0)
    if sequences.shape != (4, 150, 262) or not np.isfinite(sequences).all():
        raise ValueError("bad feature output {}".format(sequences.shape))
    joints = sequences[..., :66].reshape(4, 150, 22, 3)
    joints = gaussian_filter1d(joints, 1, axis=1, mode="nearest")
    return sequences.astype(np.float32), joints.astype(np.float32)


def capture_torch_noise(call):
    draws = []
    randn, randn_like = torch.randn, torch.randn_like
    def capture(fn, *args, **kwargs):
        value = fn(*args, **kwargs)
        array = value.detach().cpu().contiguous().numpy()
        draws.append([list(array.shape), hashlib.sha256(array.tobytes()).hexdigest()])
        return value
    torch.randn = lambda *a, **k: capture(randn, *a, **k)
    torch.randn_like = lambda *a, **k: capture(randn_like, *a, **k)
    try:
        result = call()
    finally:
        torch.randn, torch.randn_like = randn, randn_like
    digest = hashlib.sha256(json.dumps(draws, separators=(",", ":")).encode()).hexdigest()
    return result, {"torch_random_tensor_count": len(draws), "torch_random_sequence_sha256": digest,
                    "initial_noise_sha256": draws[0][1] if draws else None}


def run(args):
    entries = base.get_test_entries()
    if args.limit:
        entries = entries[:args.limit]
    retrievals = base.get_retrieval_map()
    model = base.load_model_strict(args.device, save_audit=False)
    out_root = OUT / ("full_2x2_smoke" if args.smoke else "full_2x2")
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "design": "2x2 factorial: RAG {off,on} x guidance {off,on}",
        "no_rag_definition": "Group2 receives the same test caption as Group1, matching prior A/C smoke definition",
        "same_seed_all_arms": True,
        "base_seed": args.seed,
        "window": 150,
        "sampler": "unmodified DPMotion forward_single / 1000-step ancestral DDPM",
        "distance_m": args.distance,
        "distance_weight": args.distance_weight,
        "dpmotion_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base.DPMOTION, text=True).strip(),
        "intergen_checkpoint_sha256": base.sha256(base.CKPT),
        "runtime": {"python": sys.version, "torch": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "device": str(args.device)},
        "conditions": list(CONDITIONS),
        "test_ids": [e["file_id"] for e in entries],
        "status": "running",
    }
    base.write_json(out_root / "config.json", manifest)
    smoke_noise = {}

    for condition, (use_rag, use_guidance) in CONDITIONS.items():
        cond_dir = out_root / condition
        feat_dir, joint_dir = cond_dir / "features", cond_dir / "joints"
        feat_dir.mkdir(parents=True, exist_ok=True)
        joint_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_root / (condition + ".jsonl")
        prior = {}
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    prior[row["file_id"]] = row
                except Exception:
                    continue
        for i, entry in enumerate(entries):
            fid = entry["file_id"]
            feature_path = feat_dir / ("scene_{}_4p_262.npy".format(fid))
            joint_path = joint_dir / ("scene_{}_4p.npy".format(fid))
            if valid(feature_path, (4, 150, 262)) and valid(joint_path, (4, 150, 22, 3)):
                continue
            rag_row = retrievals[fid]["selected"]
            group2_text = rag_row["text"] if use_rag else entry["interaction_text"]
            seed = args.seed + int(fid)
            base.seed_all(seed)
            started = time.time()
            row = {"index": i, "file_id": fid, "condition": condition,
                   "rag": use_rag, "guidance": use_guidance,
                   "seed": seed, "group1_text": entry["interaction_text"],
                   "group2_text": group2_text,
                   "retrieval": rag_row if use_rag else None,
                   "distance_m": args.distance, "distance_weight": args.distance_weight,
                   "window": 150, "sampler": "DDPM ancestral 1000"}
            try:
                call = lambda: generate_features(model, entry, group2_text,
                    use_guidance, args.distance, args.distance_weight)
                if args.smoke:
                    (features, joints), noise_info = capture_torch_noise(call)
                    row["noise_capture"] = noise_info
                    smoke_noise[condition] = noise_info
                else:
                    features, joints = call()
                atomic_npy(feature_path, features)
                atomic_npy(joint_path, joints)
                row.update(status="ok", seconds=round(time.time() - started, 2),
                           feature_path=str(feature_path.relative_to(ROOT)),
                           joint_path=str(joint_path.relative_to(ROOT)))
            except Exception as exc:
                row.update(status="failed", error=repr(exc), seconds=round(time.time() - started, 2))
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            print("ABLATION", condition, i + 1, len(entries), fid,
                  row["status"], row.get("seconds"), flush=True)
    statuses = []
    for condition in CONDITIONS:
        log_path = out_root / (condition + ".jsonl")
        latest = {}
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    latest[row["file_id"]] = row.get("status")
        statuses.extend(latest.get(entry["file_id"]) for entry in entries)
    manifest["status"] = "complete" if len(statuses) == len(entries) * len(CONDITIONS) and all(x == "ok" for x in statuses) else "partial"
    if args.smoke:
        hashes = {x.get("torch_random_sequence_sha256") for x in smoke_noise.values()}
        manifest["smoke_torch_random_draws_identical_across_arms"] = len(hashes) == 1 and len(smoke_noise) == len(CONDITIONS)
        manifest["smoke_noise_summaries"] = smoke_noise
        if not manifest["smoke_torch_random_draws_identical_across_arms"]:
            base.write_json(out_root / "config.json", manifest)
            raise RuntimeError("2x2 smoke did not reuse identical Torch random draws")
    base.write_json(out_root / "config.json", manifest)
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=base.BASE_SEED)
    p.add_argument("--distance", type=float, default=3.0)
    p.add_argument("--distance-weight", type=float, default=5.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if args.smoke and not args.limit:
        args.limit = 1
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()
