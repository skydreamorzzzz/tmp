#!/usr/bin/env python3
"""Reproducible driver around the unmodified public DPMotion inference/eval code."""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DPMOTION = ROOT / "code/DPMotion"
CLIP_CACHE = ROOT / "assets/clip"
CKPT = ROOT / "assets/intergen/intergen.ckpt"
BASE_SEED = 42
TRACE_STEPS = {999, 901, 900, 700, 500, 200, 101, 100, 99, 50, 0}

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(DPMOTION))
os.chdir(str(DPMOTION))  # DPMotion's own modules resolve config/data from cwd.

import clip
_clip_load = clip.load


def _cached_clip_load(name, *args, **kwargs):
    kwargs.setdefault("download_root", str(CLIP_CACHE))
    return _clip_load(name, *args, **kwargs)


clip.load = _cached_clip_load

from configs import get_config
from models.intergen import InterGen
from models.nets import MotionDiffusion
import models.group_person_losses as group_losses
from DPMotion import InteractionKnowledgeBase, split_interaction_text
import dpmotion_eval as official_eval


def sha256(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_strict(device="cuda:0", save_audit=True):
    cfg = get_config("configs/model.yaml")
    model = InterGen(cfg)
    checkpoint = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    raw = checkpoint["state_dict"]
    state = {(k[6:] if k.startswith("model.") else k): v for k, v in raw.items()}
    target = model.state_dict()
    matched = sorted(set(state) & set(target))
    mismatches = [{"key": k, "checkpoint": list(state[k].shape), "model": list(target[k].shape)}
                  for k in matched if tuple(state[k].shape) != tuple(target[k].shape)]
    missing = sorted(set(target) - set(state))
    unexpected = sorted(set(state) - set(target))
    if save_audit:
        write_json(OUT / "environment/model_compatibility.json", {
            "checkpoint": str(CKPT), "sha256": sha256(CKPT), "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"), "source_keys": len(raw),
            "matched_keys": len(matched), "missing_keys": missing,
            "unexpected_keys": unexpected, "shape_mismatches": mismatches,
            "strict_load": False if (missing or unexpected or mismatches) else True,
            "prefix_handling": "remove only leading model. from Lightning state_dict",
            "normalization": {
                "mean_sha256": sha256(DPMOTION / "data/global_mean.npy"),
                "std_sha256": sha256(DPMOTION / "data/global_std.npy"),
                "shape": list(np.load(DPMOTION / "data/global_mean.npy").shape),
                "same_as_intergen": all(np.array_equal(np.load(DPMOTION / "data" / name),
                                                        np.load(ROOT / "code/InterGen-official/data" / name))
                                         for name in ["global_mean.npy", "global_std.npy"]),
            },
        })
    if missing or unexpected or mismatches:
        del checkpoint, raw, state
        raise RuntimeError("DPMotion InterGen strict-load audit failed; see environment/model_compatibility.json")
    result = model.load_state_dict(state, strict=True)
    del checkpoint, raw, state
    model.eval().to(torch.device(device if torch.cuda.is_available() else "cpu"))
    if save_audit:
        meta = json.loads((OUT / "environment/model_compatibility.json").read_text())
        meta["strict_load"] = True
        meta["load_result"] = {"missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)}
        meta["device"] = str(next(model.parameters()).device)
        write_json(OUT / "environment/model_compatibility.json", meta)
    return model


def load_kb(path=None):
    path = Path(path or OUT / "rag/dpmotion_kb.pkl")
    kb = InteractionKnowledgeBase()
    kb.load(str(path))
    return kb


def selected_retrieval(kb, text, top_k=20, threshold=0.90):
    results = kb.query(text, top_k=top_k)
    selected = None
    rank = None
    reason = None
    for i, row in enumerate(results):
        if row["score"] < threshold:
            selected, rank, reason = row, i + 1, "first_similarity_below_threshold"
            break
    if selected is None and text != "" and len(results) > 5:
        selected, rank, reason = results[5], 6, "rank6_fallback_no_below_threshold"
    if selected is None:
        selected, rank, reason = {"interaction_text": text, "score": None}, None, "group1_text_fallback"
    return results, {"text": selected["interaction_text"], "score": selected.get("score"),
                     "rank": rank, "reason": reason}


def build_kb(args):
    started = time.time()
    annots = Path(args.annots_dir).resolve()
    files = sorted(annots.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    file_ids_hash = hashlib.sha256("\n".join(p.name for p in files).encode()).hexdigest()
    kb = InteractionKnowledgeBase()
    kb.build_from_annots(str(annots), dedup_sim_threshold=args.dedup_threshold)
    out = OUT / "rag"
    out.mkdir(parents=True, exist_ok=True)
    pkl = out / "dpmotion_kb.pkl"
    kb.save(str(pkl))
    np.save(out / "dpmotion_embeddings.npy", kb.embeddings)
    with (out / "dpmotion_entries.jsonl").open("w", encoding="utf-8") as f:
        for entry in kb.entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    write_json(out / "build_metadata.json", {
        "dpmotion_commit": os.popen("git -C {} rev-parse HEAD".format(DPMOTION)).read().strip(),
        "annotation_dir": str(annots), "annotation_files": len(files), "annotation_filenames_sha256": file_ids_hash,
        "dedup_sim_threshold": args.dedup_threshold, "clip": "ViT-L/14@336px cached official checkpoint",
        "kb_entries_after_exact_and_embedding_dedup": len(kb.entries),
        "embedding_shape": list(kb.embeddings.shape), "embedding_dtype": str(kb.embeddings.dtype),
        "pickle_sha256": sha256(pkl), "seconds": time.time() - started,
        "split_filter": None,
        "split_filter_note": "Official build_from_annots reads every .txt in annotations; test annotations are included.",
    })
    print("Built KB", len(kb.entries), "seconds", round(time.time() - started, 2), flush=True)


def precompute_rag(args):
    kb = load_kb(args.kb_path)
    entries = json.loads((DPMOTION / "data/test_texts_120.json").read_text())
    out = OUT / "rag/retrievals.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, e in enumerate(entries):
        top, picked = selected_retrieval(kb, e["interaction_text"], 20, 0.90)
        row = {"index": i, "file_id": e["file_id"], "query": e["interaction_text"],
               "top_k": 20, "threshold": 0.90, "top_results": top, "selected": picked}
        rows.append(row)
        print("RAG", i + 1, len(entries), e["file_id"], picked["rank"], picked["score"], flush=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(OUT / "rag/retrieval_summary.json", {
        "queries": len(rows), "top_k": 20, "threshold": 0.90,
        "selected_below_threshold": sum(r["selected"]["reason"] == "first_similarity_below_threshold" for r in rows),
        "fallback_rank6": sum(r["selected"]["reason"] == "rank6_fallback_no_below_threshold" for r in rows),
        "same_text_fallback": sum(r["selected"]["reason"] == "group1_text_fallback" for r in rows),
        "retrievals_sha256": sha256(out),
    })


def _root_data(diffusion, x_start, sampling_info):
    starts = sampling_info.get("all_starts", [])
    cur = sampling_info.get("current_indices", [])
    grouped = []
    dim = starts[0].shape[-1]
    if x_start.shape[-1] == 2 * dim and len(cur) == 2:
        replacements = {cur[0]: x_start[..., :dim], cur[1]: x_start[..., dim:]}
    elif x_start.shape[-1] == dim and len(cur) == 1:
        replacements = {cur[0]: x_start}
    elif x_start.shape[-1] == 2 * dim and len(cur) == 1:
        replacements = {cur[0]: x_start[..., :dim]}
    else:
        replacements = {}
    for i, old in enumerate(starts):
        src = replacements.get(i, old)
        physical = src * diffusion.normalizer.motion_std.to(src.device) + diffusion.normalizer.motion_mean.to(src.device)
        grouped.append(physical[..., :66].reshape(physical.shape[0], physical.shape[1], 22, 3)[..., 0, :])
    return grouped


_TRACE = {"events": [], "current": None, "gp_loss": None, "enabled": True}
_ORIGINAL_GRADIENTS = MotionDiffusion.gradients
_ORIGINAL_P_SAMPLE = MotionDiffusion.p_sample_with_grad
_ORIGINAL_GP = group_losses.compute_group_person_losses


def _gp_spy(*args, **kwargs):
    result = _ORIGINAL_GP(*args, **kwargs)
    _TRACE["gp_loss"] = result[1]
    return result


group_losses.compute_group_person_losses = _gp_spy


def _distance_snapshot(diffusion, x_start, sampling_info):
    roots = _root_data(diffusion, x_start, sampling_info)
    groups = sampling_info["inter_graph"]["groups"]
    names = list(groups)
    if len(names) < 2:
        return {}
    a, b = groups[names[0]], groups[names[1]]
    center_a = torch.stack([roots[i] for i in a]).mean(0)
    center_b = torch.stack([roots[i] for i in b]).mean(0)
    center = torch.linalg.vector_norm((center_a - center_b)[..., [0, 2]], dim=-1).mean()
    cross = [torch.linalg.vector_norm((roots[i] - roots[j])[..., [0, 2]], dim=-1).mean()
             for i in a for j in b]
    return {"group_center_distance_mean_m": float(center.detach().cpu()),
            "cross_member_distances_mean_m": [float(x.detach().cpu()) for x in cross],
            "cross_member_min_mean_m": float(torch.stack(cross).min().detach().cpu())}


def _gradients_spy(self, x_start, x, sampling_info):
    if not _TRACE["enabled"]:
        return torch.zeros((), device=x.device), torch.zeros_like(x)
    loss, grad = _ORIGINAL_GRADIENTS(self, x_start, x, sampling_info)
    event = _TRACE.get("current")
    if event is not None:
        event["loss_total"] = float(loss.detach().cpu()) if torch.is_tensor(loss) else float(loss)
        event["gradient_l2_norm"] = float(torch.linalg.vector_norm(grad.detach()).cpu()) if torch.is_tensor(grad) else 0.0
        event["gradient_mean_abs"] = float(grad.detach().abs().mean().cpu()) if torch.is_tensor(grad) else 0.0
        event["gradient_max_abs"] = float(grad.detach().abs().max().cpu()) if torch.is_tensor(grad) else 0.0
        event["gradient_nonzero_fraction"] = float((grad.detach() != 0).float().mean().cpu()) if torch.is_tensor(grad) else 0.0
        event["group_loss_components"] = _TRACE.get("gp_loss")
        try:
            event.update(_distance_snapshot(self, x_start, sampling_info))
        except Exception as exc:
            event["distance_snapshot_error"] = repr(exc)
        tval = event["t"]
        current = sampling_info.get("current_indices", [])
        single_dim = sampling_info.get("all_starts", [x])[0].shape[-1]
        if tval <= self.t_bar_gli and len(current) == 2 and x_start.shape[-1] == 2 * single_dim:
            # The sampler concatenates both people's streams along the final
            # dimension, while the normalizer remains per-person (262-D).
            d = single_dim
            std = self.normalizer.motion_std.to(x_start.device)
            mean = self.normalizer.motion_mean.to(x_start.device)
            a = x_start[..., :d] * std + mean
            b = x_start[..., d:] * std + mean
            ja = a[..., :66].reshape(a.shape[0], a.shape[1], 22, 3)
            jb = b[..., :66].reshape(b.shape[0], b.shape[1], 22, 3)
            with torch.no_grad():
                event["gli_raw"] = float(self.topology_loss(ja, jb).sum().cpu())
            event["gli_scaled_weight"] = 0.6
    return loss, grad


def _p_sample_spy(self, model, x, t, *args, **kwargs):
    tval = int(t.reshape(-1)[0].item())
    info = kwargs.get("sampling_info") or {}
    idx = info.get("current_indices", [])
    event = None
    if tval in TRACE_STEPS:
        event = {"t": tval, "current_indices": list(idx), "guidance_enabled": bool(_TRACE["enabled"]),
                 "distance_weight": float(info.get("distance_weight", 1.0)),
                 "effective_distance_weight": float(info.get("distance_weight", 1.0)) * 1000.0 / (tval + 100.0),
                 "outer_gradient_scale": 5.0 + 5.0 * tval / 1000.0,
                 "distance_guidance_active": tval <= 900 and bool(info.get("inter_graph")),
                 "gli_guidance_active": tval <= 100 and len(idx) == 2}
    previous = _TRACE.get("current")
    _TRACE["current"] = event
    _TRACE["gp_loss"] = None
    try:
        result = _ORIGINAL_P_SAMPLE(self, model, x, t, *args, **kwargs)
        if event is not None:
            _TRACE["events"].append(event)
        return result
    finally:
        _TRACE["current"] = previous


MotionDiffusion.gradients = _gradients_spy
MotionDiffusion.p_sample_with_grad = _p_sample_spy


def get_test_entries():
    return json.loads((DPMOTION / "data/test_texts_120.json").read_text(encoding="utf-8"))


def get_retrieval_map():
    p = OUT / "rag/retrievals.jsonl"
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["file_id"]: r for r in rows}


def atomic_save(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def _load_or_rebuild_model():
    return load_model_strict("cuda:0")


def _run_scene(model, entry, group2, distance, weight, seed, guidance, outpath, tracepath=None):
    _TRACE["enabled"] = bool(guidance)
    _TRACE["events"] = []
    _TRACE["current"] = None
    seed_all(seed)
    graph = official_eval.build_4person_inter_graph(entry["interaction_text"], group2, distance)
    joints = official_eval.generate_single_scene(model, graph, entry["interaction_text"], group2,
                                                window_size=150, distance_weight=weight)
    if not np.isfinite(joints).all():
        raise FloatingPointError("non-finite generated joints")
    atomic_save(outpath, joints.astype(np.float32))
    if tracepath:
        tracepath.parent.mkdir(parents=True, exist_ok=True)
        write_json(tracepath, {"file_id": entry["file_id"], "seed": seed, "guidance_enabled": bool(guidance),
                               "distance_weight": weight, "distance_m": distance,
                               "window": 150, "events": _TRACE["events"]})
    return joints


def run_smoke(args):
    entries = get_test_entries()
    entry = next(e for e in entries if e["file_id"] == args.file_id)
    retrievals = get_retrieval_map()
    r = retrievals[entry["file_id"]]
    rag_text = r["selected"]["text"]
    model = _load_or_rebuild_model()
    modes = [
        ("A_intergen_no_rag_no_guidance", entry["interaction_text"], False),
        ("B_intergen_plus_rag_no_guidance", rag_text, False),
        ("C_intergen_plus_guidance_no_rag", entry["interaction_text"], True),
        ("D_rag_plus_guidance", rag_text, True),
    ]
    records = []
    for name, group2, guide in modes:
        start = time.time()
        dest = OUT / "ablations/four_way" / name / ("scene_{}_4p.npy".format(args.file_id))
        trace = OUT / "guidance" / (name + "_scene_{}.json".format(args.file_id)) if guide else None
        joints = _run_scene(model, entry, group2, args.distance, args.distance_weight,
                            args.seed + int(args.file_id), guide, dest, trace)
        records.append({"condition": name, "file_id": args.file_id, "group1_text": entry["interaction_text"],
                        "group2_text": group2, "retrieval": r["selected"], "seed": args.seed + int(args.file_id),
                        "shape": list(joints.shape), "mean_abs_root_difference_from_A": None,
                        "output": str(dest.relative_to(ROOT)), "seconds": round(time.time() - start, 2),
                        "guidance_trace": str(trace.relative_to(ROOT)) if trace else None})
        print("SMOKE_DONE", json.dumps(records[-1], ensure_ascii=False), flush=True)
    base = np.load(OUT / "ablations/four_way/A_intergen_no_rag_no_guidance/scene_{}_4p.npy".format(args.file_id))
    for rec in records:
        arr = np.load(ROOT / rec["output"])
        rec["mean_abs_joint_diff_vs_A"] = float(np.abs(arr - base).mean())
        rec["max_abs_joint_diff_vs_A"] = float(np.abs(arr - base).max())
    write_json(OUT / "ablations/four_way/smoke_summary.json", {
        "seed_base": args.seed, "test_id": args.file_id, "window": 150,
        "distance_weight": args.distance_weight, "non_interactive_distance_m": args.distance,
        "shared_rag_text": rag_text,
        "all_conditions_same_scene_and_derived_seed": True,
        "conditions": records,
    })


def _valid_scene_file(path):
    try:
        x = np.load(path)
        return x.ndim == 4 and x.shape[0] == 4 and x.shape[2:] == (22, 3) and np.isfinite(x).all()
    except Exception:
        return False


def run_120(args):
    entries = get_test_entries()
    if args.limit:
        entries = entries[:args.limit]
    retrievals = get_retrieval_map()
    model = _load_or_rebuild_model()
    dest = OUT / "samples/full"
    dest.mkdir(parents=True, exist_ok=True)
    log = OUT / "logs/full_generation.jsonl"
    for i, entry in enumerate(entries):
        fid = entry["file_id"]
        path = dest / "scene_{}_4p.npy".format(fid)
        retrieval = retrievals[fid]
        if _valid_scene_file(path):
            print("SKIP_VALID", i + 1, len(entries), fid, flush=True)
            continue
        seed = args.seed + int(fid)
        started = time.time()
        rec = {"index": i, "file_id": fid, "seed": seed, "group1_text": entry["interaction_text"],
               "group2_text": retrieval["selected"]["text"], "selected": retrieval["selected"],
               "top20": retrieval["top_results"], "window": 150, "distance_m": args.distance,
               "distance_weight": args.distance_weight, "angle_weight": 0.0, "guidance": True}
        try:
            joints = _run_scene(model, entry, rec["group2_text"], args.distance, args.distance_weight,
                                seed, True, path)
            rec.update({"status": "ok", "shape": list(joints.shape), "seconds": round(time.time() - started, 2),
                        "path": str(path.relative_to(ROOT))})
        except Exception as exc:
            rec.update({"status": "failed", "error": repr(exc), "seconds": round(time.time() - started, 2)})
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print("SCENE", i + 1, len(entries), fid, rec["status"], rec.get("seconds"), flush=True)


def evaluate(args):
    summary = official_eval.evaluate_all(str(OUT / "samples/full"), str(OUT / "metrics/repo_metrics_full.json"),
                                         device=args.eval_device)
    print("EVALUATION_COMPLETE", summary.get("num_scenes"), flush=True)


def model_audit(args):
    load_model_strict(args.device)
    print((OUT / "environment/model_compatibility.json").read_text())


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("audit-model"); q.add_argument("--device", default="cuda:0"); q.set_defaults(fn=model_audit)
    q = sub.add_parser("build-kb"); q.add_argument("--annots-dir", default=str(ROOT / "data/InterHuman/annots")); q.add_argument("--dedup-threshold", type=float, default=0.95); q.set_defaults(fn=build_kb)
    q = sub.add_parser("rag-probe"); q.add_argument("--kb-path", default=None); q.set_defaults(fn=precompute_rag)
    q = sub.add_parser("smoke"); q.add_argument("--file-id", default="2636"); q.add_argument("--seed", type=int, default=BASE_SEED); q.add_argument("--distance", type=float, default=3.0); q.add_argument("--distance-weight", type=float, default=5.0); q.set_defaults(fn=run_smoke)
    q = sub.add_parser("run120"); q.add_argument("--seed", type=int, default=BASE_SEED); q.add_argument("--distance", type=float, default=3.0); q.add_argument("--distance-weight", type=float, default=5.0); q.add_argument("--limit", type=int, default=0); q.set_defaults(fn=run_120)
    q = sub.add_parser("evaluate"); q.add_argument("--eval-device", default="cpu"); q.set_defaults(fn=evaluate)
    args = p.parse_args()
    torch.set_num_threads(4)
    args.fn(args)


if __name__ == "__main__":
    main()
