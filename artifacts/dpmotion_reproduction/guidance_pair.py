#!/usr/bin/env python3
"""Matched guidance on/off checks with captured Torch noise draws."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import repro as base


def capture_random_calls(call):
    draws = []
    orig_randn, orig_randn_like = torch.randn, torch.randn_like

    def record(fn, *args, **kwargs):
        value = fn(*args, **kwargs)
        a = value.detach().cpu().contiguous().numpy()
        draws.append({"shape": list(a.shape), "dtype": str(a.dtype),
                      "sha256": hashlib.sha256(a.tobytes()).hexdigest()})
        return value

    def randn(*args, **kwargs):
        return record(orig_randn, *args, **kwargs)

    def randn_like(*args, **kwargs):
        return record(orig_randn_like, *args, **kwargs)

    torch.randn, torch.randn_like = randn, randn_like
    try:
        result = call()
    finally:
        torch.randn, torch.randn_like = orig_randn, orig_randn_like
    return result, draws


def main():
    ids = ["2636", "5647", "2813", "4000"]
    entries = {x["file_id"]: x for x in base.get_test_entries()}
    retrievals = base.get_retrieval_map()
    model = base.load_model_strict("cuda:0", save_audit=False)
    root = OUT / "ablations/guidance_pairs"
    root.mkdir(parents=True, exist_ok=True)
    summaries = []

    for fid in ids:
        entry = entries[fid]
        selected = retrievals[fid]["selected"]
        prompt = selected["text"]
        seed = base.BASE_SEED + int(fid)
        condition_rows = {}
        draw_lists = {}
        for name, enabled in [("off", False), ("on", True)]:
            path = root / name / ("scene_{}_4p.npy".format(fid))
            trace_path = (OUT / "guidance_pairs" / ("scene_{}_on.json".format(fid))) if enabled else None
            result, draws = capture_random_calls(lambda: base._run_scene(
                model, entry, prompt, 3.0, 5.0, seed, enabled, path, trace_path))
            draw_lists[name] = draws
            condition_rows[name] = {"path": str(path.relative_to(ROOT)),
                                    "mean_random_draw_count": len(draws),
                                    "random_draw_sequence_sha256": hashlib.sha256(
                                        json.dumps(draws, sort_keys=True).encode()).hexdigest(),
                                    "first_draw_sha256": draws[0]["sha256"] if draws else None,
                                    "last_draw_sha256": draws[-1]["sha256"] if draws else None}
            print("GUIDANCE_PAIR_DONE", fid, name, len(draws), flush=True)
        same = draw_lists["off"] == draw_lists["on"]
        summaries.append({"file_id": fid, "seed": seed,
                          "group1_text": entry["interaction_text"],
                          "group2_text": prompt, "retrieval": selected,
                          "window": 150, "distance_m": 3.0, "distance_weight": 5.0,
                          "sampler": "unmodified upstream forward_single; ancestral DDPM 1000",
                          "same_torch_random_draws_exactly": same,
                          "torch_random_draw_count": len(draw_lists["on"]),
                          "conditions": condition_rows})
        base.write_json(root / ("scene_{}_pair.json".format(fid)), summaries[-1])
        if not same:
            raise RuntimeError("Torch RNG draw sequence differs for paired guidance condition: " + fid)

    base.write_json(root / "pairs_manifest.json", {"pairs": summaries,
                    "rationale": "Same seed, prompt, group graph and sampling settings; exact equality of captured torch.randn/randn_like outputs verifies paired random noise."})


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
