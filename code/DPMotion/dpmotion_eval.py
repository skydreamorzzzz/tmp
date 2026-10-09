#!/usr/bin/env python3
"""
DPMotion 批量生成 + 评估脚本

评估指标：
  - PeneBone:   骨骼穿透深度 + 穿透率
  - CFrame:     接触帧占比 (%)
  - Contact:    平均每帧接触关节对数
  - Diversity:  场景内动作多样性 (motion embedding 距离均值)

用法：
    python dpmotion_eval.py --all
    python dpmotion_eval.py --eval-only --npy-dir results/dpmotion_eval
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from collections import OrderedDict
from typing import List, Dict

import numpy as np
import torch

# ---------------------------------------------------------------------------
_DPMOTION_DIR = os.path.dirname(os.path.abspath(__file__))
if _DPMOTION_DIR not in sys.path:
    sys.path.insert(0, _DPMOTION_DIR)
os.chdir(_DPMOTION_DIR)

from DPMotion import InteractionKnowledgeBase, split_interaction_text, load_model
from utils.utils import MotionNormalizer

_EVAL_DIR = os.path.join(_DPMOTION_DIR, "eval")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
from metrics_penebone import compute_penebone_metrics
from metrics_contact import compute_contact_metrics
from metrics_core import diversity_motion_embedding

# ============================================================================
# 常量
# ============================================================================
FPS = 30
DEFAULT_WINDOW = 150
DEFAULT_NUM_SAMPLES = 120
DEFAULT_NON_INTERACT_DIST = 3.0
ANNOTS_DIR = "/projects/wbl/InterGen/data/annots"
SPLIT_FILE = "/projects/wbl/InterGen/data/split/test.txt"
KB_PATH = os.path.join(_DPMOTION_DIR, "dpmotion_kb.pkl")

# ============================================================================
# 数据准备
# ============================================================================

def load_test_texts(
    split_file: str = SPLIT_FILE,
    annots_dir: str = ANNOTS_DIR,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = 42,
) -> List[Dict]:
    # 优先使用本地预处理的测试数据
    local_test_json = os.path.join(_DPMOTION_DIR, "data/test_texts_120.json")
    if os.path.exists(local_test_json):
        with open(local_test_json, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        print(f"[DATA] 从本地加载 {len(entries)} 条测试文本: {local_test_json}")
        rng = random.Random(seed)
        if len(entries) > num_samples:
            entries = rng.sample(entries, num_samples)
        print(f"[DATA] 选取 {len(entries)} 条评估文本")
        return entries
    
    # 回退到原始方式（从 InterGen 数据集加载）
    with open(split_file, 'r') as f:
        test_ids = [line.strip() for line in f if line.strip()]
    print(f"[DATA] 测试集共 {len(test_ids)} 个序列")

    entries = []
    for fid in test_ids:
        annot_path = os.path.join(annots_dir, f"{fid}.txt")
        if not os.path.exists(annot_path):
            continue
        with open(annot_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            continue
        text = lines[0]
        t1, t2 = split_interaction_text(text)
        entries.append({
            'file_id': fid, 'interaction_text': text, 'text1': t1, 'text2': t2,
        })

    rng = random.Random(seed)
    if len(entries) > num_samples:
        entries = rng.sample(entries, num_samples)
    print(f"[DATA] 选取 {len(entries)} 条评估文本")
    return entries

# ============================================================================
# 批量生成
# ============================================================================

def build_4person_inter_graph(
    motion_text_g1: str, motion_text_g2: str,
    non_interact_dist: float = DEFAULT_NON_INTERACT_DIST,
) -> Dict:
    dist_str = f">={non_interact_dist}m"
    return {
        'char_to_idx': {"P1": 0, "P2": 1, "P3": 2, "P4": 3},
        'idx_to_char': {0: "P1", 1: "P2", 2: "P3", 3: "P4"},
        'groups': {"Group1": [0, 1], "Group2": [2, 3]},
        'person_to_group': {},
        'distance_constraints': {
            "Group1": {'member_distances': None, 'group_distances': dist_str},
            "Group2": {'member_distances': None, 'group_distances': dist_str},
        },
        'Interaction_Definitions': [
            {'Interaction_id': 'Group1', 'Interaction_objects': ['P1', 'P2'],
             'interaction_type': 'Group', 'motion_text': motion_text_g1,
             'non-interactive_distances': dist_str},
            {'Interaction_id': 'Group2', 'Interaction_objects': ['P3', 'P4'],
             'interaction_type': 'Group', 'motion_text': motion_text_g2,
             'non-interactive_distances': dist_str},
        ],
    }


def generate_single_scene(
    model, inter_graph: Dict,
    motion_text_g1: str, motion_text_g2: str,
    window_size: int = DEFAULT_WINDOW, distance_weight: float = 5.0,
) -> np.ndarray:
    device = next(model.parameters()).device
    normalizer = MotionNormalizer()

    motion_cond_map = {}
    for gname, text in [("Group1", motion_text_g1), ("Group2", motion_text_g2)]:
        tmp = {"text": [text]}
        tmp = model.text_process(tmp)
        motion_cond_map[gname] = tmp["cond"].detach()

    batch = OrderedDict()
    batch["motion_lens"] = torch.zeros(1, 1, device=device).long()
    batch["motion_lens"][:] = window_size
    batch["text"] = [motion_text_g1]
    batch["motion_text_map"] = motion_cond_map
    batch["distance_weight"] = distance_weight
    batch["angle_weight"] = 0.0
    batch["inter_graph"] = inter_graph

    batch = model.text_process(batch)
    batch.update(model.decoder.forward_single(batch))

    outs = []
    for output in batch["output"]:
        outs.append(normalizer.backward(output.cpu().detach().numpy()))
    sequences = np.concatenate(outs, axis=0)

    joints3d = sequences[..., :22 * 3].reshape(sequences.shape[0], -1, 22, 3)
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(joints3d, 1, axis=1, mode='nearest')


def batch_generate(
    test_texts: List[Dict], kb: InteractionKnowledgeBase, output_dir: str,
    window_size: int = DEFAULT_WINDOW, non_interact_dist: float = DEFAULT_NON_INTERACT_DIST,
    distance_weight: float = 5.0, overwrite: bool = False,
) -> List[Dict]:
    os.makedirs(output_dir, exist_ok=True)
    model = load_model()
    records, total = [], len(test_texts)

    for idx, entry in enumerate(test_texts):
        fid = entry['file_id']
        npy_path = os.path.join(output_dir, f"scene_{fid}_4p.npy")

        if not overwrite and os.path.exists(npy_path):
            print(f"[{idx+1}/{total}] SKIP scene_{fid}_4p.npy")
            records.append({'index': idx, 'file_id': fid, 'npy_path': npy_path,
                            'text_g1': entry['interaction_text'], 'status': 'skipped'})
            continue

        text_g1 = entry['interaction_text']
        kb_results = kb.query(text_g1, top_k=20)
        text_g2 = text_g1
        for r in kb_results:
            if r['score'] < 0.90:
                text_g2 = r['interaction_text']
                break
        if text_g2 == text_g1 and len(kb_results) > 5:
            text_g2 = kb_results[5]['interaction_text']

        inter_graph = build_4person_inter_graph(text_g1, text_g2, non_interact_dist)
        print(f"[{idx+1}/{total}] 生成 scene_{fid}")

        t0 = time.time()
        try:
            joints3d = generate_single_scene(
                model, inter_graph, text_g1, text_g2,
                window_size=window_size, distance_weight=distance_weight)
            np.save(npy_path, joints3d)
            elapsed = time.time() - t0
            print(f"  OK ({elapsed:.1f}s)")
            records.append({'index': idx, 'file_id': fid, 'npy_path': npy_path,
                            'text_g1': text_g1, 'text_g2': text_g2,
                            'window': window_size, 'shape': list(joints3d.shape),
                            'elapsed_sec': round(elapsed, 2), 'status': 'ok'})
        except Exception as e:
            print(f"  FAIL: {e}")
            records.append({'index': idx, 'file_id': fid, 'text_g1': text_g1,
                            'error': str(e), 'status': 'failed'})

    log_path = os.path.join(output_dir, "generation_log.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    ok = sum(1 for r in records if r['status'] == 'ok')
    skip = sum(1 for r in records if r['status'] == 'skipped')
    fail = sum(1 for r in records if r['status'] == 'failed')
    print(f"[SUMMARY] 成功={ok} 跳过={skip} 失败={fail}")
    return records


# ============================================================================
# 评估
# ============================================================================

def evaluate_all(
    npy_dir: str,
    output_path: str,
    device: str = "cpu",
) -> Dict:
    import glob

    npy_files = sorted(glob.glob(os.path.join(npy_dir, "scene_*_4p.npy")))
    if not npy_files:
        print(f"[EVAL] 未找到 npy 文件: {npy_dir}/scene_*_4p.npy")
        return {}

    print(f"\n{'=' * 70}")
    print(f"DPMotion 评估: {len(npy_files)} 个场景")
    print(f"{'=' * 70}")

    interaction_pairs = [(0, 1), (2, 3)]
    non_interaction_pairs = [(0, 2), (0, 3), (1, 2), (1, 3)]

    per_scene = []
    # PeneBone 汇总
    all_penebone, all_pene_rate = [], []
    all_penebone_inter, all_pene_inter_rate = [], []
    all_penebone_non, all_pene_non_rate = [], []
    # CFrame / Contact 汇总
    all_cframe, all_contact = [], []
    all_cframe_inter, all_contact_inter = [], []
    all_cframe_non, all_contact_non = [], []
    # 场景内多样性
    all_diversity = []

    for fi, npy_path in enumerate(npy_files):
        scene_name = Path(npy_path).stem
        try:
            joints = np.load(npy_path)
        except Exception as e:
            print(f"  [{fi+1}] SKIP {scene_name}: {e}")
            continue
        if joints.ndim != 4 or joints.shape[0] != 4 or joints.shape[2] != 22:
            print(f"  [{fi+1}] SKIP {scene_name}: shape={joints.shape}")
            continue

        # --- PeneBone ---
        pene = compute_penebone_metrics(
            joints, joint_radius=0.02, device=device,
            interaction_pairs=interaction_pairs,
            non_interaction_pairs=non_interaction_pairs)

        # --- CFrame / Contact ---
        ct = compute_contact_metrics(
            joints,
            interaction_pairs=interaction_pairs,
            non_interaction_pairs=non_interaction_pairs,
            contact_threshold=0.10)

        # --- 场景内多样性 ---
        div = diversity_motion_embedding(joints)

        result = {
            'scene': scene_name,
            'num_frames': int(joints.shape[1]),
            'penebone': pene['penebone'],
            'pene_rate': pene['pene_rate'],
            'penebone_inter': pene['penebone_inter'],
            'pene_inter_rate': pene['pene_inter_rate'],
            'penebone_non': pene['penebone_non'],
            'pene_non_rate': pene['pene_non_rate'],
            'cframe_all': ct['cframe_all'],
            'contact_all': ct['contact_all'],
            'cframe_inter': ct['cframe_inter'],
            'contact_inter': ct['contact_inter'],
            'cframe_non': ct['cframe_non'],
            'contact_non': ct['contact_non'],
            'diversity': div,
        }
        per_scene.append(result)

        all_penebone.append(pene['penebone'])
        all_pene_rate.append(pene['pene_rate'])
        all_penebone_inter.append(pene['penebone_inter'])
        all_pene_inter_rate.append(pene['pene_inter_rate'])
        all_penebone_non.append(pene['penebone_non'])
        all_pene_non_rate.append(pene['pene_non_rate'])
        all_cframe.append(ct['cframe_all'])
        all_contact.append(ct['contact_all'])
        all_cframe_inter.append(ct['cframe_inter'])
        all_contact_inter.append(ct['contact_inter'])
        all_cframe_non.append(ct['cframe_non'])
        all_contact_non.append(ct['contact_non'])
        all_diversity.append(div)

        if (fi + 1) % 10 == 0 or fi == 0:
            print(f"  [{fi+1}/{len(npy_files)}] {scene_name}: "
                  f"PB={pene['penebone']:.3f} CF={ct['cframe_all']:.1f}% "
                  f"Ct={ct['contact_all']:.1f} Div={div:.3f}")

    if not per_scene:
        print("[EVAL] 无有效场景")
        return {}

    n = len(per_scene)
    _m = lambda a: float(np.mean([x for x in a if x == x])) if a else float('nan')
    _s = lambda a: float(np.std([x for x in a if x == x])) if a else float('nan')

    summary = {
        'num_scenes': n,
        'metrics': {
            'penebone_mean': _m(all_penebone),
            'penebone_std': _s(all_penebone),
            'pene_rate_mean': _m(all_pene_rate),
            'pene_rate_std': _s(all_pene_rate),
            'penebone_inter_mean': _m(all_penebone_inter),
            'pene_inter_rate_mean': _m(all_pene_inter_rate),
            'penebone_non_mean': _m(all_penebone_non),
            'pene_non_rate_mean': _m(all_pene_non_rate),
            'cframe_all_mean': _m(all_cframe),
            'cframe_all_std': _s(all_cframe),
            'contact_all_mean': _m(all_contact),
            'contact_all_std': _s(all_contact),
            'cframe_inter_mean': _m(all_cframe_inter),
            'contact_inter_mean': _m(all_contact_inter),
            'cframe_non_mean': _m(all_cframe_non),
            'contact_non_mean': _m(all_contact_non),
            'intra_scene_diversity_mean': _m(all_diversity),
            'intra_scene_diversity_std': _s(all_diversity),
        },
        'per_scene': per_scene,
    }

    m = summary['metrics']
    print(f"\n{'=' * 70}")
    print(f"评估汇总 ({n} 个场景)")
    print(f"{'=' * 70}")
    print(f"  PeneBone (骨骼穿透):")
    print(f"    总体穿透深度:    {m['penebone_mean']:.6f} ± {m['penebone_std']:.6f}")
    print(f"    总体穿透率:      {m['pene_rate_mean']:.6f} ± {m['pene_rate_std']:.6f}")
    print(f"    交互对穿透深度:  {m['penebone_inter_mean']:.6f}")
    print(f"    交互对穿透率:    {m['pene_inter_rate_mean']:.6f}")
    print(f"    非交互对穿透深度:{m['penebone_non_mean']:.6f}")
    print(f"    非交互对穿透率:  {m['pene_non_rate_mean']:.6f}")
    print(f"  CFrame (接触帧占比 %):")
    print(f"    总体:            {m['cframe_all_mean']:.2f}% ± {m['cframe_all_std']:.2f}%")
    print(f"    交互对:          {m['cframe_inter_mean']:.2f}%")
    print(f"    非交互对:        {m['cframe_non_mean']:.2f}%")
    print(f"  Contact (平均接触关节对数/帧):")
    print(f"    总体:            {m['contact_all_mean']:.4f} ± {m['contact_all_std']:.4f}")
    print(f"    交互对:          {m['contact_inter_mean']:.4f}")
    print(f"    非交互对:        {m['contact_non_mean']:.4f}")
    print(f"  场景内多样性:")
    print(f"    Diversity:       {m['intra_scene_diversity_mean']:.4f} ± {m['intra_scene_diversity_std']:.4f}")
    print(f"{'=' * 70}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[EVAL] 结果已保存: {output_path}")
    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DPMotion 批量生成 + 评估")
    parser.add_argument('--all', action='store_true', help='生成 + 评估')
    parser.add_argument('--generate-only', action='store_true', help='仅生成')
    parser.add_argument('--eval-only', action='store_true', help='仅评估')

    parser.add_argument('--num-samples', type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument('--window', type=int, default=DEFAULT_WINDOW)
    parser.add_argument('--non-interact-dist', type=float, default=DEFAULT_NON_INTERACT_DIST)
    parser.add_argument('--distance-weight', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--kb-path', type=str, default=KB_PATH)
    parser.add_argument('--annots-dir', type=str, default=ANNOTS_DIR)
    parser.add_argument('--split-file', type=str, default=SPLIT_FILE)

    parser.add_argument('--npy-dir', type=str,
                        default=os.path.join(_DPMOTION_DIR, 'results/dpmotion_eval'))
    parser.add_argument('--eval-output', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--eval-device', type=str, default='cpu')

    args = parser.parse_args()
    if not (args.all or args.generate_only or args.eval_only):
        args.all = True

    eval_output = args.eval_output or os.path.join(args.npy_dir, "eval_results.json")

    if args.all or args.generate_only:
        test_texts = load_test_texts(
            split_file=args.split_file, annots_dir=args.annots_dir,
            num_samples=args.num_samples, seed=args.seed)
        kb = InteractionKnowledgeBase()
        kb.load(args.kb_path)
        batch_generate(
            test_texts, kb, output_dir=args.npy_dir,
            window_size=args.window, non_interact_dist=args.non_interact_dist,
            distance_weight=args.distance_weight, overwrite=args.overwrite)

    if args.all or args.eval_only:
        evaluate_all(npy_dir=args.npy_dir, output_path=eval_output,
                     device=args.eval_device)


if __name__ == '__main__':
    main()
