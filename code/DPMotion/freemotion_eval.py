#!/usr/bin/env python3
"""
FreeMotion 基线 — 使用 FreeMotion 代码生成 4 人动作并计算评估指标。

注意：FreeMotion 官方未发布 checkpoint（epoch999.ckpt），本脚本使用 InterGen
checkpoint 部分加载（主分支 282 keys 匹配，control_branch 220 keys 随机初始化）。

不修改 FreeMotion-Codes-main 中的任何文件。

用法：
    # 生成 + 评估
    python freemotion_eval.py --all

    # 仅评估已有结果
    python freemotion_eval.py --eval-only
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
# 路径
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FREEMOTION_DIR = "/projects/wbl/FreeMotion-Codes-main"

# DPMotion 自身（评估指标 + 工具）
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# FreeMotion（模型代码，只读引用）
if _FREEMOTION_DIR not in sys.path:
    sys.path.insert(0, _FREEMOTION_DIR)

# ---------------------------------------------------------------------------
# 评估指标（来自 DPMotion/eval/）
# ---------------------------------------------------------------------------
_EVAL_DIR = os.path.join(_SCRIPT_DIR, "eval")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
from metrics_penebone import compute_penebone_metrics
from metrics_contact import compute_contact_metrics
from metrics_core import diversity_motion_embedding

# 复用 DPMotion 的文本拆分和知识库
sys.path.insert(0, _SCRIPT_DIR)
from DPMotion import InteractionKnowledgeBase, split_interaction_text

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
FPS = 30
DEFAULT_WINDOW = 150
DEFAULT_NUM_SAMPLES = 120
ANNOTS_DIR = "/projects/wbl/InterGen/data/annots"
SPLIT_FILE = "/projects/wbl/InterGen/data/split/test.txt"
KB_PATH = os.path.join(_SCRIPT_DIR, "dpmotion_kb.pkl")
INTERGEN_CKPT = "/projects/wbl/multi-person-interaction/checkpoints/intergen.ckpt"
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "results/freemotion_eval")


# ============================================================================
# 加载 FreeMotion 模型（用 InterGen 权重部分初始化）
# ============================================================================

def load_freemotion_model(device: str = "cuda:0"):
    """加载 FreeMotion InterGenSpatialControlNet，用 InterGen ckpt 部分填充。"""
    saved_cwd = os.getcwd()
    saved_path = sys.path[:]

    # 临时让 FreeMotion 的模块优先于 DPMotion 自带的同名模块
    for p in list(sys.path):
        if p == _SCRIPT_DIR:
            sys.path.remove(p)
    if _FREEMOTION_DIR not in sys.path:
        sys.path.insert(0, _FREEMOTION_DIR)

    # 清除已缓存的冲突模块
    mods_to_purge = [k for k in sys.modules if k.startswith(('models', 'configs', 'utils', 'common'))]
    saved_mods = {k: sys.modules.pop(k) for k in mods_to_purge}

    os.chdir(_FREEMOTION_DIR)

    from configs import get_config
    from models.intergen import InterGenSpatialControlNet

    cfg = get_config("configs/model_inter.yaml")
    model = InterGenSpatialControlNet(cfg)

    ckpt = torch.load(INTERGEN_CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    mapped_sd = {}
    for k, v in sd.items():
        clean = k.replace("model.", "", 1)
        if clean in model.state_dict():
            mapped_sd[clean] = v
        else:
            alt = clean.replace("decoder.net.", "decoder.net.net.")
            if alt in model.state_dict():
                mapped_sd[alt] = v

    missing, unexpected = model.load_state_dict(mapped_sd, strict=False)
    loaded = len(mapped_sd)
    total = len(model.state_dict())
    print(f"[FreeMotion] 加载 {loaded}/{total} 权重 "
          f"(缺失 {len(missing)} keys, 主要是 control_branch)")

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()

    # 恢复模块和路径
    for k in [k for k in sys.modules if k.startswith(('models', 'configs', 'utils', 'common'))]:
        sys.modules.pop(k, None)
    sys.modules.update(saved_mods)
    sys.path[:] = saved_path
    os.chdir(saved_cwd)
    return model, dev


# ============================================================================
# pipeline_generate (复制自 FreeMotion evaluator，不修改原文件)
# ============================================================================

def pipeline_generate_4person(model, motion_lens, texts, text_multi_person, device):
    """
    仿照 FreeMotion evaluator.pipeline_generate 生成 4 人动作。

    Args:
        model: InterGenSpatialControlNet
        motion_lens: torch.Tensor (1,1) 帧数
        texts: [text1, text2, text3, text4] 每人动作文本
        text_multi_person: [scene_text] 场景描述
        device: torch device

    Returns:
        output: (1, T, 4, 262) tensor
    """
    saved_cwd = os.getcwd()
    os.chdir(_FREEMOTION_DIR)

    def generate_one(motion_lens, text, text_mp, person_num=1,  # noqa
                     motion_guidance=None):
        batch = {
            "prompt": list(text),
            "text": list(text),
            "text_multi_person": list(text_mp),
            "person_num": person_num,
            "motion_lens": motion_lens,
            "motion_guidance": motion_guidance,
            "spatial_condition": None,
        }
        batch = model.forward_test(batch)
        output = batch["output"].reshape(
            batch["output"].shape[0], batch["output"].shape[1], 1, -1
        )
        return output

    generated = []
    motion = generate_one(motion_lens.to(device), texts[0], text_multi_person)
    generated.append(motion)

    for pi, txt in enumerate(texts[1:]):
        tmp = [m.detach().float().to(device) for m in generated]
        guidance = torch.cat(tmp, dim=-2)
        motion = generate_one(
            motion_lens.to(device), txt, text_multi_person,
            person_num=pi + 2, motion_guidance=guidance,
        )
        generated.append(motion)

    os.chdir(saved_cwd)
    return torch.cat(generated, dim=-2)


# ============================================================================
# 数据准备（复用 DPMotion 的逻辑）
# ============================================================================

def load_test_texts(num_samples=DEFAULT_NUM_SAMPLES, seed=42):
    # 优先使用本地预处理的测试数据
    local_test_json = os.path.join(_SCRIPT_DIR, "data/test_texts_120.json")
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
    with open(SPLIT_FILE, 'r') as f:
        test_ids = [l.strip() for l in f if l.strip()]

    entries = []
    for fid in test_ids:
        ap = os.path.join(ANNOTS_DIR, f"{fid}.txt")
        if not os.path.exists(ap):
            continue
        with open(ap, 'r', encoding='utf-8', errors='replace') as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            continue
        text = lines[0]
        t1, t2 = split_interaction_text(text)
        entries.append({
            'file_id': fid, 'interaction_text': text,
            'text1': t1, 'text2': t2,
        })

    rng = random.Random(seed)
    if len(entries) > num_samples:
        entries = rng.sample(entries, num_samples)
    print(f"[DATA] 选取 {len(entries)} 条评估文本")
    return entries


# ============================================================================
# 批量生成
# ============================================================================

def batch_generate(
    test_texts: List[Dict],
    kb: InteractionKnowledgeBase,
    output_dir: str,
    window_size: int = DEFAULT_WINDOW,
    device_id: str = "cuda:0",
    overwrite: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    model, device = load_freemotion_model(device_id)

    # 使用 FreeMotion 的 normalizer（进入其 cwd 读取 data/global_*.npy）
    saved_cwd2 = os.getcwd()
    os.chdir(_FREEMOTION_DIR)
    saved_path2 = sys.path[:]
    for p in list(sys.path):
        if p == _SCRIPT_DIR:
            sys.path.remove(p)
    sys.path.insert(0, _FREEMOTION_DIR)
    mods2 = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith(('utils',))}
    from utils.utils import MotionNormalizerTorch
    normalizer = MotionNormalizerTorch()
    for k in [k for k in sys.modules if k.startswith(('utils',))]:
        sys.modules.pop(k, None)
    sys.modules.update(mods2)
    sys.path[:] = saved_path2
    os.chdir(saved_cwd2)

    records, total = [], len(test_texts)
    motion_lens = torch.tensor([[window_size]]).long()

    for idx, entry in enumerate(test_texts):
        fid = entry['file_id']
        npy_path = os.path.join(output_dir, f"scene_{fid}_4p.npy")

        if not overwrite and os.path.exists(npy_path):
            print(f"[{idx+1}/{total}] SKIP scene_{fid}_4p.npy")
            records.append({'file_id': fid, 'status': 'skipped'})
            continue

        text_g1 = entry['interaction_text']
        t1, t2 = entry['text1'], entry['text2']

        kb_results = kb.query(text_g1, top_k=20)
        text_g2, t3, t4 = text_g1, t1, t2
        for r in kb_results:
            if r['score'] < 0.90:
                text_g2 = r['interaction_text']
                t3, t4 = r['text1'], r['text2']
                break
        if text_g2 == text_g1 and len(kb_results) > 5:
            text_g2 = kb_results[5]['interaction_text']
            t3, t4 = kb_results[5]['text1'], kb_results[5]['text2']

        texts = [[t1], [t2], [t3], [t4]]
        text_mp = [text_g1]

        print(f"[{idx+1}/{total}] FreeMotion scene_{fid}")
        t0 = time.time()
        try:
            with torch.no_grad():
                output = pipeline_generate_4person(
                    model, motion_lens, texts, text_mp, device
                )
            output_np = normalizer.backward(output).cpu().numpy()
            joints = output_np[0, :, :, :22*3].transpose(1, 0, 2)
            joints = joints.reshape(4, -1, 22, 3)

            from scipy.ndimage import gaussian_filter1d
            joints = gaussian_filter1d(joints, 1, axis=1, mode='nearest')

            np.save(npy_path, joints)
            elapsed = time.time() - t0
            print(f"  OK ({elapsed:.1f}s) shape={joints.shape}")
            records.append({
                'file_id': fid, 'text_g1': text_g1, 'text_g2': text_g2,
                'shape': list(joints.shape), 'elapsed_sec': round(elapsed, 2),
                'status': 'ok',
            })
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback; traceback.print_exc()
            records.append({'file_id': fid, 'error': str(e), 'status': 'failed'})

    log_path = os.path.join(output_dir, "generation_log.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in records if r['status'] == 'ok')
    fail = sum(1 for r in records if r['status'] == 'failed')
    print(f"[SUMMARY] 成功={ok} 失败={fail}")
    return records


# ============================================================================
# 评估（与 dpmotion_eval.py 相同指标）
# ============================================================================

def evaluate_all(npy_dir: str, output_path: str, device: str = "cpu"):
    import glob

    npy_files = sorted(glob.glob(os.path.join(npy_dir, "scene_*_4p.npy")))
    if not npy_files:
        print(f"[EVAL] 无 npy 文件: {npy_dir}")
        return {}

    print(f"\n{'='*70}")
    print(f"FreeMotion 评估: {len(npy_files)} 个场景")
    print(f"{'='*70}")

    interaction_pairs = [(0, 1), (2, 3)]
    non_interaction_pairs = [(0, 2), (0, 3), (1, 2), (1, 3)]

    per_scene = []
    all_pb, all_pr, all_pb_i, all_pr_i, all_pb_n, all_pr_n = ([] for _ in range(6))
    all_cf, all_ct, all_cf_i, all_ct_i, all_cf_n, all_ct_n = ([] for _ in range(6))
    all_div = []

    for fi, f in enumerate(npy_files):
        name = Path(f).stem
        try:
            joints = np.load(f)
        except Exception as e:
            print(f"  SKIP {name}: {e}")
            continue
        if joints.ndim != 4 or joints.shape[0] != 4 or joints.shape[2] != 22:
            print(f"  SKIP {name}: shape={joints.shape}")
            continue

        pene = compute_penebone_metrics(
            joints, joint_radius=0.02, device=device,
            interaction_pairs=interaction_pairs,
            non_interaction_pairs=non_interaction_pairs)
        ct = compute_contact_metrics(
            joints, interaction_pairs=interaction_pairs,
            non_interaction_pairs=non_interaction_pairs, contact_threshold=0.10)
        div = diversity_motion_embedding(joints)

        r = {
            'scene': name, 'num_frames': int(joints.shape[1]),
            'penebone': pene['penebone'], 'pene_rate': pene['pene_rate'],
            'penebone_inter': pene['penebone_inter'], 'pene_inter_rate': pene['pene_inter_rate'],
            'penebone_non': pene['penebone_non'], 'pene_non_rate': pene['pene_non_rate'],
            'cframe_all': ct['cframe_all'], 'contact_all': ct['contact_all'],
            'cframe_inter': ct['cframe_inter'], 'contact_inter': ct['contact_inter'],
            'cframe_non': ct['cframe_non'], 'contact_non': ct['contact_non'],
            'diversity': div,
        }
        per_scene.append(r)
        all_pb.append(r['penebone']); all_pr.append(r['pene_rate'])
        all_pb_i.append(r['penebone_inter']); all_pr_i.append(r['pene_inter_rate'])
        all_pb_n.append(r['penebone_non']); all_pr_n.append(r['pene_non_rate'])
        all_cf.append(r['cframe_all']); all_ct.append(r['contact_all'])
        all_cf_i.append(r['cframe_inter']); all_ct_i.append(r['contact_inter'])
        all_cf_n.append(r['cframe_non']); all_ct_n.append(r['contact_non'])
        all_div.append(div)

        if (fi+1) % 10 == 0 or fi == 0:
            print(f"  [{fi+1}/{len(npy_files)}] {name}: "
                  f"PB={r['penebone']:.3f} CF={r['cframe_all']:.1f}% Div={div:.3f}")

    if not per_scene:
        return {}

    n = len(per_scene)
    _m = lambda a: float(np.mean([x for x in a if x==x])) if a else float('nan')
    _s = lambda a: float(np.std([x for x in a if x==x])) if a else float('nan')

    summary = {
        'num_scenes': n,
        'metrics': {
            'penebone_mean': _m(all_pb), 'penebone_std': _s(all_pb),
            'pene_rate_mean': _m(all_pr), 'pene_rate_std': _s(all_pr),
            'penebone_inter_mean': _m(all_pb_i), 'pene_inter_rate_mean': _m(all_pr_i),
            'penebone_non_mean': _m(all_pb_n), 'pene_non_rate_mean': _m(all_pr_n),
            'cframe_all_mean': _m(all_cf), 'cframe_all_std': _s(all_cf),
            'contact_all_mean': _m(all_ct), 'contact_all_std': _s(all_ct),
            'cframe_inter_mean': _m(all_cf_i), 'contact_inter_mean': _m(all_ct_i),
            'cframe_non_mean': _m(all_cf_n), 'contact_non_mean': _m(all_ct_n),
            'intra_scene_diversity_mean': _m(all_div), 'intra_scene_diversity_std': _s(all_div),
        },
        'per_scene': per_scene,
    }

    m = summary['metrics']
    print(f"\n{'='*70}")
    print(f"FreeMotion 评估汇总 ({n} 个场景)")
    print(f"{'='*70}")
    print(f"  PeneBone:     {m['penebone_mean']:.6f} ± {m['penebone_std']:.6f}")
    print(f"  穿透率:       {m['pene_rate_mean']:.6f} ± {m['pene_rate_std']:.6f}")
    print(f"  交互对PB:     {m['penebone_inter_mean']:.6f}  非交互对PB: {m['penebone_non_mean']:.6f}")
    print(f"  CFrame:       {m['cframe_all_mean']:.2f}% ± {m['cframe_all_std']:.2f}%")
    print(f"  交互对CF:     {m['cframe_inter_mean']:.2f}%  非交互对CF: {m['cframe_non_mean']:.2f}%")
    print(f"  Contact:      {m['contact_all_mean']:.4f} ± {m['contact_all_std']:.4f}")
    print(f"  交互对Ct:     {m['contact_inter_mean']:.4f}  非交互对Ct: {m['contact_non_mean']:.4f}")
    print(f"  场景内多样性: {m['intra_scene_diversity_mean']:.4f} ± {m['intra_scene_diversity_std']:.4f}")
    print(f"{'='*70}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[EVAL] 结果: {output_path}")
    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FreeMotion 基线评估")
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--num-samples', type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument('--window', type=int, default=DEFAULT_WINDOW)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--npy-dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()
    if not (args.all or args.generate_only or args.eval_only):
        args.all = True

    device_id = f"cuda:{args.gpu}"
    eval_output = os.path.join(args.npy_dir, "eval_results.json")

    if args.all or args.generate_only:
        texts = load_test_texts(num_samples=args.num_samples, seed=args.seed)
        kb = InteractionKnowledgeBase()
        kb.load(KB_PATH)
        batch_generate(texts, kb, args.npy_dir, args.window, device_id, args.overwrite)

    if args.all or args.eval_only:
        evaluate_all(args.npy_dir, eval_output)


if __name__ == '__main__':
    main()
