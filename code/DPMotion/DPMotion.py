#!/usr/bin/env python3
"""
DPMotion — Direct Prompt Motion Generation

简化的多人动作生成管线：
  1. 从 InterHuman 双人交互数据集构建 RAG 知识库
  2. 用普通大模型（LLM）一次性生成所有交互组的动作描述、不相关组间距离和朝向
  3. 仅考虑一种距离约束：不相关交互群体之间的最小分隔距离
  4. 调用 InterGen 扩散模型执行动作生成，辅以梯度引导

与原 Agent Loop 方案的区别：
  - 不使用迭代式 Agent 对话循环，而是由 LLM 一次性输出完整场景配置
  - 不建模身份/角色信息，只关注 **动作内容 + 群体间距离 + 朝向**
  - 距离约束仅保留不相关交互群体之间的最小隔离距离（non-interactive distance）

用法：
    # Step 1: 构建知识库（仅需运行一次）
    python DPMotion.py build-kb --annots-dir /projects/wbl/InterGen/data/annots

    # Step 2: 为场景生成配置（LLM 一次性输出）
    python DPMotion.py generate \\
        --prompt "A scene with 3 groups: boxing, dancing, and tai-chi" \\
        --num-groups 3

    # Step 3: 生成动作（扩散模型 + 距离引导）
    python DPMotion.py run --config scene_config.json

    # 一步到位：从 prompt 直接到动作
    python DPMotion.py end2end \\
        --prompt "Two pairs of people: one pair is boxing, the other is dancing" \\
        --num-groups 2 --window 200
"""

import copy
import os
import sys
import re
import json
import pickle
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch as th
from torch import nn

# ---------------------------------------------------------------------------
# 项目根目录设置（所有依赖均在本目录下）
# ---------------------------------------------------------------------------
_DPMOTION_DIR = os.path.dirname(os.path.abspath(__file__))
if _DPMOTION_DIR not in sys.path:
    sys.path.insert(0, _DPMOTION_DIR)
os.chdir(_DPMOTION_DIR)

from configs import get_config
from models.intergen import InterGen
from utils.utils import MotionNormalizer, MotionNormalizerTorch


# ============================================================================
# Part 1: 交互文本拆分（将双人描述拆成两个单人描述）
# ============================================================================

_P2 = r'(?:the second|the other)\s*(?:person|one|guy|individual|man|woman)?'
_P1 = r'(?:the first|one)\s*(?:person|one|guy|individual|man|woman)?'

_SPLIT_PATTERNS = [
    re.compile(r'(' + _P1 + r'\b.+?)[.,;]\s*(' + _P2 + r'\b.+)', re.I),
    re.compile(r'(' + _P1 + r'\b.+?)\s*(?:while|,\s*while|and|,\s*and|as|,\s*as)\s+('
               + _P2 + r'\b.+)', re.I),
    re.compile(r'(one\s*(?:person)?\b.+?)[.,;]\s*(the other\s*(?:person)?\b.+)', re.I),
    re.compile(r'(one\s*(?:person)?\b.+?)\s*(?:while|and|as)\s+(the other\s*(?:person)?\b.+)',
               re.I),
]


def split_interaction_text(text: str) -> Tuple[str, str]:
    """将双人交互文本拆分成两段单人描述。"""
    text = text.strip()
    for pat in _SPLIT_PATTERNS:
        m = pat.search(text)
        if m:
            t1 = m.group(1).strip().rstrip('.,;')
            t2 = m.group(2).strip().rstrip('.,;')
            if len(t1) > 5 and len(t2) > 5:
                return t1, t2
    return text, text


# ============================================================================
# Part 2: RAG 知识库 — 从 InterHuman 双人交互数据集构建
# ============================================================================

class InteractionKnowledgeBase:
    """
    基于 CLIP 嵌入的 RAG 知识库。

    每条记录包含：
      - interaction_text: 原始双人交互描述
      - text1 / text2: 拆分后的单人描述
      - CLIP embedding（用于语义检索）
    """

    def __init__(self):
        self.entries: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self._clip_model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def build_from_annots(
        self,
        annots_dir: str,
        dedup_sim_threshold: float = 0.95,
    ):
        """从 annots 目录构建知识库（自动去重）。"""
        import clip

        print(f"[KB] 从 {annots_dir} 读取标注文件...")
        files = sorted(
            [f for f in os.listdir(annots_dir) if f.endswith('.txt')],
            key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0,
        )
        print(f"[KB] 找到 {len(files)} 个标注文件")

        raw_texts, raw_t1, raw_t2, raw_fids = [], [], [], []
        seen_normalized = set()
        for fname in files:
            fpath = os.path.join(annots_dir, fname)
            fid = fname.replace('.txt', '')
            with open(fpath, encoding='utf-8', errors='replace') as f:
                lines = [l.strip() for l in f if l.strip()]
            for line in lines:
                norm = line.lower().strip()
                if norm in seen_normalized:
                    continue
                seen_normalized.add(norm)
                t1, t2 = split_interaction_text(line)
                raw_texts.append(line)
                raw_t1.append(t1)
                raw_t2.append(t2)
                raw_fids.append(fid)

        print(f"[KB] 精确去重后: {len(raw_texts)} 条文本")

        if self._clip_model is None:
            print("[KB] 加载 CLIP 模型 (ViT-L/14@336px)...")
            self._clip_model, _ = clip.load("ViT-L/14@336px", device=self._device, jit=False)
            self._clip_model.eval()

        print("[KB] CLIP 编码中...")
        all_embs = []
        batch_size = 64
        for i in range(0, len(raw_texts), batch_size):
            batch = raw_texts[i:i + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(self._device)
            with torch.no_grad():
                feats = self._clip_model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embs.append(feats.cpu().float().numpy())
        all_embs = np.concatenate(all_embs, axis=0)

        kept_indices = []
        kept_embs = []
        for j in range(len(raw_texts)):
            if kept_embs:
                sims = np.dot(np.array(kept_embs), all_embs[j])
                if sims.max() > dedup_sim_threshold:
                    continue
            kept_indices.append(j)
            kept_embs.append(all_embs[j])

        self.entries = []
        for j in kept_indices:
            self.entries.append({
                'interaction_text': raw_texts[j],
                'text1': raw_t1[j],
                'text2': raw_t2[j],
                'file_id': raw_fids[j],
            })
        self.embeddings = np.array(kept_embs)

        print(f"[KB] 嵌入去重后: {len(self.entries)} 条 (阈值={dedup_sim_threshold})")

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'entries': self.entries, 'embeddings': self.embeddings}, f)
        print(f"[KB] 已保存至 {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.entries = data['entries']
        self.embeddings = data['embeddings']
        print(f"[KB] 已加载: {len(self.entries)} 条记录 ({path})")

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """语义检索 top-k 条目。"""
        import clip

        if self._clip_model is None:
            self._clip_model, _ = clip.load("ViT-L/14@336px", device=self._device, jit=False)
            self._clip_model.eval()

        tokens = clip.tokenize([query_text], truncate=True).to(self._device)
        with torch.no_grad():
            q_emb = self._clip_model.encode_text(tokens)
            q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)
        q_emb = q_emb.cpu().float().numpy()
        scores = (self.embeddings @ q_emb.T).squeeze(-1)

        results = []
        for idx in np.argsort(scores)[::-1][:top_k]:
            entry = self.entries[idx]
            results.append({
                'score': float(scores[idx]),
                'interaction_text': entry['interaction_text'],
                'text1': entry['text1'],
                'text2': entry['text2'],
                'file_id': entry['file_id'],
            })
        return results

    def query_batch(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        return [self.query(q, top_k) for q in queries]


# ============================================================================
# Part 3: LLM 直接生成（替代 Agent Loop）
# ============================================================================

_PROXEMICS_KNOWLEDGE = """\
## Proxemics Reference (Hall 1966):
- INTIMATE zone  (0.00-0.45m): body contact — wrestling, hugging, slow couple dance
- PERSONAL zone  (0.45-1.20m): arm's reach — boxing, handshake, pushing, fast partner dance
- SOCIAL zone    (1.20-3.70m): conversational — talking, group exercise, coordinated motion
- PUBLIC zone    (3.70m+)    : non-interacting — audience, bystander, background action

## Non-Interactive Group Separation Guidelines:
- For combat scenes:    groups should be >= 3.5m apart
- For dance/exercise:   groups should be >= 2.5m apart
- For casual activities: groups should be >= 2.0m apart
"""


def build_scene_prompt(
    user_prompt: str,
    num_groups: int,
    kb_references: List[List[Dict]],
) -> Tuple[str, str]:
    ref_block = ""
    for gi, refs in enumerate(kb_references):
        ref_block += f"\n  Group {gi + 1} reference examples:\n"
        for r in refs[:3]:
            ref_block += f"    - {r['interaction_text']} (text1: {r['text1']}, text2: {r['text2']})\n"

    system_msg = f"""\
You are an expert in human motion synthesis.
Given a scene description, generate a complete multi-group interaction configuration.

{_PROXEMICS_KNOWLEDGE}

IMPORTANT RULES:
- Only output action content (what people do physically), distance between groups, and group orientation.
- Do NOT include identity, names, roles, or personality information.
- Each group consists of exactly 2 people performing a paired interaction.
- "non_interactive_distance" is the minimum separation (meters) between ANY two unrelated groups.
- "orientation" describes how the group faces relative to the scene center."""

    user_msg = f"""\
Scene description: {user_prompt}
Number of interaction groups: {num_groups}

Knowledge base references (use as inspiration, do NOT copy directly):
{ref_block}

Generate a JSON object with this EXACT structure (no markdown fences, pure JSON):
{{
  "groups": [
    {{
      "motion_text": "<two-person interaction description>",
      "text1": "<what person 1 does>",
      "text2": "<what person 2 does>",
      "orientation": "<facing direction relative to scene>"
    }}
  ],
  "non_interactive_distance": <float, minimum meters between unrelated groups>
}}

Generate exactly {num_groups} groups. Every motion_text must describe a physically plausible two-person interaction."""

    return system_msg, user_msg


# ---------------------------------------------------------------------------
# 免费 LLM API 预设配置
#   1. 智谱 GLM-4-Flash  (完全免费，注册即用)  https://open.bigmodel.cn
#   2. 硅基流动 SiliconFlow (免费额度)          https://cloud.siliconflow.cn
# ---------------------------------------------------------------------------
LLM_PRESETS = {
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
        "env_key": "ZHIPU_API_KEY",
        "doc": "https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "env_key": "SILICONFLOW_API_KEY",
        "doc": "https://cloud.siliconflow.cn/account/ak",
    },
}
DEFAULT_LLM_PROVIDER = "zhipu"


def _resolve_llm_config(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> Tuple[str, str, str]:
    """根据参数 / 环境变量自动选择可用的免费 LLM 后端。"""
    if api_key and base_url and model:
        return api_key, base_url, model

    prov = provider or DEFAULT_LLM_PROVIDER
    if prov in LLM_PRESETS:
        preset = LLM_PRESETS[prov]
        resolved_key = api_key or os.environ.get(preset["env_key"], "")
        resolved_url = base_url or preset["base_url"]
        resolved_model = model or preset["model"]
        if resolved_key:
            return resolved_key, resolved_url, resolved_model

    for name, preset in LLM_PRESETS.items():
        key = os.environ.get(preset["env_key"], "")
        if key:
            print(f"[LLM] 自动选择 {name} ({preset['model']})")
            return key, preset["base_url"], preset["model"]

    doc_urls = "\n".join(f"  {n}: {p['doc']}" for n, p in LLM_PRESETS.items())
    raise ValueError(
        f"未找到可用的 LLM API Key。请设置以下任一环境变量:\n"
        f"  export ZHIPU_API_KEY=xxx        (推荐，完全免费)\n"
        f"  export SILICONFLOW_API_KEY=xxx\n"
        f"API Key 申请地址:\n{doc_urls}"
    )


def call_llm(
    system_msg: str,
    user_msg: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    temperature: float = 0.4,
    max_tokens: int = 1000,
) -> str:
    """调用免费 LLM API（智谱/SiliconFlow，OpenAI 兼容协议）。"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装 openai: pip install openai>=1.0")

    key, url, mdl = _resolve_llm_config(api_key, base_url, model, provider)
    client = OpenAI(api_key=key, base_url=url)
    resp = client.chat.completions.create(
        model=mdl,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def parse_llm_scene_config(content: str) -> Dict:
    for src in [
        content,
        (re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", content) or
         type("", (), {"group": lambda *_: None})()).group(1) or "",
        (re.search(r"\{[\s\S]+\}", content) or
         type("", (), {"group": lambda *_: None})()).group(0) or "",
    ]:
        if not src:
            continue
        try:
            obj = json.loads(src)
            if isinstance(obj, dict) and 'groups' in obj:
                return obj
        except (json.JSONDecodeError, TypeError):
            pass

    raise ValueError(f"无法从 LLM 输出中解析 JSON:\n{content[:500]}")


def generate_scene_config(
    user_prompt: str,
    num_groups: int,
    kb: InteractionKnowledgeBase,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    top_k_refs: int = 5,
) -> Dict:
    keywords = re.findall(r'\b(?:box|danc|fight|kick|push|hug|wave|tai.?chi|'
                          r'wrestl|spar|sword|greet|exercis|walk|run|jump)\w*',
                          user_prompt, re.I)
    if not keywords:
        keywords = [user_prompt]

    kb_refs = []
    for gi in range(num_groups):
        query = keywords[gi % len(keywords)] if keywords else user_prompt
        refs = kb.query(query, top_k=top_k_refs)
        kb_refs.append(refs)

    system_msg, user_msg = build_scene_prompt(user_prompt, num_groups, kb_refs)
    raw_output = call_llm(
        system_msg, user_msg,
        api_key=api_key, base_url=base_url, model=model, provider=provider,
    )
    config = parse_llm_scene_config(raw_output)

    if len(config.get('groups', [])) != num_groups:
        print(f"[WARN] LLM 返回 {len(config.get('groups', []))} 组，期望 {num_groups} 组")

    if 'non_interactive_distance' not in config:
        config['non_interactive_distance'] = 3.0

    return config


# ============================================================================
# Part 4: 简化的距离引导 — 仅考虑不相关群体间距离
# ============================================================================

def compute_distance_xz(root1: th.Tensor, root2: th.Tensor) -> th.Tensor:
    rel = root1 - root2
    rel_xz = rel[..., [0, 2]]
    return th.norm(rel_xz, dim=-1)


def compute_forward_direction(joints: th.Tensor) -> th.Tensor:
    j0 = joints[..., 0, :]
    j13 = joints[..., 13, :]
    j14 = joints[..., 14, :]
    vec1 = j13 - j0
    vec2 = j14 - j13
    forward = th.cross(vec1, vec2, dim=-1)
    forward_xz = forward[..., [0, 2]]
    return forward_xz / (th.norm(forward_xz, dim=-1, keepdim=True) + 1e-8)


def compute_non_interactive_loss(
    all_starts: List[th.Tensor],
    group_indices: List[List[int]],
    min_distance: float,
) -> th.Tensor:
    device = all_starts[0].device
    loss = th.tensor(0.0, device=device, dtype=all_starts[0].dtype)
    n_pairs = 0

    for gi in range(len(group_indices)):
        for gj in range(gi + 1, len(group_indices)):
            roots_i = []
            for idx in group_indices[gi]:
                B, T = all_starts[idx].shape[:2]
                joints = all_starts[idx][..., :22 * 3].reshape(B, T, 22, 3)
                roots_i.append(joints[..., 0, :])
            center_i = th.stack(roots_i, dim=0).mean(dim=0)

            roots_j = []
            for idx in group_indices[gj]:
                B, T = all_starts[idx].shape[:2]
                joints = all_starts[idx][..., :22 * 3].reshape(B, T, 22, 3)
                roots_j.append(joints[..., 0, :])
            center_j = th.stack(roots_j, dim=0).mean(dim=0)

            dist = compute_distance_xz(center_i, center_j)
            violation = th.clamp(min_distance - dist, min=0.0)
            loss = loss + (th.exp(violation) - 1.0).mean()
            n_pairs += 1

            for ii in group_indices[gi]:
                for jj in group_indices[gj]:
                    r_i = all_starts[ii][..., :3]
                    r_j = all_starts[jj][..., :3]
                    d = compute_distance_xz(r_i, r_j)
                    v = th.clamp(min_distance - d, min=0.0)
                    loss = loss + (th.exp(v) - 1.0).mean()
                    n_pairs += 1

    if n_pairs > 0:
        loss = loss / n_pairs
    return loss


def compute_orientation_loss(
    all_starts: List[th.Tensor],
    group_indices: List[List[int]],
    orientations: List[str],
    scene_center: Optional[th.Tensor] = None,
) -> th.Tensor:
    device = all_starts[0].device
    loss = th.tensor(0.0, device=device, dtype=all_starts[0].dtype)

    if scene_center is None:
        all_roots = [s[..., :3] for s in all_starts]
        scene_center = th.stack(all_roots, dim=0).mean(dim=0)

    n_terms = 0
    for gi, indices in enumerate(group_indices):
        orient = orientations[gi] if gi < len(orientations) else "facing_center"
        if "facing_center" not in orient.lower().replace(" ", "_"):
            continue

        for idx in indices:
            B, T = all_starts[idx].shape[:2]
            joints = all_starts[idx][..., :22 * 3].reshape(B, T, 22, 3)
            person_forward = compute_forward_direction(joints)
            person_root = all_starts[idx][..., :3]
            to_center = scene_center - person_root
            to_center_xz = to_center[..., [0, 2]]
            to_center_xz = to_center_xz / (th.norm(to_center_xz, dim=-1, keepdim=True) + 1e-6)

            dot = (person_forward * to_center_xz).sum(dim=-1)
            angle = th.acos(th.clamp(dot, -1.0, 1.0))
            loss = loss + angle.mean()
            n_terms += 1

    if n_terms > 0:
        loss = loss / n_terms
    return loss


# ============================================================================
# Part 5: DPMotion 生成器
# ============================================================================

class DPMotionGenerator(nn.Module):

    def __init__(self, model, distance_weight: float = 5.0):
        super().__init__()
        self.model = model
        self.distance_weight = distance_weight
        self.normalizer = MotionNormalizer()
        self.normalizer_torch = MotionNormalizerTorch()

    def generate_from_config(
        self,
        scene_config: Dict,
        window_size: int = 200,
        output_dir: str = "results/dpmotion",
        output_name: str = "dpmotion_scene",
    ) -> np.ndarray:
        self.model.eval()
        device = next(self.model.parameters()).device
        groups = scene_config['groups']
        non_interact_dist = float(scene_config.get('non_interactive_distance', 3.0))
        n_groups = len(groups)
        total_people = n_groups * 2

        print(f"\nDPMotion 生成: {n_groups} 组交互, {total_people} 人, "
              f"群体间距>={non_interact_dist:.1f}m")

        inter_graph = self._build_inter_graph(groups, non_interact_dist)

        motion_cond_map = {}
        for gi, group in enumerate(groups):
            group_name = f"Group{gi + 1}"
            text = group.get('motion_text', '')
            if text:
                tmp_batch = {"text": [text]}
                tmp_batch = self.model.text_process(tmp_batch)
                motion_cond_map[group_name] = tmp_batch["cond"].detach()

        batch = OrderedDict()
        batch["motion_lens"] = torch.zeros(1, 1, device=device).long()
        batch["text"] = [groups[0].get('motion_text', 'two people interacting')]
        batch["motion_text_map"] = motion_cond_map
        batch["distance_weight"] = self.distance_weight
        batch["angle_weight"] = 0.0
        batch["motion_lens"][:] = window_size
        batch["inter_graph"] = inter_graph

        batch = self.model.text_process(batch)
        batch.update(self.model.decoder.forward_single(batch))

        motion_output = batch["output"]
        outs = []
        for output in motion_output:
            out = self.normalizer.backward(output.cpu().detach().numpy())
            outs.append(out)
        sequences = np.concatenate(outs, axis=0)

        joints3d = sequences[..., :22 * 3].reshape(sequences.shape[0], -1, 22, 3)
        from scipy.ndimage import gaussian_filter1d
        joints3d = gaussian_filter1d(joints3d, 1, axis=1, mode='nearest')

        os.makedirs(output_dir, exist_ok=True)
        npy_path = os.path.join(output_dir, f"{output_name}_{total_people}p.npy")
        np.save(npy_path, joints3d)
        print(f"[OK] 关节数据: {npy_path}")

        feat_path = os.path.join(output_dir, f"{output_name}_{total_people}p_feat262.npy")
        np.save(feat_path, sequences)

        config_path = os.path.join(output_dir, f"{output_name}_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            safe_config = {k: v for k, v in scene_config.items() if not k.startswith('_')}
            json.dump(safe_config, f, ensure_ascii=False, indent=2)

        return joints3d

    def _build_inter_graph(self, groups: List[Dict], non_interact_dist: float) -> Dict:
        total_people = len(groups) * 2
        char_to_idx = {f"P{i + 1}": i for i in range(total_people)}
        idx_to_char = {i: f"P{i + 1}" for i in range(total_people)}

        group_map = {}
        interaction_defs = []
        distance_constraints = {}

        for gi, group in enumerate(groups):
            group_name = f"Group{gi + 1}"
            idx1 = gi * 2
            idx2 = gi * 2 + 1
            group_map[group_name] = [idx1, idx2]
            non_interact_str = f">={non_interact_dist}m"

            interaction_defs.append({
                'Interaction_id': group_name,
                'Interaction_objects': [f"P{idx1 + 1}", f"P{idx2 + 1}"],
                'interaction_type': 'Group',
                'motion_text': group.get('motion_text', ''),
                'non-interactive_distances': non_interact_str,
            })
            distance_constraints[group_name] = {
                'member_distances': None,
                'group_distances': non_interact_str,
            }

        return {
            'char_to_idx': char_to_idx,
            'idx_to_char': idx_to_char,
            'groups': group_map,
            'person_to_group': {},
            'distance_constraints': distance_constraints,
            'Interaction_Definitions': interaction_defs,
        }


# ============================================================================
# Part 6: 模型加载
# ============================================================================

def load_model(
    model_cfg_path: str = "configs/model.yaml",
    device: str = "cuda:0",
):
    model_cfg = get_config(model_cfg_path)
    model = InterGen(model_cfg)

    if model_cfg.CHECKPOINT:
        ckpt_path = model_cfg.CHECKPOINT
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join("/projects/wbl/InterGen", model_cfg.CHECKPOINT)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[MODEL] checkpoint: {ckpt_path}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model


# ============================================================================
# Part 7: CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DPMotion — 直接提示词驱动的多人动作生成"
    )
    subparsers = parser.add_subparsers(dest='command')

    p_build = subparsers.add_parser('build-kb', help='构建 RAG 知识库')
    p_build.add_argument('--annots-dir', type=str,
                         default='/projects/wbl/InterGen/data/annots')
    p_build.add_argument('--save-path', type=str, default='./dpmotion_kb.pkl')
    p_build.add_argument('--dedup-threshold', type=float, default=0.95)

    p_gen = subparsers.add_parser('generate', help='LLM 生成场景配置')
    p_gen.add_argument('--prompt', type=str, required=True)
    p_gen.add_argument('--num-groups', type=int, default=2)
    p_gen.add_argument('--kb-path', type=str, default='./dpmotion_kb.pkl')
    p_gen.add_argument('--provider', type=str, default=None,
                       choices=list(LLM_PRESETS.keys()),
                       help='LLM 后端 (默认自动选择)')
    p_gen.add_argument('--api-key', type=str, default=None)
    p_gen.add_argument('--output', type=str, default='scene_config.json')

    p_run = subparsers.add_parser('run', help='从配置文件生成动作')
    p_run.add_argument('--config', type=str, required=True)
    p_run.add_argument('--window', type=int, default=200)
    p_run.add_argument('--out-dir', type=str, default='results/dpmotion')
    p_run.add_argument('--name', type=str, default='dpmotion_scene')
    p_run.add_argument('--distance-weight', type=float, default=5.0)

    p_e2e = subparsers.add_parser('end2end', help='端到端: prompt -> 动作')
    p_e2e.add_argument('--prompt', type=str, required=True)
    p_e2e.add_argument('--num-groups', type=int, default=2)
    p_e2e.add_argument('--kb-path', type=str, default='./dpmotion_kb.pkl')
    p_e2e.add_argument('--provider', type=str, default=None,
                       choices=list(LLM_PRESETS.keys()))
    p_e2e.add_argument('--api-key', type=str, default=None)
    p_e2e.add_argument('--window', type=int, default=200)
    p_e2e.add_argument('--out-dir', type=str, default='results/dpmotion')
    p_e2e.add_argument('--name', type=str, default='dpmotion_scene')
    p_e2e.add_argument('--distance-weight', type=float, default=5.0)

    p_query = subparsers.add_parser('query-kb', help='查询知识库')
    p_query.add_argument('--query', type=str, required=True)
    p_query.add_argument('--kb-path', type=str, default='./dpmotion_kb.pkl')
    p_query.add_argument('--top-k', type=int, default=5)

    args = parser.parse_args()

    if args.command == 'build-kb':
        kb = InteractionKnowledgeBase()
        kb.build_from_annots(args.annots_dir, dedup_sim_threshold=args.dedup_threshold)
        kb.save(args.save_path)
        split_ok = sum(1 for e in kb.entries if e['text1'] != e['text2'])
        print(f"[STATS] 文本拆分成功率: {split_ok}/{len(kb.entries)} "
              f"({100 * split_ok / max(len(kb.entries), 1):.1f}%)")

    elif args.command == 'generate':
        kb = InteractionKnowledgeBase()
        kb.load(args.kb_path)
        config = generate_scene_config(
            args.prompt, args.num_groups, kb,
            api_key=args.api_key, provider=getattr(args, 'provider', None),
        )
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"[OK] 配置已保存: {args.output}")

    elif args.command == 'run':
        with open(args.config, 'r', encoding='utf-8') as f:
            scene_config = json.load(f)
        model = load_model()
        generator = DPMotionGenerator(model, distance_weight=args.distance_weight)
        generator.generate_from_config(
            scene_config, window_size=args.window,
            output_dir=args.out_dir, output_name=args.name,
        )

    elif args.command == 'end2end':
        kb = InteractionKnowledgeBase()
        kb.load(args.kb_path)
        scene_config = generate_scene_config(
            args.prompt, args.num_groups, kb,
            api_key=args.api_key, provider=getattr(args, 'provider', None),
        )
        model = load_model()
        generator = DPMotionGenerator(model, distance_weight=args.distance_weight)
        generator.generate_from_config(
            scene_config, window_size=args.window,
            output_dir=args.out_dir, output_name=args.name,
        )

    elif args.command == 'query-kb':
        kb = InteractionKnowledgeBase()
        kb.load(args.kb_path)
        results = kb.query(args.query, top_k=args.top_k)
        print(f"\n查询: {args.query}\n{'=' * 60}")
        for r in results:
            sf = "split" if r['text1'] != r['text2'] else "unsplit"
            print(f"  score={r['score']:.4f} | {sf} | file={r['file_id']}")
            print(f"    {r['interaction_text']}")
            print(f"    text1: {r['text1']}")
            print(f"    text2: {r['text2']}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
