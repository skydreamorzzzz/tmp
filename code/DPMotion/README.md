# DPMotion — Direct Prompt Motion Generation

简化的多人动作生成框架：用大模型一次性生成场景描述，扩散模型生成多人交互动作。

## 核心思路

| 维度 | 原 Agent Loop 方案 | DPMotion |
|------|-------------------|----------|
| 文本生成 | Agent 多轮迭代对话 | LLM **一次性**输出完整场景 |
| 距离约束 | 5 种（组内/组间/Person-Group/…） | **仅 1 种**：不相关群体间距离 |
| 身份建模 | 角色 ID + 角色描述 | **无身份**，仅动作内容 + 距离 + 朝向 |
| 知识库 | 分散多处 | 统一 RAG（CLIP 嵌入 + InterHuman） |

## 项目结构

```
DPMotion/
├── DPMotion.py              # 核心框架（知识库 / LLM 生成 / 距离引导 / 扩散生成）
├── dpmotion_eval.py         # 批量生成 + 评估脚本
├── dpmotion_kb.pkl          # 预构建 RAG 知识库 (19429 条)
├── requirements.txt
├── README.md
├── configs/                 # 模型配置
│   ├── __init__.py
│   └── model.yaml
├── models/                  # InterGen 扩散模型
│   ├── intergen.py
│   ├── gaussian_diffusion.py
│   ├── nets.py
│   └── ...
├── utils/                   # 工具函数
│   ├── utils.py
│   └── rotation_conversions.py
├── common/                  # 四元数运算
│   └── quaternion.py
├── data/                    # 归一化参数
│   ├── global_mean.npy
│   └── global_std.npy
├── checkpoints/             # 模型权重 (需自行放置)
│   └── intergen.ckpt
├── eval/                    # 评估指标
│   ├── metrics_penebone.py  # PeneBone 碰撞率
│   └── metrics_core.py      # 动作多样性
└── results/                 # 生成结果
    └── dpmotion_eval/
        ├── scene_*_4p.npy   # 120 个 4 人场景关节坐标
        └── eval_results.json
```

## 环境安装

```bash
pip install -r requirements.txt
```

## 模型权重

将 InterGen 模型检查点放置到 `checkpoints/intergen.ckpt`。

## LLM API 配置

DPMotion 使用**免费开源大模型 API**生成场景描述。支持以下后端（二选一）：

### 方式一：智谱 GLM-4-Flash（推荐，完全免费）

1. 访问 https://open.bigmodel.cn 注册账号
2. 在 https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys 创建 API Key
3. 设置环境变量：

```bash
export ZHIPU_API_KEY=你的API_Key
```

### 方式二：硅基流动 SiliconFlow（免费额度）

1. 访问 https://cloud.siliconflow.cn 注册账号
2. 在 https://cloud.siliconflow.cn/account/ak 创建 API Key
3. 设置环境变量：

```bash
export SILICONFLOW_API_KEY=你的API_Key
```

## 使用方法

### 1. 构建知识库（仅需运行一次，已预构建）

```bash
python DPMotion.py build-kb --annots-dir /path/to/InterHuman/annots
```

### 2. 查询知识库

```bash
python DPMotion.py query-kb --query "two people boxing"
```

### 3. LLM 生成场景配置

```bash
python DPMotion.py generate \
    --prompt "Two pairs: one pair boxing, the other dancing" \
    --num-groups 2
```

### 4. 生成动作

```bash
python DPMotion.py run --config scene_config.json --window 200
```

### 5. 端到端生成（LLM + 扩散模型）

```bash
python DPMotion.py end2end \
    --prompt "Two pairs: one pair boxing, the other dancing" \
    --num-groups 2 --window 200
```

### 6. 批量评估（120 条 InterHuman 测试文本）

```bash
# 生成 + 评估一步到位
python dpmotion_eval.py --all --num-samples 120 --window 150

# 仅评估已有结果
python dpmotion_eval.py --eval-only --npy-dir results/dpmotion_eval
```

## 评估指标

| 指标 | 说明 |
|------|------|
| PeneBone | 骨骼穿透深度总和（越低越好） |
| Penetration Rate | 穿透率（越低越好） |
| Intra-scene Diversity | 场景内 4 人动作多样性（越高越好） |
| Cross-scene Diversity | 跨场景动作多样性（越高越好） |

### 120 场景评估结果

| 指标 | 值 |
|------|-----|
| PeneBone (mean±std) | 0.3414 ± 1.2919 |
| Penetration Rate | 0.0060% |
| 交互对穿透率 | 0.0181% |
| 非交互对穿透率 | 0.0000% |
| 场景内多样性 | 3.3386 ± 0.5514 |
| 跨场景多样性 | 1.5876 |

## 技术细节

- **知识库**：从 InterHuman 7897 个双人交互标注构建，CLIP 嵌入去重后 19429 条
- **LLM 生成**：注入近体空间学知识（Hall 1966），一次性输出所有组的动作描述 + 距离 + 朝向
- **距离引导**：仅约束不相关群体间最小隔离距离，在扩散采样中通过梯度引导实现
- **扩散模型**：InterGen (1000 步 DDPM)，支持 Group 交互生成
