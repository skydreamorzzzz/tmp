# DPMotion 复现记录

状态：120/120 个 4-person scenes 已生成并通过完整性检查（无失败、无重复/缺失 ID、所有 NPY shape 与 finite 检查通过）；已由仓库 evaluator 完成正式评测。

## 实际 pipeline

固定仓库 `code/DPMotion` commit `36dafe8b2cefa8f4610723fb10f581e380548498`。使用现有 InterGen checkpoint，346 个 state-dict keys strict load 全通过；262 维 global mean/std 与官方 InterGen 逐元素相同。没有训练或改动模型源码。

每个 test 描述作为 Group1 文本，知识库从全部 InterHuman annotation 构建并用缓存 ViT-L/14@336 CLIP 向量检索 top-20。选择第一个 cosine similarity `<0.90` 的描述作为 Group2；若 top-20 都不满足，取 rank 6。当前 120 query 中 87 个命中阈值、33 个走 rank-6 fallback。代码没有解释 top-k=20 的选择依据；本次固定照用，不搜索超参。知识库包含 test annotations；120 条 query 中有 107 条 top-1 与 query 原文完全相同（2636 是一例），另 13 条 top-1 是高相似释义。阈值会跳过近重复文本，但候选池仍有 test 文本泄漏风险。

几个实际选择例子如下；“所选候选”是第二组文本，不等于 top-1。完整 top-20 和分数见 [retrieval examples](../artifacts/dpmotion_reproduction/rag/retrieval_examples.json) 及 `rag/retrievals.jsonl`。

| Test ID | Group1 query | 候选选择过程 | 最终 Group2 |
|---|---|---|---|
| 2636 | one person sidesteps to the right and the other person advances ahead. | top-1 是原 query，0.99983，被跳过；rank 2–20 均未低于阈值，按 fallback 取 rank 6，0.92380 | the first one moves ahead by running, and the other person moves forward by walking. |
| 5647 | two persons lift slightly and swing both hands upwards from the front of their bodies. | top-1 是原 query，0.99944；rank 2 为 0.90547；rank 3 首次低于阈值，0.89309 | the two persons raise both arms and swing up and down and twist their waist to the left and right while stepping up and down in place. |
| 2813 | two individuals are conversing with each other. | top-1 是近乎相同表述，0.98583；前 15 条未低于阈值，rank 16 首次满足，0.89860 | the two persons approach each other closely. |
| 4000 | two persons stroll on the street and offer assistance in pushing the cart. | top-1 是原 query，浮点相似度 1.00005；rank 2/3 是高相似改写；rank 4 首次满足，0.77188 | two people are taking a walk, and one picks up an item belonging to the first one. |

生成器按 Group1、Group2 顺序联合生成两组双人动作；当前真实路径是 1,000-step ancestral DDPM（配置文件的 `DDIM50` 未被 `forward_single → p_sample_loop_multi` 使用）。在 `t≤900` 调用 group/person loss；当前 graph 仅启用跨组中心与四个跨组成员 root 的距离项（`member_distances=None`、`person_to_group={}`，因此 intra-group/Person/orientation 子项不执行）。`t≤100` 还对每个双人 Group pair 加 `0.6×TopologyBatch` GLI。`angle_weight` 路径不在 120-case graph 中生效。详细调用链和证据见 [工程记录](REPRO_ENGINEERING_NOTES.md)。

## 结果

| 来源 / protocol | PeneBone | penetration rate | inter-pair penetration rate | non-interaction penetration rate | intra-scene diversity | cross-scene diversity |
|---|---:|---:|---:|---:|---:|---:|
| DPMotion README 声称的 120 场景 | 0.3414 ± 1.2919 | 0.0060%* | 0.0181%* | 0% | 3.3386 ± 0.5514 | 1.5876* |
| 本次 DPMotion RAG + guidance，120 场景 | 0.308329 ± 1.078201 | 0.005379%（raw 5.379e-5） | 0.016138%（raw 1.614e-4） | 0% | 3.272061 ± 0.562350 | 未计算 |

`*` README 的 cross-scene diversity 在当前源码没有计算函数/定义。penetration rate 当前 evaluator 的 raw 输出是 0–1 比例；表中百分比是 raw×100 的等价值，README 没记录换算口径。以 [完整 metrics JSON](../artifacts/dpmotion_reproduction/metrics/repo_metrics_full.json) 和 [README 差值 JSON](../artifacts/dpmotion_reproduction/metrics/readme_comparison.json) 为准。仓库 evaluator 的 PeneBone 将每帧跨人骨段距离小于0.04m的深度求和，penetration rate 统计低于该阈值的骨段对比例；interaction pairs 为 (0,1),(2,3)，non-interaction 为其余四对。intra-scene diversity 是四人的轻量轨迹/选定关节速度统计 embedding 两两欧氏距离均值。额外 CFrame/Contact 用 0.10m 关节中心阈值，单列为 repo diagnostics。

本次与 README 的差值：PeneBone mean `−0.033071`，std `−0.213699`；按百分比解释 README 速率时，总 penetration rate `−0.000621` 个百分点、interaction rate `−0.001962` 个百分点；intra-scene diversity mean `−0.066539`、std `+0.010950`。这些差值同时受未记录 RNG、README 的 rate 单位和 std 聚合定义不清影响。

仓库 evaluator 额外诊断（不是论文摘要指标）：CFrame 总体 `8.8778% ± 19.0458%`，interaction `8.8778%`、non-interaction `0%`；Contact 总体 `0.308833 ± 1.020821` 接触关节对/帧，interaction `0.308833`、non-interaction `0`。两者基于 0.10 m 关节中心距离，不是 mesh contact protocol。

DPMotion 论文公开摘要将方法称为 training-free inference，列出 Retrieval-augmented Knowledge Base、Unified Group-level Separation 和 Proxemics-guided Distance-based Constraints；并报告 multi-group InterHuman benchmark 的 Top-1 R-Precision `0.386`、FID `5.642`。当前 DPMotion `dpmotion_eval.py` 不计算这两项；公开摘要没有给出完整 evaluator/protocol，工作区也没有论文全文，因此不能将其与本次仓库碰撞/多样性 evaluator 直接比较。

| 来源 / protocol | Top-1 R-Precision | FID | 本次能否严格复算 |
|---|---:|---:|---|
| DPMotion 论文公开摘要 | 0.386 | 5.642 | 否：公开摘要无完整 protocol，当前 repo evaluator 不实现 |
| 本次 120-scene DPMotion repo eval | 未计算 | 未计算 | 当前 `dpmotion_eval.py` 仅计算几何碰撞、接触和轻量多样性 |

论文来源：[Springer chapter page](https://link.springer.com/chapter/10.1007/978-981-92-3438-7_39)。本次 DPMotion 仓库只有一个初始 commit（2026-03-02），早于论文 2026-07-14 online date；因此无法证明该代码快照就是最终发表版实验代码。

## 机制 smoke（test id 2636）

固定 test text、RAG text、seed `2678`、150 frames、DDPM1000、距离 3 m、distance weight 5；A/B/C/D 分别是无 RAG 无 guidance、仅 RAG、仅 guidance、RAG+guidance。四份 motion 均为 `(4,150,22,3)`。相同随机 seed 下 A→B、A→C、A→D 的平均绝对关节坐标差分别为 `0.561m`、`0.412m`、`1.042m`；仅比较 B→D，guidance 改变输出 `0.498m`。这是单 case 机制验证，不作为正式质量指标。官方 evaluator 对 D/2636 的 smoke 结果为 PeneBone=0、penetration rate=0、CFrame=0、Contact=0、intra-scene diversity=4.2168；该值只验证指标管线可读写，不能与120场均值比较。

RAG+guidance trace 中 t=900 group-center distance 约 `2.665m`、distance gradient L2 约 `0.62`；t=700 中心距约 `4.85m`，已高于 3m 下限。t=100 到 t=0 GLI 原始 loss 和总梯度仍非零。见 [完整 trace](../artifacts/dpmotion_reproduction/guidance/D_rag_plus_guidance_scene_2636.json)。

## Guidance on/off 成对核查

为隔离 guidance，固定四个已有检索样例（覆盖 rank-6 fallback、rank 3、rank 16、低相似 rank 4），每个 case 保持 Group1/Group2 原文、seed=`42+file_id`、窗口150、DDPM1000、距离3m、weight5 不变，只把 guidance 梯度置零或启用。驱动器捕获了每次 `torch.randn` / `randn_like` 输出；每个 case 的 on/off 两边 2,009 个随机张量摘要逐项相同，因此初始噪声及采样中的随机噪声一致。no-guidance 是 runner 中将 guidance 梯度替换为零的消融，不是上游提供的 CLI 开关。

| ID | 检索选择 | 组中心 XZ 均距 off→on (m) | 中心距<3m 帧占比 off→on | PeneBone off→on (m深度和) | 穿透率 off→on | joints 平均绝对变化 (m) |
|---|---|---:|---:|---:|---:|---:|
| 2636 | rank6 fallback, sim 0.9238 | 2.410→4.619 | 54%→0% | 0.782→0 | 0.01260%→0 | 0.498 |
| 5647 | rank3 threshold, sim 0.8931 | 0.542→3.436 | 100%→0% | 5.717→0 | 0.09146%→0 | 0.632 |
| 2813 | rank16 threshold, sim 0.8986 | 1.442→3.865 | 100%→0% | 1.288→0 | 0.02016%→0 | 0.480 |
| 4000 | rank4 threshold, sim 0.7719 | 0.722→3.316 | 100%→0% | 0→0 | 0→0 | 0.673 |

这四个固定例子中，guidance on 的组中心距都超过3m，跨组骨段穿透深度和仓库 contact-frame rate 均为0；off 条件平均有88.5%的帧组中心距小于3m。组内动作也发生变化：四例平均 joint XYZ 速度从 `0.01598` 降至 `0.01527 m/frame`，但变化不是统一刚体平移，组内两人 root 距离有升有降（逐组值、每人路径、穿透率和配对差均见 [metrics summary](../artifacts/dpmotion_reproduction/ablations/guidance_pairs/metrics_summary.json)）。这支持“当前 guidance 能拉开这四个被选 case 并降低这些几何指标”，不证明自然度或语义质量改善，也不是随机抽样的120场景效果估计。原始成对 `.npy`、随机噪声 manifest、逐步 trace 和执行日志保存在 `artifacts/dpmotion_reproduction/ablations/guidance_pairs/`、`guidance_pairs/` 与 `logs/guidance_pairs.log`。

2636 的重新捕获输出与先前四路 smoke 中对应的 RAG/no-guidance、RAG+guidance `.npy` 逐元素完全一致（max abs diff=`0`）。

## 可视化观察

[2636 的 A（无 RAG / 无 guidance）](../artifacts/dpmotion_reproduction/visualizations/2636_A_intergen_no_rag_no_guidance.gif)、[B（仅 RAG）](../artifacts/dpmotion_reproduction/visualizations/2636_B_rag_no_guidance.gif)、[C（仅 guidance）](../artifacts/dpmotion_reproduction/visualizations/2636_C_no_rag_guidance.gif)、[D（RAG + guidance）](../artifacts/dpmotion_reproduction/visualizations/2636_D_rag_plus_guidance.gif) 是本次完整 2×2 的同 case 可视化。肉眼可见 guidance 开启后两组更分开；仅 RAG 会改变布局，但不能保证分离。骨架语义不易确认，个别 limbs 仍有重叠；这只是定性观察，不作自然度结论。[5647 D](../artifacts/dpmotion_reproduction/visualizations/5647_D_rag_plus_guidance.gif) 为额外 case。

## 协议限制

本次对 README 的直接复现只比较相同 120 条 test case、相同窗口 150、距离 3m、weight 5、去噪后 Gaussian smoothing（sigma=1）。代码未固定原作者随机状态；本次显式按 `42 + file_id` 固定 Python/NumPy/Torch/CUDA 随机种子。README 结果不能逐样本重现保证。环境依赖版本低于 DPMotion requirements 声明版本，实际运行通过 smoke 验证；具体环境、seed、文件 hash 和命令保存在 `artifacts/dpmotion_reproduction/`。

## 2026-10-08：完整 RAG × guidance 2×2 消融

固定 120 个 test scenes，seed=`42+file_id`，每场150帧，原生 InterGen `forward_single`、1000-step ancestral DDPM、距离3m、distance weight 5。no-RAG 时 Group2 使用 Group1 测试描述；RAG 时 Group2 使用固定检索候选。四臂每个 case 都重置同一 seed。没有训练或改模型/参数。

| 条件 | PeneBone depth sum (m, mean ± std) | penetration rate (raw fraction) | CFrame (%) | Contact (pairs/frame) | intra-scene diversity | Group2 InterCLIP matching distance ↓ | paired cosine ↑ | R-Precision R1 / R2 / R3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A InterGen | 19.273467 ± 20.106620 | 0.003003 | 75.422222 | 9.014222 | 1.414787 ± 0.666466 | 3.872774 | 0.597097 | 0.317 / 0.500 / 0.608 |
| B +RAG | 10.930947 ± 14.672645 | 0.001754 | 66.527778 | 5.578778 | 1.530702 ± 0.708177 | 3.884032 | 0.594617 | 0.475 / 0.567 / 0.658 |
| C +guidance | 0.207924 ± 0.824690 | 0.000037 | 6.722222 | 0.163889 | 3.295393 ± 0.525749 | 3.905605 | 0.590184 | 0.217 / 0.367 / 0.475 |
| D +RAG +guidance | 0.308329 ± 1.078201 | 0.000054 | 8.877778 | 0.308833 | 3.272061 ± 0.562350 | 3.913358 | 0.588469 | 0.317 / 0.467 / 0.558 |

### 配对效应

InterCLIP Group2 的差值按同一 scene 配对；距离差 `<0` 表示 RAG/处理组更接近其条件文本。几何指标的正负方向按指标含义解释，下面报告效应量和 bootstrap 95% 区间。

| 指标 / contrast | mean Δ | 95% bootstrap CI | per-scene directional fraction |
|---|---:|---:|---:|
| InterCLIP distance: RAG effect, guidance off | 0.011258 | [-0.021914, 0.044266] | 48.3% lower |
| InterCLIP distance: RAG effect, guidance on | 0.007754 | [-0.027780, 0.043876] | 48.3% lower |
| InterCLIP distance: guidance effect, RAG off | 0.032830 | [0.018557, 0.048055] | 35.8% lower |
| InterCLIP distance: guidance effect, RAG on | 0.029326 | [0.016075, 0.042942] | 35.0% lower |

### 几何指标的 factorial effects

每个 effect 均为同场景配对差；RAG main effect 是 guidance off/on 两个 RAG 差值的平均，guidance main effect 同理，interaction 是两种 guidance 下 RAG 差值之差。CI 为 scene bootstrap 95%。

| 指标 | RAG main Δ (95% CI) | guidance main Δ (95% CI) | RAG×guidance Δ (95% CI) |
|---|---:|---:|---:|
| penebone | -4.121057 [-6.020578; -2.164799] | -14.844080 [-17.497806; -12.589581] | 8.442925 [4.525396; 12.245171] |
| pene_rate | -0.000616 [-0.000896; -0.000339] | -0.002333 [-0.002738; -0.001982] | 0.001267 [0.000710; 0.001812] |
| cframe_all | -3.369444 [-6.025069; -0.727292] | -63.175000 [-67.608611; -58.791389] | 11.050000 [5.555556; 16.911806] |
| contact_all | -1.645250 [-2.241940; -1.039352] | -7.060139 [-7.994535; -6.182784] | 3.580389 [2.465106; 4.731428] |
| diversity | 0.046291 [-0.008227; 0.102710] | 1.810982 [1.741573; 1.880742] | -0.139247 [-0.229184; -0.056370] |

### RAG 的正向信号与限制

在本次固定 120 条 Group2 文本/动作集合的 InterCLIP 120-way 配对 rank 诊断中，R@1 的逐 scene 命中率变化为：guidance off `+0.158`（bootstrap 95% CI [0.042, 0.275]），guidance on `+0.100`（[0.000, 0.200]）。这是当前实验中支持 RAG 的正向信号，但它只表示各条件自己的 120-way 文本/动作候选池中正配对排名改善；R@2/R@3 区间跨 0。
直接的 Group2 matching distance 并未改善：RAG 使均值分别增加 `+0.0113` 和 `+0.0078`（两者 95% CI 均跨 0），且只有 48.3% scene 的距离下降。因此不能把结果概括成普遍的文本动作对齐提升。KB 与 test split 重叠；120 个 query 有 107 个 top-1 原文命中。该 R@1 信号不等于无泄漏泛化收益，也不是论文完整 multi-group R-Precision protocol。


InterCLIP Group2 是 RAG 敏感的文本动作对齐诊断；120-way R-Precision 使用各条件自己的文本/动作集合，不等同论文完整 multi-group benchmark。几何 evaluator 的 PeneBone、penetration rate 和 intra-scene diversity 按仓库定义逐 scene 配对；CFrame/Contact 是仓库附加诊断。Δ>0 表示数值增加，不统一代表变好。完整 per-case 数值、所有 simple/main/interaction factorial contrasts 和 evaluator 中间特征见 `artifacts/dpmotion_reproduction/ablations/full_2x2/`。

**数据边界：** RAG 知识库由全部 InterHuman annotations 构建，包含 test split；这是已证实的检索库/test 重叠。当前 120 query 中有 107 条 top-1 与 query 原文相同。此消融能说明在该泄漏设置下 RAG 的成对效应，不能作为无泄漏泛化收益证据。no-RAG 的 Group2 复用 Group1 文本，因此这是“替换为检索文本”的因果对照；它不会隔离出独立于文本差异的抽象检索机制。

四臂最终是否出现正向 RAG 作用，以表中的 paired InterCLIP 和 geometry 结果为准；不因单个指标的有利方向宣称整体质量提高。
分层看几何指标，RAG 在 guidance off 时降低了 PeneBone `−8.343`（95% CI [−12.207, −4.430]）、penetration rate `−0.001249`（[−0.001803, −0.000689]）、CFrame `−8.894` 个百分点和 Contact `−3.435` pairs/frame；guidance on 时这些 RAG 简单效应的区间均跨 0。RAG×guidance interaction 区间在这些指标上不跨 0，表明 RAG 的几何变化依赖 guidance 是否开启。
**最可信结论：** 在此有 test overlap 的设置中，RAG 提高了本地 120-way InterCLIP R@1，并在 guidance off 时改善仓库几何穿透诊断；但 guidance on 下几何效应不明确，直接 matching distance 没改善，组内多样性变化也不稳定。把它表述为“检索文本改变了组2，并在部分当前诊断上有正向信号”是可信的；称为整体生成质量提升或无泄漏泛化证据则超出数据支持。
