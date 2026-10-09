# DPMotion 复现工程记录

持续追加。只记录实际运行遇到的问题或由当前源码/论文直接证实的差异。

## 证据分层（阅读本记录时）

- **实际发生的工程故障**：运行中真实报错或阻塞，记录症状、定位与修复。例如 guidance trace 的双人维度误判，以及 Matplotlib 3.2 的 GIF writer 故障。修复只在复现/诊断脚本，不改 DPMotion 模型路径。
- **代码证实的实现事实**：由固定 commit 的真实调用链、参数和输入 graph 直接确认，例如 1000-step ancestral DDPM、`t≤900` 距离 guidance、`t≤100` GLI、检索库读入所有 annotation、PeneBone 的几何计算。此类事实说明当前代码会做什么，不自动证明作者有意如此，也不证明它就是最终论文版。
- **尚未确认的假设/解释**：仓库没有说明的 top-k/阈值动机、README 与论文表格之间的映射、RAG 是否提升生成质量、GLI 是有意保留还是历史残留，以及论文最终实验是否使用相同 snapshot。此类事项只列为待核实问题，不写成结论。

已确认的 RAG 结果目前只证明检索文本进入条件并能改变固定样例输出；尚无 120-scene 的 RAG/no-RAG 配对质量评估，因此不据此宣称 RAG 有正向效果。test annotation 与检索库重叠属于代码和实际 top-k 记录可证实的数据边界问题；阈值会跳过高相似候选，但 33 个 rank-6 fallback 仍高于 0.90。

## 2026-10-07：仓库和复现协议考古

| 项目 | 表现 / 证据 | 根因或处理 | 对算法语义 / 论文比较的影响 | 面试讨论价值 |
|---|---|---|---|---|
| 工作区不是 Git 根目录 | 当前工作目录包含多个独立工程；`git status` 在根目录报 “not a git repository”。 | 单独固定 `code/DPMotion`、`code/InterGen-official`、`code/FreeMotion-Codes` 的 commit/status。 | 避免把邻接工程改动误认为 DPMotion 源码；不改变算法。 | 高：多仓库实验的 provenance。 |
| DPMotion 权重与 normalizer 未随仓库提供 | `configs/model.yaml` 默认 `checkpoints/intergen.ckpt`，仓库无 checkpoint 和 `data/global_{mean,std}.npy`。 | 复用现有 `assets/intergen/intergen.ckpt`；将归一化路径链接到官方 InterGen 文件。验证 346/346 keys、0 missing/unexpected/shape mismatch strict load；mean/std 的 262 维逐元素一致，SHA256 分别 `2f97b888…03b6` / `dd5988b9…3e4b`。 | 文件复用不改变归一化；strict-load 已通过。 | 高：预训练权重/归一化共同定义模型输入分布。 |
| 代码加载没有严格检查权重 | `DPMotion.load_model` 调用 `load_state_dict(..., strict=False)`，不报告 missing/unexpected keys。 | 复现 runner 将记录 keys、shape，并先尝试 strict load；若失败再逐项审计，不静默吞错。 | strict 成功则无差异；若只能兼容加载，需要按实际结果说明。 | 高：`strict=False` 可能把架构不兼容伪装为成功。 |
| 仓库没有预构建 RAG pickle | README 称 `dpmotion_kb.pkl` 已预构建，但 git 忽略 `*.pkl`，克隆目录里没有该文件。 | 计划从本机 InterHuman annotations 和缓存 CLIP 重建，保存输入清单、参数和 hash。 | 依赖 dedup 顺序/CLIP 权重与官方资产一致；重建内容之后与 README 数量核对。 | 高：发布说明与 Git 可复现资产不一致。 |
| 论文全文与 README 的基准不是同一证据 | Springer 公开页面摘要报告 Top-1 R-Precision 0.386、FID 5.642；仓库 README 报告 PeneBone/penetration/diversity。完整章节不在本地，publisher 页面显示订阅内容。 | 分别标记“论文摘要声称”和“仓库 README 数值”；不把 README 诊断指标冒充论文表格。持续尝试只用可访问的一手公开材料确认协议。 | 论文 Top-1/FID 的 multi-group evaluator/protocol 尚不能从摘要复现；指标不可直接比较。 | 高：证据等级和评测 protocol 的区分。 |
| README 与有效采样路径不完全一致 | README 称只保留 unrelated-group distance；`dpmotion_eval.py → forward_single → p_sample_loop_multi → p_sample_with_grad → gradients` 在 `t≤900` 调用 Group/Person loss，且 `t≤100` 对每个 Group pair 额外加 `0.6×TopologyBatch(GLI)`。 | 固定评测入口逐调用核对。精确哪些项在当前 graph 启用，见复现报告；正在用运行日志核实梯度。 | 若 GLI 梯度非零，README 对采样方法的简化会漏报实际启用项。 | 高：源码存在与实验路径可达性的差别。 |
| orientation / Person loss 在 120-case eval 中没有 graph 输入 | eval graph 只有两个 `interaction_type='Group'` 节点，`person_to_group={}`；同时 batch 写入 `angle_weight=0.0`。LLM 的 `orientation` 不被 `build_4person_inter_graph` 传递。 | 按实际 graph/参数追踪，不将这些函数仅因存在于源码就算入论文 eval。 | 当前 120-case 路径不会执行 Person-target/orientation loss；不是算法删改。 | 高：dead / inactive branch 定位。 |
| eval 的 `seed` 实际只控制 test list 抽样 | 固定本地 `test_texts_120.json` 正好 120 项，不触发抽样；`dpmotion_eval.py` 未调用 `torch.manual_seed`、`np.random.seed` 或 `random.seed` 来设采样噪声。 | 实验驱动脚本将以 CLI seed 派生并记录每个 `file_id` 的 Python/NumPy/Torch/CUDA seed，以便可恢复。 | seed plumbing 固定随机实现，但采样分布/算法不变；和作者未知 RNG 状态下的 README 输出不保证逐样本相同。 | 高：命令行有 seed 不代表生成器被 seed。 |

后续每个 smoke / 120-scene 失败、修复、最终指标在本文件继续添加；结论以运行产物更新。

## 2026-10-07：RAG 实跑与知识库构建

| 项目 | 表现 / 证据 | 根因或处理 | 对算法语义 / 论文比较的影响 | 面试讨论价值 |
|---|---|---|---|---|
| 官方知识库构建慢且无批次进度 | `InteractionKnowledgeBase.build_from_annots` 对 7,897 个文件做 exact dedup 得 22,915 条后，CLIP 编码及 embedding 贪心去重总耗时 419.04 秒；阶段中只有“CLIP 编码中...”一条日志。 | 按原函数和阈值 0.95 构建，未改检索逻辑；保存原始 KB、embedding、entries、文件名清单 hash 与构建元数据。 | 只是可观测性/耗时问题，未改变算法。当前保留 19,443 条，README 先前记录 19,429 条，差 14 条；当前 annotation 文件名集合 hash 和缓存 CLIP 应一并比较，不能通过调阈值凑数。 | 中：不可见的长预处理阶段会影响复现排期。 |
| RAG 知识库包含 test annotations | 官方 `build_from_annots` 遍历 annots 目录全部 `.txt`，未按 split 过滤；真实 query 2636 的 top-1 就是 query 原文（相似度 0.99983）。另一例因 float 精度 top-1 cosine 为 1.00008。 | 保持原始构建协议并在输出里记录所有 test query 的 top-20 和候选选择理由；不移除 test 数据以免擅改协议。 | 相似度 `<0.90` 会跳过 query 自身及近同义高相似候选，但 retrieval pool 仍包含测试描述，需将结果视作潜在检索泄漏；float32 归一化点积允许极小超 1 值。 | 高：检索系统的 train/test 边界与 dedup 规则。 |
| top-20 / 0.90 选择规则有 rank-6 回退 | 120 条实际 query 中 87 条选取第一个 `<0.90` 的结果，33 条 top-20 均未低于阈值而固定取 rank 6；无空检索回退。例 2636 的 top-1 原文被跳过，rank2–5 仍 ≥0.90，故选 rank6、相似度 0.92380。 | 严格复用 `dpmotion_eval.py` 同一启发式，记录完整候选、余弦分数、所选 rank 和原因。 | rank6 回退并不满足 `<0.90`，所以“两个 group 必然低语义相似”并非该代码保证；33/120 个 case 明确违反阈值。 | 高：启发式的 fallback 会弱化表面约束。 |
| Smoke trace 维度错误（仅诊断代码） | 2636 四条件 smoke 的第 3 组在 t=100 结束前报 `tensor a (524) must match ... b (262)`；上游采样已正确拼接两个人的 x0，错误发生在 runner 计算 GLI 原始值的 denormalization。 | 追踪器误用拼接输入的 `x.shape[-1]=524` 当每个人维度，而 normalizer 仍为单人 262-D；已在 runner 中按 `sampling_info['all_starts'][0].shape[-1]` 拆两人后分别反归一化，并重跑 smoke。 | 该错误没有修改 DPMotion 源码或模型，只使第一轮第三/第四条件未跑完；重跑用于生成有效 trace。 | 高：联合张量的布局与 per-person normalizer 边界。 |

## 2026-10-07：Guidance smoke、渲染和正式生成

| 项目 | 表现 / 证据 | 根因或处理 | 对算法语义 / 论文比较的影响 | 面试讨论价值 |
|---|---|---|---|---|
| 四组同噪声 smoke 证明 RAG 与 guidance 都改变输出 | 2636，derived seed `42+2636=2678`，window150；A/B/C/D 分别是 no-RAG/no-guidance、RAG/no-guidance、no-RAG/guidance、RAG+guidance。所有输出 `(4,150,22,3)`、有限值。A→B 平均绝对关节坐标差 0.561m；A→C 0.412m；A→D 1.042m；严格 B→D guidance 配对差 0.498m。 | 每个条件重置 Python/NumPy/Torch/CUDA seed，固定 group1/test case、RAG 文本、窗口/采样路径。已保存四个 `.npy`、JSON 摘要和 guidance trace。 | 这是单一 case 的机制 smoke，不能代表 120-case 质量指标；无 guidance 仅让梯度更新为零，保留同一 sampler/model/noise draw。 | 高：成对 ablation 隔离 RAG 与 guidance 的影响。 |
| Guidance 实际有非零梯度并改变 pairwise distance | RAG+guidance trace 每 timestep 两条记录（group pair 顺序采样）。t900：distance weight=5、dynamic factor=1、effective weight=5、outer scale=9.5；group center distance约2.665m、min cross-member约2.026m；total group loss约11.4、gradient L2约0.62。t700 effective=6.25、outer=8.5，center约4.85m、min cross-member约3.86m。t200 distance gradient约0.0014–0.0017；t100起 GLI raw约0.108/0.201，total gradient约0.0047/0.0115；t0 GLI raw约0.113/0.193、gradient约0.020/0.023。 | 保存完整采样节点于 `guidance/D_rag_plus_guidance_scene_2636.json`。源码公式为 `distance_weight*1000/(t+100)`，外层乘 `5+5*t/1000`；GLI `0.6×TopologyBatch` 于 t≤100。 | 本 case 的 group center 从 t900 的 2.665m 被推至 t700 的约4.85m，超出3m下限；distance loss 在后续近零但 GLI 在低噪声尾段继续活动。证明 README 的“仅 group distance”不描述完整当前路径。 | 高：扩散过程中损失、梯度、动态权重的实证。 |
| 官方 README 评价文本回退和当前 eval 代码略有不一致 | `dpmotion_eval.batch_generate` 在所考版本中若无低于0.90的项，先令 `text_g2=text_g1`，随后遇到 rank6 会回退到 `kb_results[5]`；runner `selected_retrieval` 在所有非空输入无低阈值时直接 rank6。两种实现对空/异常 query 的退路略异，但120个有效 query 都非空、top20长度20，因此实跑不受影响。 | 逐行对比并保留全部有效 query 的原结果。 | 只涉及空输入/KB异常分支；当前120场景选择一致。 | 中：边界条件要与主实验输入域一起核验。 |
| 可视化首轮失败于 Matplotlib 3.2 writer fallback | 官方 `plot_3d_motion_multi` 调用 `ani.save(path,fps=...)`；环境 Matplotlib 3.2.0 没有 ffmpeg executable，`Animation.save` fallback 用 `next(writers)` 对 registry 调用而抛 TypeError。把 `animation.writer='pillow'` 也不被该版本 rc 参数接受。 | 在独立 `visualize.py` 里将绘图函数使用的 `FuncAnimation` 包装为显式 `PillowWriter`；只影响 gif 导出，不改 DPMotion 源码和 motion 数组。已写出 2636 的 A/D GIF，并检查中间帧。 | 兼容性修复仅对可视化；motion/evaluation 全程不变。肉眼观察到 4 个 skeleton 均可渲染、两组有明显空间分离；动作类别难凭单个样例确认，部分 limb 轨迹/叠影看起来不够自然。 | 中：下游可视化依赖与生成算法依赖应分离。 |
| 正式 120-scene 执行较 smoke 慢 | 当前完整设置 `DDPM1000`、t≤900 计算 guidance；运行中首场 scene 约1.5分钟，timestep约12/s。 | 后台执行、每场生成后原子写入 NPY 与 JSONL，有限值/shape检查支持 resume；单场异常记入日志并继续。未减少 timestep 或并行改变随机数。 | 预计整套约数小时；运行日志/已完成样本保留。 | 高：显式评估预算和可续跑设计。 |
| `STRATEGY: ddim50` 不是本次真实 sampler | `configs/model.yaml` 声明 `DDIM50`；但 `InterDiffusion.forward_single → p_sample_loop_multi` 显式传 `timestep_respacing=[diffusion_steps]`，该配置为1000，因此实测进度为1000步 `p_sample_with_grad`，走 ancestral DDPM。 | 以真实 eval 调用栈而非配置标签记录 protocol；未擅自改成50步 DDIM。 | 这是关键 protocol 差异，直接影响运行时间和输出，不应称作 DDIM50。 | 高：配置名与真实入口实现不一致。 |
| 运行依赖低于 DPMotion requirements | 当前 Python3.8.20 / torch1.13.1+cu117 / NumPy1.23.5 / SciPy1.4.1 / Matplotlib3.2.0；`requirements.txt` 声明较新 torch、NumPy、SciPy。官方 DPMotion模型 strict load、DDPM、Gaussian smoothing、metric smoke 在现有环境能跑。 | 复用现有环境而未升级大包；完整 runtime 与 pip freeze 已保存。渲染因 Matplotlib 3.2 writer fallback 另做独立 Pillow shim。 | 当前结果来自旧依赖环境；没有已见的生成/eval API不兼容，但数值库与 GPU driver 仍是环境复现因素。 | 中：依赖范围声明与实际可运行下限。 |
| README 指标展示不能从当前代码无歧义复算 | 当前 PeneBone 实现将阈值定为 `2×joint_radius=0.04m`，PeneBone 是跨帧/骨段对穿透深度总和，penetration rate 是穿透骨段对数/总骨段对数的 0–1 比例；README 写百分号形式。README cross-scene diversity `1.5876` 在仓库没有计算函数/协议；当前 intra-scene diversity 则为 4 人轻量轨迹统计 embedding 两两距离均值。 | 按当前 `eval/metrics_penebone.py`、`metrics_core.py` 和 `dpmotion_eval.evaluate_all` 作为官方仓库可执行协议，完整保留 JSON 原值和每 scene 结果。CFrame/Contact 为额外的关节距离诊断，10cm threshold，不等同 mesh/contact evaluator。 | 论文/README中的数字只有在相同定义、单位和聚合顺序下才可严格对比；cross-scene值不能由现有实现重算。 | 高：命名相同不代表统计量定义一致。 |
| 评测 sanity check | 对 smoke 的 D/2636 单场调用仓库 `dpmotion_eval.evaluate_all` 成功，scene shape 有效；PeneBone/rate、CFrame、Contact 都为有限值0，intra-scene diversity=4.216815。与已观察到的跨 group 距离大于3m相容。 | 先用单场验证官方 evaluator 能读 generation `.npy` 并写结构化 JSON，再用于120场。未因0碰撞而临时调阈值。 | 只验证 pipeline 与该样例的指标可算；单场不做模型优劣结论。 | 中：先做 evaluator smoke，避免长实验结束后才发现格式错。 |
| RAG `top_k=20` 的参数动机未在当前仓库解释 | `dpmotion_eval.py` 固定 `kb.query(text, top_k=20)`，README 仅描述检索步骤，没有说明20是经验证的最优候选数量或为何与0.90阈值配合。 | 将 top-20 当作既定实验 protocol；不做 top-k 搜索、不改阈值。`<0.90` 作用由代码行为可观察为跳过 self/高相似语句；33个 rank-6 fallback 仍高于阈值。 | 动机不能从代码证据推断，可能是经验超参；当前结果固定沿用。 | 高：隐藏检索超参与 fallback 会改变 group composition。 |
| 第二固定 case 的视觉检查 | 从正式生成结果渲染 5647 并检查 GIF；四人/两组分离可读，动作语义仍难判断，偶见肢体线条叠影。 | 复用官方 22-joint multi plot，仅做 stride=3 的显示抽帧。 | 显示抽帧只用于 GIF，正式 `.npy` 与 evaluator 仍使用全部150帧。 | 低/中：定性观察有限，不以单帧动画代替动作质量评估。 |
| 仓库快照早于正式发表 | 当前 DPMotion clone 只有 commit `36dafe8`（2026-03-02），Springer 页面显示 2026-07-14 first online；没有后续 git history 可把实现变更映射到论文定稿。 | 固定报告使用的代码 SHA，并把公开摘要数值与当前 repo README/evaluator 分开。 | 源码能复现仓库 snapshot protocol，但不能宣称验证了最终发表版完整 protocol。 | 高：方法代码与正式出版版本的版本对应关系。 |
| 检索 top-1 对 test query 的重复率 | 扫描全部120条 `retrievals.jsonl`，107条 top-1 文本与 query 原文完全相等；其余13条 top-1 为相似改写，cosine均较高。 | 保持原检索池不排除 test annotation；记录原文和候选，不改变主实验。 | `<0.90` 首个阈值会跳过这些 top-1 self / close paraphrases；rank6 fallback 又可能仍≥0.90。 | 高：RAG 测试污染不仅是可能，实际 top-1 self retrieval 占多数。 |
| README 的 ± 与当前聚合标准差约定不明 | `evaluate_all` 对有效 scene 指标先按 scene 求 mean 和 `np.std`（NumPy 默认 population std, `ddof=0`）；README列出 `±` 却没有注明聚合 axis 或 ddof。 | 保持 evaluator 的原始 mean/std，不反向调样本标准差去贴近 README。 | 均值可依相同 per-scene metric 对照；标准差存在统计约定差异风险。 | 中：报告均值±方差必须说明聚合口径。 |
| Group/Person loss 函数内只有 inter-group 分支在当前 graph 有有效约束 | `build_4person_inter_graph` 给两个 Group 都设 `member_distances=None`，所以 intra-group distance loss 不入 loss_dict；两组均给 `group_distances='>=3.0m'`，所以 inter-group loss 的中心距离和4个跨组成员 root XZ 距离项会执行。`person_to_group={}`，所以 person-target、其 orientation、person-other、person-person 分支的 loop 均为空。 | 已结合实际 graph 字段和单场 trace 的 `group_loss_components={'inter_group': ...}` 核验。GLI 是 `gradients` 内单独添加，非该 loss function 的 person branch。 | README“只留 unrelated group distance”可近似描述 active 距离项，但遗漏 t≤100 的独立 GLI；源码里其余 Person/Group loss 分支当前均未启用。 | 高：区分可调用函数、有效 graph constraint 与真实 loss component。 |
| 仓库内 `freemotion_eval.py` 不是 DPMotion 论文 eval 入口 | 该脚本另行导入 FreeMotion control model，以硬编码 `/projects/wbl/...` 路径加载部分 InterGen 权重并生成4人；`dpmotion_eval.py` 才是当前 README 120-scene 结果对应入口。 | 通过当前调用入口选择和路径检查排除 `freemotion_eval.py`；它没有参与本轮 DPMotion 生成/指标计算。 | 不混入 FreeMotion probe 结果，也不把其随机初始化 control branch 误当 DPMotion 论文实现。 | 高：同仓不同实验脚本常被误认为同一 protocol。 |

## 2026-10-07：120 场正式生成与评测完成

- 正式生成 120/120，无失败、缺失 ID、重复 ID、错误 shape 或非 finite 输出；全部 `(4,150,22,3)`。逐场生成均值 89.88s、中位数 88.92s、范围 61.88–150.72s，生成耗时合计 10,785.5s。完整清点在 `artifacts/dpmotion_reproduction/metrics/full_integrity.json` 和 `full_run_summary.json`。
- 仓库 evaluator 原样执行得到：PeneBone `0.3083289 ± 1.0782015`；penetration rate raw fraction `5.37919e-5`；interaction PeneBone `0.3083289` / raw rate `1.61376e-4`；non-interaction depth/rate均为0；intra-scene diversity `3.2720608 ± 0.5623503`。额外 diagnostics：CFrame `8.8778% ±19.0458%`，Contact `0.308833 ±1.020821` pairs/frame。
- 与 README 数值相比，PeneBone mean −0.033071、std −0.213699，intra-scene diversity mean −0.066539、std +0.010950。若将 raw rates×100 转成百分比，则整体 rate `0.005379%`（README `0.0060%`），interaction rate `0.016138%`（README `0.0181%`）。README 没有写清 rate 单位换算和 std 的 `ddof`；原作者 RNG 未记录，本次显式 seed=`42+file_id`，所以不宣称逐样本 bitwise reproducibility。Cross-scene diversity README值1.5876没有当前源码实现/定义，仍不可复算。

## 问题类型索引

**实际遇到并处理的工程故障**

- Guidance 诊断器曾把双人拼接张量的 524 维错当单人 motion 维度，GLI trace denormalize 在 `t=100` 报 shape mismatch；只修复诊断器拆分方式，随后重跑 smoke。见上文“Smoke trace 维度错误”。
- Matplotlib 3.2 的 GIF writer fallback 因 ffmpeg 缺失和旧版 writer registry 行为报错；可视化 wrapper 指定 PillowWriter 后恢复导出，没有改 motion 或模型。见“可视化首轮失败”。
- InterCLIP 初始化时 `clip.load` 默认指向 `~/.cache/clip`，没有识别 `assets/clip` 现有模型，曾开始重复下载 891MB 权重。终止下载并清理未完成缓存文件后，在 evaluator driver 中显式设置 `download_root=assets/clip`；FreeMotion 的 `interclip.ckpt` 随后对 InterGen evaluator 架构 strict-load 成功（207 keys）。
- DPMotion clone 缺 checkpoint/normalizer 和预构建 KB；从已有 InterGen/CLIP/InterHuman 资产复用与重建后继续，346/346 参数严格加载及 mean/std 一致性均已核验。这是发布资产缺省，不是模型参数不兼容。

**固定代码快照直接证实、但不能据此判断作者意图的事实**

- `forward_single → p_sample_loop_multi` 实际是 1000-step ancestral DDPM；`model.yaml` 的 `DDIM50` 标签未被该路径使用。
- 当前 4-person graph 在 `t≤900` 启用跨组距离，在 `t≤100` 加 GLI；Person/orientation 分支因 graph/weight 输入而未启用。
- RAG 构建读入全部 annotations，实际 120 query 中 107 条 top-1 是原文；33 条 fallback 的 rank-6 仍高于 0.90。
- PeneBone 当前按 0.04m 骨段最近距离阈值，对每场所有帧/骨段对穿透深度求和；cross-scene diversity 在当前 evaluator 没有计算函数。

**尚未证实、不得当作结论的假设**

- top-k=20、相似度阈值0.90、rank-6 fallback 的设计动机，以及 README 的简化是否有意省略 GLI。
- 当前遗留的 GLI 是否参与最终论文实验、当前仓库快照是否等同出版定稿代码。仓库首 commit 早于论文 online date，论文完整 protocol 未能从公开摘要恢复。
- RAG 是否在控制 guidance 后提升动作语义或整体质量；单样例证明检索文本能改变动作，不证明“变好”。尚未有覆盖120场景的 RAG/no-RAG 成对指标证据。
- README 的随机状态、penetration-rate 单位换算、标准差聚合口径，以及 cross-scene diversity 的具体特征定义。

## 2026-10-08：方法有效性配对核查

| 检查 | 实际结果 / 证据 | 边界 |
|---|---|---|
| Guidance on/off 是否使用相同随机噪声 | 对 2636、5647、2813、4000 固定 RAG 文本、组图、seed (`42+file_id`)、DDPM1000、窗口150、距离3m和weight5。on/off 的 `torch.randn` 与 `torch.randn_like` 生成张量序列逐个 hash 完全相同，每例2,009个张量；因此初始噪声及每个采样步随机项相同。详见 `ablations/guidance_pairs/pairs_manifest.json`。 | no-guidance 在诊断 runner 里将 guidance gradient 置零；同一 sampler/model/text。只固定并验证 PyTorch sampling noise 序列，实验不用于证明所有库/设备间的 bitwise 可复现。 |
| Guidance 是否改变组间距离 / 几何穿透 | 四例组中心 XZ 均距 off→on：2636 `2.410→4.619m`，5647 `0.542→3.436m`，2813 `1.442→3.865m`，4000 `0.722→3.316m`。off 时组中心距小于3m的帧占比为 `54%,100%,100%,100%`，on均为0。PeneBone depth sum off→on 为 `0.782→0, 5.717→0, 1.288→0, 0→0m`；contact-frame rate on均为0。 | 是四个预先挑选、覆盖多种检索 rank 的机制例子，不是随机的120-case消融；只能说明当前距离 guidance 对这些 case 有效地实现空间分离、减少几何接触/穿透。 |
| Guidance 是否改变组内动作 | 每例 on/off 平均 joint XYZ 绝对差 `0.480–0.673m`；四例 mean joint speed 均值 `0.01598→0.01527m/frame`。组内双人 root 距离有升有降，例如 2813 Group2 `1.437→0.729m`、2636 Group1 `1.626→2.573m`。 | 输出改变不等同于动作质量提升；没有在此引入主观/语义质量分数。所有 per-scene、per-group 项见 `ablations/guidance_pairs/metrics_summary.json`。 |
| 评测实现边界 | PeneBone 是当前代码对21条骨段/人、每帧跨人骨段对，用最近线段距离和 `2×0.02m=0.04m` 阈值算 `max(0,0.04-distance)`，对深度求和；穿透率为距离低于阈值的骨段对比例。 | 这是骨段阈值 proxy，不是带 mesh/capsule 的物理碰撞检测。 |
| Cross-scene Diversity 缺项原因 | README 有该指标名称和值，但当前 DPMotion Python 源码里没有计算入口、motion embedding 定义或 aggregation protocol；对整个仓库检索也只找到 README 提及。 | 属于实现/定义缺失，不是本轮运行报错。不能从README单个数值反推实现。 |

本次配对结果支持 guidance 的几何分离作用，并显示它会连带改变组内轨迹；不能据此声称自然度、交互语义或整体质量变好。RAG 是否改善生成质量仍未验证；测试文本与 KB 重叠是已证实的数据边界问题，必须和任何 RAG 效果结论一起披露。

复跑一致性补充：重新捕获的 2636 guidance off/on 两个 `.npy` 分别与先前四路 smoke 的 RAG/no-guidance 与 RAG+guidance 文件逐元素完全相同（max abs diff=`0`）。这同时验证了该 case 的 reseeding 和条件实现没有引入不同输出。

## 2026-10-08：完整 RAG × guidance 2×2 消融完成

- 120 个 test scenes × A/B/C/D 四臂全部生成；完整性为 480/480 scene-condition，指标见 `ablations/full_2x2/`。
- A=no-RAG/no-guidance；B=RAG/no-guidance；C=no-RAG/guidance；D=RAG+guidance。seed=`42+file_id`，每场150帧、DDPM1000、distance=3m、weight=5。
- RAG KB 与 test split 重叠是已证实的数据泄漏风险；120 个 query 有 107 个 top-1 原文命中。InterCLIP Group2 matching distance 没有正向变化，但本次非论文 protocol 的 120-way R@1 有改善；两者边界见主报告。
- no-guidance 通过 runner 诊断 hook 将 guidance gradient 置零；上游模型和采样代码未改。
- guidance 条件生成较慢且出现数次进度长尾；进程可恢复且 480/480 完成。单次观察到 NFS I/O wait，不能据此归因所有延迟。
- 首次报告汇总因 evaluator contrast key 大小写不一致抛出 `KeyError`；几何与 InterCLIP 指标已成功生成，修正汇总映射后完整写入报告。未重跑生成或改动指标。
