# DPMotion 复现成果审计

审计日期：2026-10-09。**完成度：固定仓库快照的生成、几何评测及本地 2×2 消融已完成；最终论文完整指标协议尚未复现。** 不能用单一百分比表示论文复现完成度。

最重要的五个发现：

1. **成果完整且相互一致。** 本轮直接读取主实验 120 份 NPY、四臂 480 份 joints 和 480 份 features：全部 shape/finite 合格；四臂日志各 120 个唯一预期 ID、status=ok、seed 正确。D 臂全部 120 份 joints 与主实验逐元素一致。几何 JSON 均值和保存的 InterCLIP embedding 所算 Group2 distance/R@1 与汇总误差均为 0。
2. **RAG 测试库重叠已证实。** KB 读取全部 annotation，120 条 query 的 top-1 有 107 条原文命中；87 条按 `<0.90` 选择，33 条回退 rank 6 且仍 ≥0.90。不能宣称无泄漏泛化提升。
3. **真实采样是 DDPM1000，guidance 是距离项加 GLI。** `DDIM50` 配置标签不控制此次入口。距离在 t≤900 启用，GLI 在 t≤100 启用；“只做距离约束”和“50 步采样”均不能描述此次实验。
4. **guidance 明显降低当前几何穿透，但质量结论有限。** A/B 的 PeneBone 为 19.2735/10.9309，C/D 为 0.2079/0.3083；Group2 matching distance 反而增加。多样性 embedding 含 root 的绝对 XZ 均值，空间分离会影响该指标，不能将其增长直接等同动作多样性改善。
5. **README 接近复现，论文指标协议不一致。** 主实验 PeneBone=0.308329、diversity=3.272061，与 README 0.3414/3.3386 接近但非精确复现；cross-scene diversity 无当前实现。论文 R@1=0.386/FID=5.642 尚未按对应协议计算。本地 Group2 120-way R@1 是额外诊断。

## 1. 上一轮上下文与资产

工作区顶层不是 Git 仓库；实际仓库为 `code/DPMotion`。本轮 `git rev-parse HEAD` 为 `36dafe8b2cefa8f4610723fb10f581e380548498`，`git status --short` 为空。复现 runner 位于独立 artifacts 目录：源码干净不代表运行没有 hook（见下文）。

| 项目 | 本轮核查与证据 |
|---|---|
| 模型 | `assets/intergen/intergen.ckpt` 存在；本轮重算 SHA256=`bab341123c27af9d8c3ee7ebc43c2d341bfd0f4103425d06f8af6c717a086f8d`，与配置相同。epoch299/global_step70200、346 keys strict-load 成功由 [model_compatibility.json](../artifacts/dpmotion_reproduction/environment/model_compatibility.json) 和 `logs/model_audit.log` 记录；本轮未重新加载模型。 |
| Normalizer | DPMotion 的 mean/std 链接到 `code/InterGen-official/data/`；本轮文件 hash 与 compatibility JSON 相符（mean `2f97b888…03b6`、std `dd5988b9…3e4b`）。 |
| 数据 | `code/DPMotion/data/test_texts_120.json` 共120条，ID全部属于 `data/InterHuman/split/test.txt`，interaction_text 全部能在对应 `annots_utf8/<id>.txt` 找到。数据是固定本地 manifest，不应理解成重新随机抽120条。 |
| KB/CLIP | KB pickle、entries JSONL、embedding NPY 和 `assets/clip/ViT-L-14-336px.pt` 存在；构建记录为7897个 annotation文件、19443条去重后描述、(19443,768) embedding、无 split filter。见 [build_metadata.json](../artifacts/dpmotion_reproduction/rag/build_metadata.json)。本轮未重建或重算 CLIP。 |
| 配置/环境 | [reproduction.json](../artifacts/dpmotion_reproduction/configs/reproduction.json)：150帧、30fps、CFG3.5、距离3m、distance_weight5、angle_weight0、Gaussian sigma1 smoothing。seed=`42+int(file_id)`，runner逐场设置 Python/NumPy/Torch/CUDA。记录环境 Python3.8.20、Torch1.13.1+cu117；完整 runtime/pip freeze 在 `environment/`。 |
| 最新产物 | 主实验 `samples/full/`；完整消融 `ablations/full_2x2/{A_no_rag_no_guidance,B_rag_no_guidance,C_no_rag_guidance,D_rag_guidance}/{joints,features}/`；每臂JSONL、4份repo metrics、comparison、InterCLIP JSON/NPZ均在。4-case guidance pairs、2636四路smoke、trace与GIF也保留。 |

本轮未发现上述核心成果缺失，故继续审计。资产完整性检查与保存统计的重汇总不等于独立重新执行几何 evaluator；本轮没有重跑生成、训练或模型推理，没有覆盖旧日志/指标。

## 2. 机制考古

- **RAG实际作用：** [DPMotion.py](../code/DPMotion/DPMotion.py) `build_from_annots:114` 遍历全部txt；精确文本去重及 embedding相似度0.95去重不等于按split隔离。`query:200`检索CLIP文本向量；[dpmotion_eval.py](../code/DPMotion/dpmotion_eval.py) `batch_generate:189–198`用top20选择Group2描述。实际评测入口将该描述编码进 `motion_text_map`，不是从KB复制参考motion或空间坐标。论文摘要提“spatial initializations”，与这条入口的确切对应关系**待核查**。全部候选证据：`rag/retrievals.jsonl`。
- **真实采样：** [nets.py](../code/DPMotion/models/nets.py) `forward_single:302–346`重建全1000步diffusion并调用 `p_sample_loop_multi`；[gaussian_diffusion.py](../code/DPMotion/models/gaussian_diffusion.py) `p_sample_with_grad:1435–1462`先采 ancestral随机项，再减guidance梯度。日志有1000步进度，与调用链一致。
- **实际loss：** graph在 `dpmotion_eval.py:109–132` 中设两组[0,1]/[2,3]、member_distances=None、person_to_group={}。[group_person_losses.py](../code/DPMotion/models/group_person_losses.py) `226–298`实际启用组中心XZ距离及4个跨组成员root XZ距离，二者平均；Person/orientation及组内距离分支没有有效输入。距离权重=`5×1000/(t+100)`，外层梯度尺度=`5+5t/1000`；GLI独立加`0.6×TopologyBatch`（`gaussian_diffusion.py:2689–2807`），作用于当前双人组。这里的GLI/距离是推理期loss，不是本轮训练loss。
- **消融定义：** [full_ablation.py](../artifacts/dpmotion_reproduction/full_ablation.py) `54–85` 和 [repro.py](../artifacts/dpmotion_reproduction/repro.py) `240–243`：no-guidance通过运行时hook返回零loss/梯度，关闭距离及GLI整体；no-RAG令Group2复用Group1描述。A是DPMotion同一多组sampler关闭两种机制的基线，不能当成独立官方InterGen原生benchmark。尚无GLI与距离分别消融。
- **随机数证据边界：** 全120四臂均按同seed重置，有代码和逐场记录；`full_ablation.py:173–179`仅smoke捕获Torch噪声hash，不能说480次生成都有逐随机张量hash证明。4个guidance pairs另有每例2009个噪声张量相等记录（`ablations/guidance_pairs/pairs_manifest.json`）。

## 3. 指标、结果与协议边界

`dpmotion_eval.py:250–378`逐scene汇总，std为`np.std`默认ddof=0；interaction pairs=(0,1),(2,3)，non-interaction为剩余4对。

| 指标 | README | 本次主实验/D | 审计结论 |
|---|---:|---:|---|
| PeneBone mean ± std | 0.3414 ±1.2919 | 0.308329 ±1.078201 | 当前repo协议已完成，数值接近；不能称精确重现 |
| penetration rate | 0.0060% | raw 0.0000537919 = 0.00537919% | README百分比解释下可比较；原作者换算/随机状态待核查 |
| interaction rate | 0.0181% | raw 0.000161376 = 0.0161376% | 同上；报告中的“inter-pair”实际对应组内interaction两对 |
| non-interaction rate | 0% | 0 | 当前几何proxy下复现 |
| intra-scene diversity | 3.3386 ±0.5514 | 3.272061 ±0.562350 | 当前repo实现已完成，README聚合约定待核查 |
| cross-scene diversity | 1.5876 | 未计算 | 当前源码缺定义/入口，尚未复现 |
| 论文Top1 R-Precision/FID | 0.386 / 5.642 | 未按论文协议计算 | 尚未复现；本地InterCLIP不能替代 |

定义依据：[metrics_penebone.py](../code/DPMotion/eval/metrics_penebone.py) `73–195`用21条骨段/人，跨人骨段最近距离低于0.04m时累加`max(0,0.04-d)`，PeneBone是整场深度和、rate是骨段对比例；不是mesh穿透。`eval/metrics_contact.py:18–103`按关节中心距离<0.10m算接触帧百分比和每帧接触关节对数。`eval/metrics_core.py:106–154`用root位置、轨迹长度、速度及选定关节速度拼接embedding，取4人6对欧氏距离平均；含绝对位置，不能保证平移不变。

完整消融JSON独立重汇总结果：

| 条件 | PeneBone | rate（raw） | diversity | Group2 matching distance↓ | Group2 R@1↑ |
|---|---:|---:|---:|---:|---:|
| A 无RAG/无guidance | 19.273467 | 0.003003254 | 1.414787 | 3.872774 | 0.316667 |
| B RAG/无guidance | 10.930947 | 0.001753779 | 1.530702 | 3.884032 | 0.475000 |
| C 无RAG/guidance | 0.207924 | 0.000036575 | 3.295393 | 3.905605 | 0.216667 |
| D RAG/guidance | 0.308329 | 0.000053792 | 3.272061 | 3.913358 | 0.316667 |

证据：[repo_ablation_comparison.json](../artifacts/dpmotion_reproduction/ablations/full_2x2/repo_ablation_comparison.json)、[interclip_alignment.json](../artifacts/dpmotion_reproduction/ablations/full_2x2/interclip_alignment.json)及相邻NPZ。InterCLIP用未smoothing的262维生成特征，几何指标用smoothing后的XYZ；两种评测对象不同。`eval_full_ablation_interclip.py:54–87,121–153`用每臂自己的Group2文本/动作120×120距离矩阵，对角线排名，不是同一个跨条件候选池。RAG改变条件文本及候选池，R@1增长不能单独证明同文本下对齐改善。保存的bootstrap结果显示RAG matching-distance差值区间跨0，而guidance的distance增量区间为正；未独立重跑bootstrap。

论文公开页本轮核实：[Springer](https://link.springer.com/chapter/10.1007/978-981-92-3438-7_39)摘要确有0.386/5.642及training-free描述，但无完整protocol且正文需订阅。first online=2026-07-14；页面建议引用年份为2027，不能仅写“2026出版”而忽略此差异。当前仓库commit早于online日期，最终发表代码对应关系**待核查**。

## 4. 导师可能追问的工程复盘

| 问题 | 根因 | 已采取的解决方式 | 验证证据 | 工程启示 |
|---|---|---|---|---|
| 缺权重/normalizer能否直接复用？ | 发布仓库未附资产；上游load非strict会掩盖缺键 | runner只去掉前缀并strict-load，normalizer链接官方文件 | `repro.py:68–101`；`logs/model_audit.log`、compatibility JSON；本轮hash相符 | 权重和归一化必须共同锁定，加载成功须有键/shape证据 |
| `--seed`为何不够？中断如何续跑？ | 上游seed只用于文本抽样；仅凭文件存在跳过可能接受半成品 | 每场seed_all；临时NPY+原子替换；shape/finite检查和追加JSONL | `repro.py:60–65,313–324,395–428`；`full_ablation.py:38–51`；本轮120/480清点通过 | 可恢复生成需要逐场provenance和有效性判据 |
| 为什么配置50步却运行很久？ | `forward_single`覆盖respacing；真实guidance执行到900且含GLI | 记录真实调用链，保留DDPM1000 | `nets.py:310–331`、1000步日志；`metrics/full_run_summary.json`记录主生成约10785.5秒 | 入口执行行为优先于配置标签；NFS导致所有长尾的说法待核查 |
| trace为何524/262报错？ | 诊断器把双人拼接维度当单人normalizer维度 | 从all_starts取单人维度，拆两人分别反归一化；重跑smoke | `logs/smoke_2636.log`报shape mismatch；`repro.py:188–205`及retry日志、有效trace | 联合张量布局与单人特征契约要显式区分；此修复在诊断代码 |
| GIF导出失败是否影响结果？ | 旧Matplotlib的writer fallback报registry非iterator | 独立wrapper显式PillowWriter | `logs/visualize_2636.log:10`；`visualize.py:22–30`；5个GIF存在 | 渲染故障与模型/数值产物分开定位 |
| 汇总KeyError是否表示实验失败？ | contrast名称大小写不一致 | 修正汇总key映射 | `logs/full_ablation_finalize_initial_keyerror.log`报`rag_effect_without_guidance`；`finalize_full_ablation.py:50–54`现用`RAG_effect_without_guidance`；finalize日志成功 | 指标计算与报告渲染应分开，避免为展示错误重跑昂贵实验 |
| 为何工程笔记仍说没有120场RAG消融？ | 追加式记录保留了早期状态 | 审计采用最后完整产物，指出过时段落；未改旧文档 | `REPRO_ENGINEERING_NOTES.md`前段/假设列表与末尾完成记录矛盾；四臂真实文件齐全 | 历史记录不能充当当前状态；FUTURE_IDEAS的seed建议针对官方CLI，runner已实现 |

## 5. 尚待核查

原作者RNG、README rate换算及std定义；cross-scene diversity实现；论文完整split、multi-group R-Precision/FID输入和聚合协议；最终出版代码/权重版本；top20/0.90/rank6设计动机；GLI在最终论文中的角色；InterCLIP当前262维输入是否与最终论文的预处理/坐标规约一致；自然度与整体语义质量提升。恢复来源应优先为作者最终代码/实验配置、论文正文与补充材料、原始评测脚本及日志，不能从摘要数字反推。
