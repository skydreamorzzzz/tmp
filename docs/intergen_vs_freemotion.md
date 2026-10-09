# InterGen vs FreeMotion（不训练）

原生 InterGen 和用户指定的 zero-B 单人路径均使用同一预训练 generator（epoch299 / step70200，SHA256 `bab341123c27af9d8c3ee7ebc43c2d341bfd0f4103425d06f8af6c717a086f8d`），严格加载346个state_dict键；没有修改checkpoint。原生官方工程固定在 `b1750df4a859443b8a69329f759ce3a7503ae179`。

**单人实现。** 最小移植 [DPMotion single_forward](https://github.com/skydreamorzzzz/DPMotion/blob/36dafe8b2cefa8f4610723fb10f581e380548498/models/nets.py) 的zero-B机制，代码为 [single_path.py](../artifacts/intergen_vs_freemotion/single_path.py)。每次去噪令262维归一化输入B为0，保留双人共享Transformer/cross-attention和B隐状态更新，只返回A；A用262维DDIM50扩散。新增参数0，保持原state_dict键/shape；与原生双人denoiser输入`[A,0]`的A输出最大差1.7e-06，通过FP32数值等价检查。这里只移植inference dispatch，DPMotion的graph/空间guidance/RAG和1000步采样未引入。这是双人预训练模型构造的单人路径，不是FreeMotion Stage1或论文重训的InterGen*。

**双人实现。** 未改动的InterGen `forward_test → InterDiffusion.forward → InterDenoiser.forward`，524维联合扩散，同时更新A/B，使用原始互动文本；没有FreeMotion control branch。单人则使用同一case的真实`text1/text2`分别独立生成A，再拼成`[T,2,262]`，用于论文表2式评测；保存文件中的第二条也是独立A输出，不是zero-B伪动作。

**权重不能等价变为FreeMotion。** 只做命名wrapper处理，single匹配282键、缺48、unexpected64、shape mismatch0；全部参数覆盖82.546%，backbone仅67.403%。缺的是活跃的`condition_sa_block`，共50,380,800参数；InterGen是带独立`xf_norm`的cross-attention，不能改名充当SA。multi匹配282/缺220；官方backbone→control复制后匹配390/缺112，仍没有Stage2学习结果，zero-linear16张量/8,396,800参数保持0。真正依赖Stage2学习的是参与条件注入的control交互层及zero-linear；`control_branch.out`虽计入state_dict，但return_intermediate路径不使用。text encoder/adapter100%匹配。逐key/shape事实见 [weight_audit.json](../artifacts/intergen_vs_freemotion/weight_audit.json)，没有继续补权重。

以下按两种论文protocol分别列出参考值；本轮各一次完整测试，**不能据此作严格apples-to-apples优劣结论**。

表2单人设定：两条独立单人动作拼成双人输入，由InterCLIP评估，并非HumanML3D单人evaluator（[ECCV正文](../papers/FreeMotion_ECCV2024.pdf)第11页；同时阅读了[补充材料](../papers/FreeMotion_ECCV2024_supplementary.pdf)）。

| 方法 | FID↓ | R-P@1/2/3↑ | MM Distance↓ | Diversity | Multimodality |
|---|---:|---|---:|---:|---:|
| FreeMotion paper，表2 | 12.975 ± 0.171 | 0.264 ± 0.005 / 0.394 ± 0.006 / 0.473 ± 0.006 | 3.885 ± 0.035 | 7.702 ± 0.027 | 1.300 ± 0.063 |
| InterGen single，zero-B | 9.1330 | 0.2481 / 0.3939 / 0.4763 | 3.8782 | 7.5276 | 1.4408 |

表1双人设定：InterHuman双人互动动作，InterCLIP评估（ECCV正文第10页）。

| 方法 | FID↓ | R-P@1/2/3↑ | MM Distance↓ | Diversity | Multimodality |
|---|---:|---|---:|---:|---:|
| FreeMotion paper，表1 | 6.740 ± 0.130 | 0.326 ± 0.003 / 0.462 ± 0.006 / 0.544 ± 0.006 | 3.848 ± 0.002 | 7.828 ± 0.130 | 1.226 ± 0.046 |
| InterGen native two-person | 6.2370 | 0.4375 / 0.5890 / 0.6581 | 3.8010 | 7.8858 | 1.2751 |

**Protocol差异。** 原test split1177条，下载包缺ID4106的一条motion，另77条不足15输入帧按官方loader过滤，实际1099条全部生成；固定基础seed1234、caption/crop/actor顺序及每case长度（≤299有效帧），两模式共用manifest。统一DDIM50、eta0、cosine1000、CFG3.5、FP32和官方global mean/std（两个工程文件字节一致）。按相同固定shuffle取11批×96=1056条做FID/R-P/MM Distance/Diversity，43条保留motion但按官方drop_last不进主指标。两模式同一1056条GT、同一100个MM文本；每文本30次、10对MM，Diversity300对。

复用未改动的FreeMotion InterCLIP wrapper和metric函数，现有interclip.ckpt严格加载。FID特征×6；Diversity特征×6且距离÷2；MM Distance/Multimodality没有额外×6。长度排序同时作用于text/motion；evaluator输入物理motion而非归一化motion。检索仍用原始互动caption，沿用发布的pair evaluator；论文没有发布单独表2切换脚本。单人独立生成配对方式与表2对应，无法声称逐项复现表2全部未公开细节。

本轮同一GT的R-P@1=0.4176、MM Distance=3.7847、Diversity=7.7804；论文Real对应0.452±0.008、3.755±0.008、7.748±0.064。GT校准基线也不完全一致，进一步限制直接比较。

论文主要指标20次、MM5次并给95%CI，本轮各1次，无CI；发布eval.py则MM也循环20次。冻结随机caption/crop/候选分组、同GT子集和同MM文本，属于可重复的完整测试抽样。主样本与MM重复样本分别生成；发布loader对MM选中case复用首个重复样本。生成按等长case合批，native batch≤8，single对两个caption分别调用batch≤8，非论文的逐case主生成。FreeMotion双人加入per-person与global文本；本轮native只用global、single只用per-person。论文InterGen*还经过CA dropout10%及单人文本重训，本轮released InterGen没有此训练。

**可视化观察。** 固定test1/4/6，未做平滑、actor平移或重定向，见 [单人GIF](../artifacts/intergen_vs_freemotion/visualizations/single_6.gif) / [双人GIF](../artifacts/intergen_vs_freemotion/visualizations/native_two_person_6.gif)。抽样帧中两种输出均有可辨认人体骨架；case4两种动作变化都较小，用力握手不明显；case6原生双人呈现较清楚的共同起身，zero-B单人没有同样明显的起身过程。没有据此证明手部接触、无穿插或全测试集质量。逐case证据见 [observations.md](../artifacts/intergen_vs_freemotion/visualizations/observations.md)。

所有主生成及30次重复均保存 `.npy`，特征和ID存 `*_features.npz`，数值见 [metrics.json](../artifacts/intergen_vs_freemotion/metrics.json)。单人文件每个case两条独立A运动，native文件两条联合运动；输出与评测可重新计算。
