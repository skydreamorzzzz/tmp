# InterGen → FreeMotion 审计（2026-10-07）

当前 `assets/intergen/intergen.ckpt` 已更新为真实 InterGen generator，epoch299 / step70200。此前文件误放 InterCLIP 的结论不适用于新文件。SHA256：`bab341123c27af9d8c3ee7ebc43c2d341bfd0f4103425d06f8af6c717a086f8d`。

仅移除 Lightning `model.` 前缀并增加 FreeMotion wrapper 层级，不改变 attention 含义：

| 目标 | matched keys | missing | unexpected | shape mismatch | 参数覆盖率 |
|---|---:|---:|---:|---:|---:|
| single | 282 | 48 | 64 | 0 | 82.546% |
| multi，尚未复制control | 282 | 220 | 64 | 0 | 52.760% |
| multi，官方Stage1→Stage2复制 | 390 | 112 | 128 | 0 | 75.829% |

覆盖率按全部 named_parameters 的 numel 计算，包含 frozen CLIP，排除 buffer/重复别名。**single backbone 实际仅67.403%**，104,177,926 /154,558,726参数。text encoder和额外两层text adapter均100%匹配；输入投影、时间嵌入、第一层SA、FFN、输出投影可初始化。

single 每层缺 `condition_sa_block` 的6个张量，共48张量/50,380,800参数。InterGen 对应模块是cross-attention，另有独立 `xf_norm`；8层 `norm` 与 `xf_norm` 权重均不相同（最大绝对差0.2136–0.3326）。将 `ca_block` 政名为 `condition_sa_block` 会丢弃训练参数并改变K/V来源。single不能基本直接复用。

multi 可复制同样67.403%的control参数作为初始化，但没有FreeMotion Stage2学习结果；仍缺control的48个condition-SA张量，以及zero-linear的16张量/8,396,800参数。官方zero-linear初值严格为0，未训练时motion condition注入为0。Stage2必须学习control交互行为和zero-linear，初始化覆盖率不等于交互能力。

CFG无训练参数；发布代码的spatial接口不能视作论文训练过的spatial-control模型。

全部逐key/逐shape列表、原始前缀比较及真实filtered load结果见 [weight_audit.json](../artifacts/freemotion_intergen_probe/weight_audit.json)。没有训练、没有生成FreeMotion probe，其指标和生成可视化为N/A。96条真实测试动作的InterCLIP sanity通过，但不能当作生成指标。

后续已按用户指令改为原生InterGen双人/zero-B单人，见 [intergen_vs_freemotion.md](intergen_vs_freemotion.md)。论文表2是两条独立单人生成拼成双人动作后评估；表1是双人交互生成。论文协议及数值见 [paper_protocol.json](../artifacts/intergen_vs_freemotion/paper_protocol.json)。
