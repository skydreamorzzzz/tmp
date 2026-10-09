# FreeMotion 方法级复现记录

本工程为 engineering reconstruction，不是官方旧环境或论文数值的严格复现。
官方代码：`code/FreeMotion-Codes`，提交 `df92bdd982e25363fd27c86899e22b3b40230de2`。
已跟踪的官方 Python 源码未修改。

## 当前训练

`runs/official_budget_v1`：正式完整训练预算，Stage1 2500 epochs，然后自动以其最后检查点初始化 Stage2 1000 epochs。
Stage1 effective batch=80、Stage2 effective batch=30；优先相同 physical batch，只有实际 CUDA OOM 才减小 physical batch 并梯度累积。单 RTX 4090，FP32，4 个数据加载 worker。
两个阶段 AdamW 基础 LR=1e-4、weight_decay=2e-5，保留官方 warmup/cosine；初始实际 LR=1e-5 来自 warmup。没有因速度或显存减少 epoch。
原 `runs/full_data_v1` 的 30→20 epochs 任务已按用户要求终止，检查点保留但不用于正式训练初始化。
没有设置 max-steps 或抽样子集；官方 loader 自身的最短序列过滤和镜像增强保留。

- `runs/official_budget_v1/status.json`：当前阶段、训练 PID、失败状态。
- `runs/official_budget_v1/stage1.log` / `stage2.log`：训练输出。
- 各阶段目录的 `metrics.jsonl`：逐步 loss、分项、梯度范数、学习率。
- 各阶段目录的 `epochs.jsonl`：完整 epoch 的平均 loss。
- 各阶段目录的 `last.pt`：每个 epoch 原子替换，含模型、AdamW、scheduler、随机数状态和下一步位置。
- `pipeline.pid` 是后台管理进程 PID。对话断开不会停止训练。

仅在旧进程已经退出后重新启动（自动从最后保存的 epoch 恢复）：

```bash
cd /home/featurize/work/FreeMotion-reproduction
nohup .venv/bin/python -u engineering/pipeline.py > runs/official_budget_v1/pipeline.log 2>&1 < /dev/null &
```

中途崩溃最多损失当前未保存的一个 epoch。OOM 重启从最后检查点恢复，若尚无检查点则重启当前阶段；日志以 epoch/batch/step 标记，重放部分可能重复。没有自动无限重试。

## 保留的算法

直接调用官方模型、diffusion、文本和 motion condition、condition dropout、loss、归一化和数据预处理。
Stage2 使用官方权重映射复制 Stage1 backbone 到 control branch，保留 backbone 冻结和 zero-linear 初始化。
训练保留 AdamW（lr=1e-4、weight_decay=2e-5）、梯度裁剪 0.5、官方 10 epoch warmup/cosine scheduler。
cosine 的总周期是官方 2500/1000 epochs。
冻结的 CLIP 及官方额外文本编码层行为保持原样。

## 与官方运行方式的差异

1. 用 `engineering/train.py` 普通 PyTorch 循环代替 Lightning；不再追求旧传递依赖逐版本匹配。
   `.venv` 复用机器已有 PyTorch 1.13.1+cu117，不修改基础 torch/CUDA。
2. 正式预算恢复官方 epochs=2500/1000、effective batch=80/30。若 OOM，使用小 physical batch 累积：simple 项按有效帧数加权，几何项及 DM/JA 按 t<=T_BAR 的样本数加权；RO 保留官方跨 batch/time 归约，使用全 batch 的 timestep gate 和有效帧数权重；每个 effective batch 只做一次裁剪和 AdamW 更新。保留各原始归约的 epsilon。随机 dropout/噪声的抽样顺序可能与一次大 batch 不同，不承诺逐位一致。OOM 使用退出码 86 触发有限的 physical batch 回退，其他异常直接报告失败。
3. `drop_last=False`，保留每个 epoch 最后一批；官方丢弃最后不足 batch 的样本。
4. 显式设置随机种子，并保存 optimizer/scheduler/RNG。多 worker 增强下的 epoch 中间恢复不承诺逐位一致。
   Stage2 完整检查点恢复绕过官方用于初始化的 load_state_dict 重映射，防止覆盖已经训练的 control branch。
5. 采用绝对路径数据视图、显式 Stage1 检查点路径、本地官方 CLIP 缓存，避免路径和重复下载问题。
6. 公开下载包缺少必需文件的 11 个 ID 仅在工程 split 中排除；原始上传数据和 split 不变。
   训练 6022→6013，验证 580→579，测试 1177→1176；详情 `reports/engineering_data_protocol.json`。
   官方 loader 还会过滤不足 15 帧的序列。
7. 两个非 UTF-8 标注 4642、5834 做 GB18030→UTF-8 无损编码转换，使用 overlay，不改文本内容或原始文件。
   详情 `reports/annotation_encoding.json`。
8. 官方最终 FreeMotion 权重没有发布，当前从零训练，不使用第三方模型权重。

## 资产和验证边界

全量上传 InterHuman 和 separate_annots 通过 `data/InterHuman` 引用；工程训练视图为 `data/InterHuman-engineering`。
InterCLIP 位于 `code/FreeMotion-Codes/eval_model/interclip.ckpt`（链接到用户上传文件），供后续评估使用，不参与生成模型的训练初始化。
CLIP `assets/clip/ViT-L-14-336px.pt` 已按官方 URL SHA256 校验。
论文和 supplementary 在 `papers/`。

Stage1 前向、反向、检查点保存及恢复已经运行通过。全量训练启动不等于训练完成。
训练结束后仍需检查损失曲线、生成 motion 的骨架和时序质量，再使用 InterCLIP 评估。
在这些完成之前，不宣称合理 motion 已验证或论文指标已复现。

正式启动结果：Stage1 physical batch 80 在 24GB 4090 上实际 OOM，自动切换 physical 40、累积 2 次、effective 80。前 40 次更新 loss/梯度有限，峰值 allocated 14.87 GiB。Stage2 尚未开始，届时优先 physical 30，仅 OOM 后累积回退。累积归约验证见 `reports/accumulation_validation.json`。
