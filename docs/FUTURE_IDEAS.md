# Future ideas（本轮不实现）

- 将 `torch/NumPy/Python` seed 真正接到生成路径，并写入每 scene provenance；当前 CLI 的 `--seed` 只影响 test list 抽样，固定本地 120 条时不控制扩散噪声。
- 将本轮复现 runner 的逐 scene 原子保存、shape/有限值校验和 resume 机制回馈到官方 CLI；本轮 driver 已实现这些机制，官方源文件仍是“仅按文件存在跳过、结束时写总日志”。
- 对 GLI、group-distance guidance 分别做独立消融，并把有效 timestep、loss 和实际空间距离写入结果；当前对外 README 将复杂实现概括为单一距离约束。
- 为多组场景发布与论文 Top-1 R-Precision / FID 对应的 evaluator、split 与完整聚合协议。目前仓库 `dpmotion_eval.py` 的正式实现是碰撞/接触/几何多样性，二者不可互相替代。
