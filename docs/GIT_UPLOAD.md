# Git 上传配置

工作区根目录已初始化为独立 Git 仓库，默认分支 `main`；Git LFS 3.8.0 已安装并通过官方 release 的 SHA256 清单校验，已启用本仓库 LFS hook。尚未设置远程、提交身份，未执行 commit/push。

`.gitignore` 排除模型权重（包括权重软链接）、本地数据集、虚拟环境、Python缓存、检索KB/embedding缓存和本地密钥文件；保留复现脚本、docs、normalizer、评测JSON、实验日志及生成结果。所有现有权重均已核查被忽略，本地文件保留。`.gitattributes` 仅使用 LFS 管理 NPY/NPZ/GIF/MP4 等实验产物。

三个上游仓库现在作为普通源码目录提交，避免生成不可携带本地改动的 gitlink。原 `.git` 元数据完整移至本仓库 `.git/upstream-repositories/{DPMotion,InterGen-official,FreeMotion-Codes}`，原commit、远程及修改状态记在 `UPSTREAM_REPOSITORIES.json`。备份属于本地 `.git`，不会推送。需要查看旧仓库状态时可运行：

```sh
git --git-dir=.git/upstream-repositories/DPMotion --work-tree=code/DPMotion status
```

`code/FreeMotion-Codes/eval_model/interclip.ckpt` 本地链接至 `FreeMotion-Codes/eval_model/interclip.ckpt`；DPMotion checkpoint 和 normalizer 的仓库内相对链接保留。权重文件和权重链接均不上传；新机器复现时需另行取得权重并恢复对应路径。

## 手动上传

在工作区根目录运行，替换姓名、邮箱和远程URL；URL不要包含token：

```sh
cd /home/featurize/work/FreeMotion-reproduction
git config --local user.name '你的姓名'
git config --local user.email '你的邮箱'
git remote add origin https://你的平台/你的账号/你的仓库.git
git add .
git lfs ls-files
git diff --cached --stat
git commit -m 'Add reproduction source, audit and results'
git push -u origin main
```

HTTPS认证提示输入密码时使用平台token；实验产物的LFS对象会由pre-push hook一起上传。若已存在origin，用 `git remote set-url origin ...`。新机器clone后执行 `git lfs pull` 获取LFS实验产物；模型权重、数据集和检索缓存需另行恢复/重建，路径和权重hash见复现文档。
