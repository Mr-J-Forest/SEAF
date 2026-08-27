# TSC-Fusion 项目指南

设计与修复必须解决一般性根因，不得针对单个样本、单次运行或某个结果写特例。

## 同步到服务器

```bash
./sync_to_server.sh
```

将本地代码同步到远程训练服务器。排除规则自动从 `.gitignore` 读取，`Data/`、`outputs/`、`.venv/` 等文件不会被上传。

| 服务器 | 地址 |
|--------|------|
| Host | `connect.westd.seetacloud.com` |
| Port | `39323` |
| User | `root` |
| 远程路径 | `/root/TSC-Fusion` |

同步使用 `tar + ssh` 管道，不依赖 rsync，Windows/Mac/Linux 均可运行。

## 在服务器上训练

SSH 连接后后台训练。正式实验优先走冻结矩阵和 campaign 目录：

```bash
cd /root/TSC-Fusion
nohup .venv/bin/python -u scripts/run_experiment_queue.py \
  --stage screen --campaign <training_source_hash>_screen \
  --max-parallel 2 \
  > run_logs/screen.log 2>&1 < /dev/null &
```

参数说明：
| 参数 | 作用 | 示例 |
|------|------|------|
| `--note` | 训练备注（必填） | `--note "stride4_batch16"` |
| `--epochs` | 覆盖训练轮数 | `--epochs 100` |
| `--lr` | 覆盖学习率 | `--lr 1e-4` |
| `--batch_size` | 覆盖批次大小；time-group GTB 必须覆盖整组（旧轨道 151，ORAS5 76） | `--batch_size 76` |
| `--max-parallel` | 同时运行的独立实验数；ORAS5 已校准为 2 | `--max-parallel 2` |

ORAS5 1° 配置使用每个作业 2 个 DataLoader worker，并发 2 个训练作业。该组合在
90 GiB cgroup 内存上实测训练约占 75 GiB；训练后完整评估由队列自动串行化，避免
两个预测汇总同时触发 OOM。不要在未重新测量内存峰值时提高并发数。

查看进度：
```bash
tail -f train.log
```

## 服务器硬件

- GPU：1× RTX 5090（32 GB）
- CPU：宿主可见 208 逻辑核；容器 CPU quota 为 25 核
- 主机 RAM：754 GiB；容器 cgroup 上限 90 GiB，无 swap
- 临时盘：`/root/autodl-tmp` 50 GB
- 路径：`/root/TSC-Fusion/`（指向 `/root/autodl-tmp/TSC-Fusion`）
- Python：`.venv/bin/python`（PyTorch 2.8.0 + CUDA 12.8）
