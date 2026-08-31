# TSC-Fusion 项目长期记忆

## 训练服务器
- 当前主力训练服务器（westc）：`connect.westc.seetacloud.com:48312`，root，RTX 4090 24GB，代码在 `/root/TSC-Fusion`（→ `/root/autodl-tmp/TSC-Fusion`）
- 同步：`export TSC_SERVER=root@connect.westc.seetacloud.com TSC_SERVER_PORT=48312 && ./sync_to_server.sh`（AGENTS.md 已更新为 westc；AutoDL 重启/迁移后端口会变，需到控制台查新 SSH 命令）
- 实例已启用 NVIDIA MPS（nvidia-cuda-mps-control 常驻）
- campaign 命名：`<training_source_hash前10位>_oras5_recent_screen_fast_vN` / `..._ablation_screen_fast_vN`，baseline 队列 + 消融 watcher（while kill -0 <queue_pid> 模式）自动接续

## mmap 预组装缓存（2026-08-29 引入）
- `scripts/build_preassembled_mmap.py` 从原始 nc 构建；`scripts/validate_preassembled_mmap.py` 校验 input/target 与旧加载器逐元素一致 + 参考预报 atol 1e-5
- data_loader 新增显式 `preassembled_mmap_dir` 开关（默认关闭），矩阵 `_stage_overrides` 传递；mmap_mode='c'（copy-on-write）防写回
- 缓存位于服务器 `/root/autodl-tmp/TSC-Fusion/.cache/preassembled_mmap_v1`；旧预处理缓存 `/tmp/seaf_cache/oras5_1979_2014`（21G）仍在用

## 训练协议约定
- 并行协议：训练阶段 `--max-parallel 2`（每作业 num_workers=2 + persistent_workers + pin_memory），评估阶段由 interprocess evaluation lock 自动串行（防 90GiB cgroup OOM）
- AMP：2026-08-27 起默认 bfloat16（`mixed_precision_dtype: auto`）；梯度瞬时溢出按跳步处理，连续 30 次才判发散（`nonfinite_grad_skip_limit`）
- 本地 Windows 无法跑测试（无 torch、WSL 被禁、.venv 为 Linux），测试一律在服务器 `.venv/bin/python -m unittest discover -s tests` 运行

## 已修复的关键 bug
- torch checkpoint 序列化 NumPy uint32 RNG state 崩溃（31f79f4）
- fp16 梯度溢出立即抛错杀死全部 screen 实验（dae1abf，见 2026-08-27 日志）
- bf16 autocast 下 `evaluate()` 直接 `.numpy()` 触发 `TypeError: Got unsupported ScalarType BFloat16`（5482005）—— 纯评估导出 bug，收集预测时改 `outputs.float().cpu().numpy()` 即可，与精度无关
- 非 GTB 模型 `data_loader` 未返回样本索引，评估报 `预测样本与 sample_indices 数量不一致`（cf7e19a）—— `return_sample_index` 在 train/val/test 一律置 True

## 评估重算逃生舱
- `scripts/eval_best_only.py`：纯评估路径 bug 修复后，用 `OceanModelTrainer.evaluate` 给已训 `best_model.pth` 重算指标，绕过 `load_checkpoint` 跨版本 hash 闸门（`_load_best_model_weights` 只载 model_state_dict）。参数 `--config / --result_dir / --split validation`。重算前务必先停掉其它占内存的训练，否则 ~90GiB cgroup OOM。
- 改了训练相关文件（train.py/data_loader.py 等）后 `training_source_hash` 会变 → 旧 campaign 无法 resume（闸门报「belongs to a different source hash」），需新 campaign（命名嵌新 hash 前缀）。
