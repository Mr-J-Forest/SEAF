"""
TSC-Fusion 与基线模型的海洋温盐训练脚本
使用统一配置文件确保参数一致性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:  # pragma: no cover - only used in minimal environments
        def __init__(self, *args, **kwargs):
            print("警告: tensorboard 未安装，训练日志将不写入 TensorBoard")

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass
import numpy as np
import os
import json
import pickle
import time
import random
import platform
import subprocess
import hashlib
import gc
from contextlib import contextmanager
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入统一配置
from config import DEFAULT_CONFIG, load_config, merge_configs, validate_config
from convlstm_model import create_ocean_model
from data_loader import create_data_loaders
from font_config import setup_chinese_fonts
from metrics_utils import (
    compute_metric_report,
    compute_period_group_report,
    compute_sample_group_report,
    resolve_variable_slices,
)

# 初始化中文字体
setup_chinese_fonts()


@contextmanager
def interprocess_evaluation_lock(lock_path: Optional[str]):
    """Serialize memory-heavy post-training evaluation across local processes."""
    if not lock_path:
        yield 0.0
        return

    absolute_path = os.path.abspath(lock_path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    handle = open(absolute_path, 'a+b')
    wait_started = time.perf_counter()
    try:
        if os.name == 'nt':
            import msvcrt

            if os.path.getsize(absolute_path) == 0:
                handle.write(b'0')
                handle.flush()
            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.2)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

        wait_seconds = time.perf_counter() - wait_started
        yield wait_seconds
    finally:
        try:
            if os.name == 'nt':
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _sanitize_note_for_path(note: str, max_length: int = 40) -> str:
    """将备注内容转换为适合路径命名的短标识"""
    if not note:
        return ""
    cleaned = note.strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"[\\/:*?\"<>|]", "-", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip("-_")
    if not cleaned:
        return ""
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("-_")
    return cleaned

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict:
    """Capture every RNG that can affect the next training epoch."""
    numpy_state = np.random.get_state()
    state = {
        'python': random.getstate(),
        'numpy': {
            'bit_generator': numpy_state[0],
            # NumPy's MT19937 state is uint32.  PyTorch 2.8 can use this
            # dtype in memory but its zip serializer has no registered
            # storage type for torch.uint32, so storing a tensor here makes
            # every checkpoint fail at the end of the first epoch.  A plain
            # integer list is portable across PyTorch versions and is
            # converted back to uint32 only when restoring NumPy's state.
            'state': numpy_state[1].astype(np.uint32, copy=True).tolist(),
            'position': int(numpy_state[2]),
            'has_gauss': int(numpy_state[3]),
            'cached_gaussian': float(numpy_state[4]),
        },
        'torch_cpu': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[dict]) -> bool:
    """Restore a checkpoint RNG snapshot; tolerate legacy checkpoints."""
    if not state:
        return False
    random.setstate(state['python'])
    numpy_state = state['numpy']
    raw_numpy_state = numpy_state['state']
    if torch.is_tensor(raw_numpy_state):
        # Read checkpoints produced before the portable-list representation.
        raw_numpy_state = raw_numpy_state.cpu().numpy()
    np.random.set_state((
        numpy_state['bit_generator'],
        np.asarray(raw_numpy_state, dtype=np.uint32),
        int(numpy_state['position']),
        int(numpy_state['has_gauss']),
        float(numpy_state['cached_gaussian']),
    ))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if torch.cuda.is_available() and state.get('torch_cuda'):
        torch.cuda.set_rng_state_all([item.cpu() for item in state['torch_cuda']])
    return True


_RESUME_RUNTIME_CONFIG_KEYS = frozenset({
    'epochs',
    'resume_dir',
    'explicit_result_dir',
    'training_note',
    'results_dir',
    'predictions_dir',
    'logs_dir',
    'timestamp_format',
    'log_interval',
    'val_log_interval',
    'post_training_evaluation',
})


def training_config_fingerprint(config: dict) -> str:
    """Hash every training-semantic setting while ignoring output controls."""
    semantic = {
        key: value for key, value in config.items()
        if key not in _RESUME_RUNTIME_CONFIG_KEYS
    }
    payload = json.dumps(
        semantic,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def read_synced_source_state(project_dir: Optional[str] = None) -> dict:
    """Read the immutable source identity emitted by ``sync_to_server.sh``."""
    path = os.path.join(project_dir or os.getcwd(), 'source_state.json')
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            state = json.load(handle)
        return state if isinstance(state, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def peak_host_memory_bytes() -> Optional[int]:
    """Return this process's peak resident set size in bytes when supported.

    Linux reports ``ru_maxrss`` in KiB while macOS reports bytes.  The formal
    training server is Linux; returning ``None`` on platforms without the
    standard ``resource`` module keeps local Windows tooling importable.
    """
    try:
        import resource

        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    if peak_rss < 0:
        return None
    return peak_rss if platform.system() == 'Darwin' else peak_rss * 1024


def summarize_dataset_protocol(dataset) -> dict:
    """Return compact cardinalities and index bounds for an audit manifest."""
    sequences = list(getattr(dataset, 'sequences', []))
    starts = [int(start) for start, _ in sequences]
    origins = sorted(set(starts))
    target_starts = [start + int(dataset.sequence_length) for start in starts]
    target_ends = [
        start + int(dataset.sequence_length) + int(dataset.prediction_length) - 1
        for start in starts
    ]
    spatial_windows = [
        {
            'lon_range': [float(value) for value in region.get('lon_range', [])],
            'lat_range': [float(value) for value in region.get('lat_range', [])],
            'region_type': region.get('region_type'),
        }
        for region in getattr(dataset, 'all_regions_data', [])
    ]
    spatial_payload = json.dumps(
        spatial_windows, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    )
    origin_payload = json.dumps(origins, separators=(',', ':'))
    times = list(getattr(dataset, 'times', []))
    damped_coefficients = {
        name: np.asarray(values, dtype=np.float32).tolist()
        for name, values in getattr(
            dataset, 'damped_persistence_coefficients', {}
        ).items()
    }
    return {
        'mode': getattr(dataset, 'mode', None),
        'samples': len(sequences),
        'spatial_windows': len(spatial_windows),
        'spatial_window_sha256': hashlib.sha256(
            spatial_payload.encode('utf-8')
        ).hexdigest(),
        'origin_count': len(origins),
        'origin_sha256': hashlib.sha256(origin_payload.encode('utf-8')).hexdigest(),
        'history_start_min': min(starts) if starts else None,
        'history_start_max': max(starts) if starts else None,
        'target_start_min': min(target_starts) if target_starts else None,
        'target_end_max': max(target_ends) if target_ends else None,
        'time_first': str(times[0]) if times else None,
        'time_last': str(times[-1]) if times else None,
        'stride_lon': float(getattr(dataset, 'stride_lon', 0.0)),
        'stride_lat': float(getattr(dataset, 'stride_lat', 0.0)),
        'window_grid_policy': getattr(dataset, '_WINDOW_GRID_POLICY', None),
        'cache_format_version': getattr(dataset, '_CACHE_FORMAT_VERSION', None),
        'damped_anomaly_persistence': {
            'estimation_split': 'train',
            'method': 'pooled_origin_regression_clipped_0_1_by_variable_lead_depth',
            'coefficients': damped_coefficients,
        },
    }


class OceanModelTrainer:
    """
    海洋模型训练器
    """

    def __init__(self, config: dict = None):
        """
        初始化训练器

        Args:
            config: 统一配置字典，如果为None则使用默认配置
        """
        if config is None:
            config = DEFAULT_CONFIG

        self.config = config

        # Set seed for reproducibility (if specified in config)
        seed = config.get('seed', None)
        if seed is not None:
            set_seed(seed)
        elif config.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        # 设置设备
        if config['device'] == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config['device'])

        # 更新配置中的设备，确保后续模型初始化使用正确的设备字符串
        config['device'] = str(self.device)
        print(f"使用设备: {self.device}")

        # 设置性能优化（仅在未指定随机种子时启用，以确保可复现性）
        if config.get('cudnn_benchmark', False) and config.get('seed') is None:
            torch.backends.cudnn.benchmark = True
            print("启用cuDNN benchmark")

        # 创建数据加载器
        print("创建数据加载器...")
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            config['data_path'],
            config,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            persistent_workers=config.get('persistent_workers', False),
            prefetch_factor=config.get('prefetch_factor', 2),
        )
        self.data_protocol = self._build_data_protocol()

        # 获取第一个批次来确定实际的输入输出维度
        print("检测数据维度...")
        sample_batch = next(iter(self.train_loader))
        if isinstance(sample_batch, (list, tuple)):
            if len(sample_batch) < 2:
                raise ValueError("训练数据批次缺少输入或目标张量")
            sample_input = sample_batch[0]
            sample_target = sample_batch[1]
        else:
            raise TypeError("训练数据批次类型不受支持，期望为tuple或list")
        actual_input_channels = sample_input.shape[2]  # [batch, seq, channels, height, width]
        actual_output_channels = sample_target.shape[2]  # [batch, pred, channels, height, width]

        print(f"实际输入通道数: {actual_input_channels}")
        print(f"实际输出通道数: {actual_output_channels}")

        # 更新配置
        config['actual_input_channels'] = actual_input_channels
        config['actual_output_channels'] = actual_output_channels

        def _serialize_channel_slices(slice_map):
            serialized = {}
            for name, channel_slice in getattr(slice_map, "items", lambda: [])():
                if isinstance(channel_slice, slice):
                    serialized[name] = [channel_slice.start, channel_slice.stop]
            return serialized

        train_dataset = getattr(self.train_loader, 'dataset', None)
        if train_dataset is not None:
            input_slices = _serialize_channel_slices(getattr(train_dataset, 'input_channel_slices', {}))
            target_slices = _serialize_channel_slices(getattr(train_dataset, 'target_channel_slices', {}))
            if input_slices:
                config['input_channel_slices'] = input_slices
            if target_slices:
                config['target_channel_slices'] = target_slices

        # 创建模型
        self.model = create_ocean_model(config).to(self.device)
        self.forward_model = self.model
        self.compile_requested = bool(config.get('compile_model', False))
        self.compile_active = False
        self.compile_fallback_used = False
        self._compiled_forward_verified = False
        if config.get('compile_model', False) and self.device.type == 'cuda' and hasattr(torch, 'compile'):
            try:
                self.forward_model = torch.compile(self.model, dynamic=True)
                self.compile_active = True
                print("启用 torch.compile（原始模型仍用于保存兼容权重）")
            except Exception as exc:
                if not config.get('allow_compile_fallback', False):
                    raise RuntimeError("配置要求 torch.compile，但初始化失败") from exc
                self.compile_fallback_used = True
                self.config['compile_model'] = False
                print(f"警告: torch.compile 启用失败，按配置回退 eager 模式: {exc}")

        self.amp_enabled = bool(config.get('mixed_precision', False) and self.device.type == 'cuda')
        self.grad_scaler = torch.amp.GradScaler('cuda', enabled=self.amp_enabled)
        print(f"混合精度训练: {'开启' if self.amp_enabled else '关闭'}")

        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        print(f"模型参数数量: {self.parameter_count:,}")
        expected_parameter_count = config.get('expected_parameter_count')
        if (
            expected_parameter_count is not None
            and self.parameter_count != int(expected_parameter_count)
        ):
            raise RuntimeError(
                "模型参数量偏离冻结协议："
                f"actual={self.parameter_count:,}, "
                f"expected={int(expected_parameter_count):,}"
            )

        # 设置损失函数和优化器
        self.criterion = nn.MSELoss()

        self.target_variables = config['target_variables']
        raw_target_slices = config.get('target_channel_slices', {})
        self.target_channel_slices = resolve_variable_slices(
            self.target_variables,
            raw_target_slices,
            actual_output_channels,
        )
        raw_weights = config.get('target_loss_weights', {})
        weight_sum = sum(float(raw_weights[name]) for name in self.target_variables)
        self.target_loss_weights = {
            name: float(raw_weights[name]) / weight_sum for name in self.target_variables
        }

        optimizer_type = str(config.get('optimizer_type', 'adam')).lower()
        optimizer_class = optim.AdamW if optimizer_type == 'adamw' else optim.Adam
        optimizer_betas = tuple(float(value) for value in config.get(
            'optimizer_betas', [0.9, 0.999]
        ))
        self.optimizer = optimizer_class(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=optimizer_betas,
        )
        print(
            f"优化器: {optimizer_class.__name__} "
            f"(lr={config['learning_rate']:.3g}, weight_decay={config['weight_decay']:.3g}, "
            f"betas={optimizer_betas})"
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['scheduler_factor'],
            patience=config['scheduler_patience'],
            min_lr=config['min_lr']
        )
        self.epochs_without_improvement = 0
        self.best_epoch = None

        # 训练状态
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.epoch_times = []
        self.epoch = 0

        # 创建结果目录（自动编号）或使用现有目录
        if config.get('resume_dir') and os.path.exists(config['resume_dir']):
            self.result_dir = config['resume_dir']
            print(f"恢复训练，使用现有目录: {self.result_dir}")
        elif config.get('explicit_result_dir'):
            self.result_dir = os.path.abspath(config['explicit_result_dir'])
            os.makedirs(self.result_dir, exist_ok=True)
            print(f"使用指定结果目录: {self.result_dir}")
        else:
            base_dir = config['results_dir']
            os.makedirs(base_dir, exist_ok=True)
            timestamp = datetime.now().strftime(config['timestamp_format'])
            note_slug = _sanitize_note_for_path(config.get('training_note', ''))
            result_dir_name = f"results_{timestamp}"
            if note_slug:
                result_dir_name += f"_{note_slug}"
            result_index = self._next_result_index(base_dir)
            numbered_name = f"{result_index}_{result_dir_name}"
            self.result_dir = os.path.join(base_dir, numbered_name)
            os.makedirs(self.result_dir, exist_ok=True)
            print(f"结果目录编号: {result_index} -> {self.result_dir}")

        # TensorBoard日志
        self.writer = SummaryWriter(os.path.join(self.result_dir, 'logs'))

        # 保存配置
        config_file = os.path.join(self.result_dir, config['config_filename'])
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)

        scalers_path = os.path.join(self.result_dir, config.get('scalers_filename', 'scalers.pkl'))
        with open(scalers_path, 'wb') as f:
            pickle.dump(train_dataset.scalers, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 保存训练备注
        if config.get('training_note'):
            note_path = os.path.join(self.result_dir, 'training_note.txt')
            # 如果是追加模式，使用 'a'
            mode = 'a' if config.get('resume_dir') else 'w'
            with open(note_path, mode, encoding='utf-8') as note_file:
                if mode == 'a':
                    note_file.write("\n--- 继续训练 ---\n")
                note_file.write(f"备注: {config['training_note']}\n")
                note_file.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    @staticmethod
    def _next_result_index(base_dir: str) -> int:
        existing = []
        try:
            for name in os.listdir(base_dir):
                path = os.path.join(base_dir, name)
                if not os.path.isdir(path):
                    continue
                prefix = name.split('_', 1)[0]
                if prefix.isdigit():
                    existing.append(int(prefix))
        except FileNotFoundError:
            return 0
        return max(existing) + 1 if existing else 0

    def _build_data_protocol(self) -> dict:
        """Freeze the exact sample geometry used by this run."""
        data_path = os.path.realpath(self.config['data_path'])
        data_stat = os.stat(data_path)
        return {
            'data_identity': {
                'path': data_path,
                'size_bytes': int(data_stat.st_size),
                'mtime_ns': int(data_stat.st_mtime_ns),
            },
            'split_context_policy': self.config.get(
                'split_context_policy', 'carry_history'
            ),
            'train': summarize_dataset_protocol(self.train_loader.dataset),
            'validation': summarize_dataset_protocol(self.val_loader.dataset),
            'test': summarize_dataset_protocol(self.test_loader.dataset),
            'loader_batches': {
                'train': len(self.train_loader),
                'validation': len(self.val_loader),
                'test': len(self.test_loader),
            },
        }

    def close(self) -> None:
        """Release persistent DataLoader workers and log handles promptly."""
        writer = getattr(self, 'writer', None)
        if writer is not None:
            writer.close()
        for name in ('train_loader', 'val_loader', 'test_loader'):
            loader = getattr(self, name, None)
            iterator = getattr(loader, '_iterator', None)
            shutdown = getattr(iterator, '_shutdown_workers', None)
            if callable(shutdown):
                shutdown()
            if loader is not None and hasattr(loader, '_iterator'):
                loader._iterator = None
            dataset = getattr(loader, 'dataset', None)
            source = getattr(dataset, 'dataset', None)
            close_source = getattr(source, 'close', None)
            if callable(close_source):
                close_source()

    def _forward(self, inputs):
        """执行模型前向；compile 协议默认严格，避免静默混用后端。"""
        try:
            outputs = self.forward_model(inputs)
            if self.compile_active:
                self._compiled_forward_verified = True
            return outputs
        except Exception as exc:
            can_fallback = (
                self.compile_active
                and not self._compiled_forward_verified
                and self.config.get('allow_compile_fallback', False)
                and not isinstance(exc, torch.cuda.OutOfMemoryError)
            )
            if not can_fallback:
                raise
            print(f"警告: compiled 首次前向失败，按配置回退 eager 模式: {exc}")
            self.forward_model = self.model
            self.compile_active = False
            self.compile_fallback_used = True
            self.config['compile_model'] = False
            return self.model(inputs)

    def compute_gradient_loss(self, pred, target):
        """
        计算空间梯度损失。默认分别匹配 x/y 分量，保留梯度方向；
        magnitude 模式仅用于显式敏感性实验。
        """
        # pred, target: (B, T, C, H, W)
        # 计算空间梯度 (Sobel算子近似)
        # 定义Sobel核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3) / 8.0

        b, t, c, h, w = pred.shape
        pred_flat = pred.reshape(b*t*c, 1, h, w)
        target_flat = target.reshape(b*t*c, 1, h, w)

        # 计算梯度
        pred_padded = F.pad(pred_flat, (1, 1, 1, 1), mode='replicate')
        target_padded = F.pad(target_flat, (1, 1, 1, 1), mode='replicate')
        grad_pred_x = F.conv2d(pred_padded, sobel_x)
        grad_pred_y = F.conv2d(pred_padded, sobel_y)
        grad_target_x = F.conv2d(target_padded, sobel_x)
        grad_target_y = F.conv2d(target_padded, sobel_y)

        if self.config.get('gradient_loss_mode', 'vector') == 'magnitude':
            grad_pred_mag = torch.sqrt(grad_pred_x.square() + grad_pred_y.square() + 1e-6)
            grad_target_mag = torch.sqrt(grad_target_x.square() + grad_target_y.square() + 1e-6)
            return F.mse_loss(grad_pred_mag, grad_target_mag)

        return 0.5 * (
            F.mse_loss(grad_pred_x, grad_target_x)
            + F.mse_loss(grad_pred_y, grad_target_y)
        )

    def compute_weighted_loss(self, outputs, targets):
        """
        计算温度和盐度的加权损失

        Args:
            outputs: 模型输出 (batch_size, pred_len, channels, height, width)
            targets: 目标值 (batch_size, pred_len, channels, height, width)

        Returns:
            加权损失值
        """
        # 计算梯度损失
        grad_loss = outputs.new_zeros(())
        if self.config.get('use_gradient_loss', True):
            grad_loss = self.compute_gradient_loss(outputs, targets)
            grad_weight = self.config.get('gradient_loss_weight', 0.1) # 默认权重0.1
            grad_loss = grad_loss * grad_weight

        variable_losses = {}
        weighted_loss = outputs.new_zeros(())
        for var_name in self.target_variables:
            ch_slice = self.target_channel_slices[var_name]
            var_loss = self.criterion(outputs[:, :, ch_slice], targets[:, :, ch_slice])
            variable_losses[var_name] = var_loss
            weighted_loss = weighted_loss + self.target_loss_weights[var_name] * var_loss

        total_loss = weighted_loss + grad_loss
        detached_losses = {name: float(value.detach().item()) for name, value in variable_losses.items()}
        return total_loss, detached_losses, float(grad_loss.detach().item())

    def train_epoch(self) -> float:
        """
        训练一个epoch

        Returns:
            平均训练损失
        """
        self.model.train()
        self.forward_model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        valid_batches = 0
        valid_samples = 0

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch+1} 训练')
        for batch_idx, batch in enumerate(progress_bar):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # 前向传播
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                outputs = self._forward(inputs)
                loss, variable_losses, grad_loss_val = self.compute_weighted_loss(outputs, targets)

            # 非有限 batch 会破坏跨实验公平性，必须立即失败并保留现场。
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"epoch {self.epoch + 1} batch {batch_idx} 训练损失非有限: {loss.item()}"
                )

            # 检查输出是否包含异常值
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                raise FloatingPointError(
                    f"epoch {self.epoch + 1} batch {batch_idx} 训练输出包含 NaN/Inf"
                )

            # 反向传播
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=float(self.config.get('grad_clip_norm', 1.0)),
            )

            # 检查梯度
            if not torch.isfinite(grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"epoch {self.epoch + 1} batch {batch_idx} 梯度范数非有限: {grad_norm.item()}"
                )

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            batch_samples = int(inputs.shape[0])
            total_loss += loss.item() * batch_samples
            valid_batches += 1
            valid_samples += batch_samples

            # 更新进度条
            if len(self.target_variables) > 1:
                postfix = {'Loss': f'{loss.item():.6f}', 'Avg': f'{total_loss/valid_samples:.6f}'}
                postfix.update({name: f'{value:.6f}' for name, value in variable_losses.items()})
                postfix['Grad'] = f'{grad_loss_val:.6f}'
                progress_bar.set_postfix(postfix)
            else:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Avg Loss': f'{total_loss/valid_samples:.6f}'
                })

            # 记录到TensorBoard
            global_step = self.epoch * num_batches + batch_idx
            self.writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
            for var_name, var_loss in variable_losses.items():
                self.writer.add_scalar(f'Loss/Train_{var_name}', var_loss, global_step)
            self.writer.add_scalar('Loss/Train_Gradient', grad_loss_val, global_step)

        if valid_batches == 0:
            raise RuntimeError("训练 epoch 中没有有效 batch")
        avg_loss = total_loss / valid_samples
        return avg_loss

    def validate_epoch(self) -> float:
        """
        验证一个epoch

        Returns:
            平均验证损失
        """
        self.model.eval()
        self.forward_model.eval()
        total_objective = 0.0
        total_selection_loss = 0.0
        num_batches = len(self.val_loader)

        # 检查验证集是否为空
        if num_batches == 0:
            print("警告: 验证集为空，跳过验证")
            return float('inf')  # 返回无穷大表示无法验证

        valid_batches = 0  # 记录有效批次数量
        valid_samples = 0

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f'Epoch {self.epoch+1} 验证')

            for batch in progress_bar:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    inputs, targets, _ = batch
                else:
                    inputs, targets = batch

                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                    outputs = self._forward(inputs)
                    loss, variable_losses, grad_loss_val = self.compute_weighted_loss(outputs, targets)

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"epoch {self.epoch + 1} validation batch {valid_batches} 损失非有限: {loss.item()}"
                    )
                if not torch.isfinite(outputs).all():
                    raise FloatingPointError(
                        f"epoch {self.epoch + 1} validation batch {valid_batches} 输出包含 NaN/Inf"
                    )
                batch_samples = int(inputs.shape[0])
                selection_loss = sum(
                    self.target_loss_weights[name] * value
                    for name, value in variable_losses.items()
                )
                total_objective += loss.item() * batch_samples
                total_selection_loss += selection_loss * batch_samples
                valid_batches += 1
                valid_samples += batch_samples

                if len(self.target_variables) > 1:
                    postfix = {
                        'ValMSE': f'{selection_loss:.6f}',
                        'Avg': f'{total_selection_loss/max(valid_samples, 1):.6f}',
                        'Obj': f'{loss.item():.6f}',
                    }
                    postfix.update({name: f'{value:.6f}' for name, value in variable_losses.items()})
                    postfix['Grad'] = f'{grad_loss_val:.6f}'
                    progress_bar.set_postfix(postfix)
                else:
                    progress_bar.set_postfix({
                        'Val Loss': f'{loss.item():.6f}',
                        'Avg Val Loss': (
                            f'{total_selection_loss / max(valid_samples, 1):.6f}'
                        ),
                    })

                # 记录验证损失到TensorBoard
                if valid_batches == 1:  # 只记录第一个批次，避免过多记录
                    val_step = self.epoch
                    self.writer.add_scalar('Loss/Val_Batch', loss.item(), val_step)
                    for var_name, var_loss in variable_losses.items():
                        self.writer.add_scalar(f'Loss/Val_{var_name}', var_loss, val_step)
                    self.writer.add_scalar('Loss/Val_Gradient', grad_loss_val, val_step)

        # 防止除零错误
        if valid_batches == 0:
            print("警告: 所有验证批次都包含异常值，无法计算验证损失")
            return float('inf')

        # Checkpoint selection excludes gradient regularization so the same
        # criterion is used by full/no-gradient/magnitude ablations.
        return total_selection_loss / valid_samples

    def save_checkpoint(self, is_best: bool = False):
        """
        保存模型检查点

        Args:
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': self.epoch,
            'next_epoch': self.epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epoch_times': self.epoch_times,
            'epochs_without_improvement': self.epochs_without_improvement,
            'best_epoch': self.best_epoch,
            'grad_scaler_state_dict': self.grad_scaler.state_dict(),
            'rng_state': capture_rng_state(),
            'config': self.config,
            'training_config_fingerprint': training_config_fingerprint(self.config),
            'source_hash': read_synced_source_state().get('source_hash'),
            'training_source_hash': read_synced_source_state().get('training_source_hash'),
        }

        # 保存最新检查点
        if self.config.get('save_last', True):
            torch.save(checkpoint, os.path.join(self.result_dir, self.config['checkpoint_filename']))

        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, os.path.join(self.result_dir, self.config['model_filename']))
            print(f"保存最佳模型，验证损失: {self.best_val_loss:.6f}")
        if not self.config.get('save_best_only', True):
            torch.save(checkpoint, os.path.join(self.result_dir, f'epoch_{self.epoch + 1:04d}.pth'))

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点

        Args:
            checkpoint_path: 检查点文件路径
        """
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            saved_config = checkpoint.get('config')
            saved_fingerprint = checkpoint.get('training_config_fingerprint')
            if saved_fingerprint is None and isinstance(saved_config, dict):
                saved_fingerprint = training_config_fingerprint(saved_config)
            current_fingerprint = training_config_fingerprint(self.config)
            if saved_fingerprint is None:
                raise RuntimeError(
                    '检查点缺少训练配置，拒绝不可审计的恢复；请从新结果目录重新训练'
                )
            if saved_fingerprint != current_fingerprint:
                raise RuntimeError(
                    '检查点训练配置与当前配置不一致，拒绝混合实验协议恢复'
                )
            next_epoch = int(checkpoint.get('next_epoch', checkpoint.get('epoch', -1) + 1))
            if int(self.config.get('epochs', 0)) < next_epoch:
                raise RuntimeError(
                    f"当前 epochs={self.config.get('epochs')} 小于检查点下一轮 {next_epoch}，"
                    '拒绝产生只评估未训练的伪恢复运行'
                )

            saved_source_hash = checkpoint.get('training_source_hash')
            current_source_hash = read_synced_source_state().get('training_source_hash')
            if self.config.get('strict_resume_provenance', True):
                if not saved_source_hash or not current_source_hash:
                    raise RuntimeError(
                        '严格恢复要求检查点和当前目录都具有 source_hash；'
                        '旧检查点请重新训练，或显式关闭 strict_resume_provenance'
                    )
                if saved_source_hash != current_source_hash:
                    raise RuntimeError(
                        '检查点源码哈希与当前服务器源码不一致，拒绝跨版本恢复'
                    )

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if checkpoint.get('grad_scaler_state_dict'):
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])

            self.epoch = next_epoch
            self.best_val_loss = checkpoint['best_val_loss']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.epoch_times = checkpoint.get('epoch_times', [])
            self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            self.best_epoch = checkpoint.get('best_epoch')
            restored_rng = restore_rng_state(checkpoint.get('rng_state'))

            print(f"从检查点恢复训练，epoch: {self.epoch}, 最佳验证损失: {self.best_val_loss:.6f}")
            if not restored_rng:
                print("警告: 旧检查点不含 RNG 状态；后续训练不能视为逐步完全复现")
        else:
            print(f"检查点文件不存在: {checkpoint_path}")

    def _load_best_model_weights(self) -> bool:
        """尝试加载最佳模型权重。"""
        best_model_path = os.path.join(self.result_dir, self.config['model_filename'])
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict)
            if 'best_val_loss' in checkpoint:
                self.best_val_loss = checkpoint['best_val_loss']
            self.best_epoch = checkpoint.get('best_epoch', checkpoint.get('epoch', -1) + 1)
            return True
        return False

    def plot_training_curves(self):
        """
        绘制训练曲线
        """
        if len(self.train_losses) == 0:
            return

        plt.figure(figsize=(10, 6))

        # 损失曲线
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', linewidth=2, label='训练目标（含正则项）')
        plt.plot(epochs, self.val_losses, 'r-', linewidth=2, label='验证加权 MSE（选模指标）')
        plt.xlabel('训练轮次 (Epoch)', fontsize=12)
        plt.ylabel('损失值 (Loss)', fontsize=12)
        plt.title('训练和验证损失曲线', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 添加一些美化
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def train(self, epochs: int = None, resume: bool = False):
        """
        训练模型

        Args:
            epochs: 训练轮数，如果为None则使用配置中的默认值
            resume: 是否从检查点恢复训练
        """
        if epochs is None:
            epochs = self.config['epochs']

        if resume or self.config.get('resume_dir'):
            checkpoint_path = os.path.join(self.result_dir, self.config['checkpoint_filename'])
            self.load_checkpoint(checkpoint_path)

        print(f"开始训练，总共 {epochs} 个epoch")
        print(f"结果保存在: {self.result_dir}")

        start_epoch = self.epoch

        for epoch in range(start_epoch, epochs):
            self.epoch = epoch
            start_time = time.time()

            # 训练
            train_loss = self.train_epoch()
            if not np.isfinite(train_loss):
                raise RuntimeError(f"epoch {epoch + 1} 训练损失非有限值: {train_loss}")

            print(f"Epoch {epoch+1} 训练完成，开始验证...")
            # 验证
            val_loss = self.validate_epoch()
            if not np.isfinite(val_loss):
                raise RuntimeError(f"epoch {epoch + 1} 验证损失非有限值: {val_loss}")

            # 更新学习率
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']

            # 如果学习率发生变化，打印通知
            if new_lr != old_lr:
                print(f"学习率调整: {old_lr:.2e} -> {new_lr:.2e}")

            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # 记录到TensorBoard
            self.writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
            self.writer.add_scalar('Loss/Val_Epoch', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # 检查是否为最佳模型（跳过无穷大的验证损失）
            is_best = False
            min_delta = float(self.config.get('min_delta', 0.0))
            if np.isfinite(val_loss) and val_loss < self.best_val_loss - min_delta:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                is_best = True
                self.epochs_without_improvement = 0
                print(f"[BEST] New best val loss: {val_loss:.6f}")
            elif np.isinf(val_loss):
                print("[WARN] Val loss is inf, skipping model update")
            else:
                self.epochs_without_improvement += 1

            epoch_time = time.time() - start_time
            self.epoch_times.append(epoch_time)

            # 保存检查点
            self.save_checkpoint(is_best)

            # 绘制训练曲线
            if (epoch + 1) % 10 == 0:
                self.plot_training_curves()

            # 打印epoch信息
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"训练损失: {train_loss:.6f} | "
                  f"验证损失: {val_loss:.6f} | "
                  f"最佳验证损失: {self.best_val_loss:.6f} | "
                  f"学习率: {self.optimizer.param_groups[0]['lr']:.2e} | "
                  f"时间: {epoch_time:.1f}s")

            # 检查损失是否为NaN
            if self.epochs_without_improvement >= int(self.config.get('early_stopping_patience', 20)):
                print(f"早停触发: 连续 {self.epochs_without_improvement} 个 epoch 无显著改善")
                break

        # 最终保存
        self.plot_training_curves()

        self.writer.close()

        if not np.isfinite(self.best_val_loss):
            raise RuntimeError("训练结束但没有得到有限的最佳验证损失")

        print(f"训练完成！最佳验证损失: {self.best_val_loss:.6f}")
        print(f"结果保存在: {self.result_dir}")

    def evaluate(self, use_best_model: bool = True, split: str = 'validation') -> Dict:
        """
        评估模型

        Args:
            use_best_model: 是否使用最佳模型
            split: 评估 validation，或在协议冻结后显式指定 test

        Returns:
            评估结果字典
        """
        evaluation_started = time.perf_counter()
        evaluation_timings = {}
        if split not in {'validation', 'test'}:
            raise ValueError(f'不支持的评估 split: {split}')
        evaluation_loader = self.val_loader if split == 'validation' else self.test_loader
        split_label = '验证' if split == 'validation' else '测试'

        if use_best_model:
            if self._load_best_model_weights():
                print("使用最佳模型进行评估")
            else:
                print("未找到最佳模型权重，使用当前权重评估")

        self.model.eval()
        self.forward_model.eval()

        # 在冻结的 validation 或 test split 上评估
        test_loss = 0.0
        evaluation_data_loss = 0.0
        test_samples = 0
        num_batches = len(evaluation_loader)

        predictions = []
        targets_list = []
        sample_indices = []

        with torch.no_grad():
            for batch in tqdm(evaluation_loader, desc=f"{split_label}集评估"):
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    inputs, targets, batch_indices = batch
                    sample_indices.extend([int(idx) for idx in batch_indices])
                else:
                    inputs, targets = batch

                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                    outputs = self._forward(inputs)
                    loss, variable_losses, _ = self.compute_weighted_loss(outputs, targets)
                batch_samples = int(inputs.shape[0])
                test_loss += loss.item() * batch_samples
                evaluation_data_loss += sum(
                    self.target_loss_weights[name] * value
                    for name, value in variable_losses.items()
                ) * batch_samples
                test_samples += batch_samples

                # 收集预测结果
                predictions.append(outputs.cpu().numpy())
                targets_list.append(targets.cpu().numpy())

        # 检查是否有测试数据
        if num_batches == 0:
            print("警告: 测试集为空，跳过评估")
            results = {
                'test_loss': float('inf'),
                'mae': float('inf'),
                'rmse': float('inf'),
                'correlation': 0.0
            }
            return results

        avg_test_loss = test_loss / test_samples
        avg_evaluation_data_loss = evaluation_data_loss / test_samples

        # 检查是否有预测结果
        if len(predictions) == 0:
            print("警告: 没有预测结果，跳过指标计算")
            results = {
                'test_loss': avg_test_loss,
                'mae': float('inf'),
                'rmse': float('inf'),
                'correlation': 0.0
            }
            return results

        # 计算其他指标
        predictions = np.concatenate(predictions, axis=0).astype(np.float32, copy=False)
        targets_array = np.concatenate(targets_list, axis=0).astype(np.float32, copy=False)
        evaluation_timings['inference_and_collection'] = (
            time.perf_counter() - evaluation_started
        )
        print(
            '[评估阶段] 推理与结果汇集完成: '
            f"{evaluation_timings['inference_and_collection']:.1f}s"
        )

        phase_started = time.perf_counter()
        eval_dataset = getattr(evaluation_loader, 'dataset', None)
        if eval_dataset is None:
            raise RuntimeError('评估 DataLoader 缺少 dataset，无法恢复物理量与样本溯源')
        channel_slices = resolve_variable_slices(
            self.target_variables,
            getattr(eval_dataset, 'target_channel_slices', {}),
            predictions.shape[2],
        )
        raw_depth_values = getattr(eval_dataset, 'levels', None) if eval_dataset is not None else None
        depth_values = (
            [float(value) for value in np.asarray(raw_depth_values).reshape(-1)]
            if raw_depth_values is not None else None
        )
        if len(sample_indices) != predictions.shape[0]:
            raise RuntimeError(
                '预测样本与 sample_indices 数量不一致；拒绝重排后伪造默认索引: '
                f'{predictions.shape[0]} != {len(sample_indices)}'
            )
        if not hasattr(eval_dataset, 'inverse_transform_targets'):
            raise RuntimeError('评估 dataset 不支持严格物理量恢复')
        if not hasattr(eval_dataset, 'build_reference_forecasts'):
            raise RuntimeError('评估 dataset 不支持冻结基线构造')

        try:
            physical_predictions = eval_dataset.inverse_transform_targets(
                predictions,
                sample_indices=sample_indices,
            ).astype(np.float32, copy=False)
            physical_targets = eval_dataset.inverse_transform_targets(
                targets_array,
                sample_indices=sample_indices,
            ).astype(np.float32, copy=False)
        except Exception as exc:
            raise RuntimeError('目标物理量恢复失败；拒绝改变指标空间后继续') from exc

        provenance = {}
        if hasattr(eval_dataset, 'build_sample_provenance'):
            provenance = eval_dataset.build_sample_provenance(sample_indices)
        evaluation_timings['schema_provenance_and_physical_recovery'] = (
            time.perf_counter() - phase_started
        )
        print(
            '[评估阶段] 通道核验、溯源与物理量恢复完成: '
            f"{evaluation_timings['schema_provenance_and_physical_recovery']:.1f}s"
        )

        # 物理量和标准化基线分两阶段构造。三套完整 baseline 若同时保留两个
        # metric space，会无谓常驻六个 validation 张量。
        phase_started = time.perf_counter()
        try:
            physical_baselines = eval_dataset.build_reference_forecasts(
                sample_indices=sample_indices,
                spaces=('physical',),
            ).get('physical', {})
        except Exception as exc:
            raise RuntimeError('物理量冻结基线构造失败；拒绝输出不完整指标') from exc

        physical_units_available = True
        physical_report = compute_metric_report(
            physical_predictions,
            physical_targets,
            self.target_variables,
            channel_slices=channel_slices,
            baselines=physical_baselines,
            metric_space='physical',
            depth_values=depth_values,
        )
        evaluation_timings['physical_metrics'] = time.perf_counter() - phase_started
        print(
            '[评估阶段] 物理量指标与基线完成: '
            f"{evaluation_timings['physical_metrics']:.1f}s"
        )

        phase_started = time.perf_counter()
        stratified_reports = {}
        if provenance:
            stratified_reports['by_origin'] = compute_sample_group_report(
                physical_predictions,
                physical_targets,
                provenance['origin_ids'],
                self.target_variables,
                channel_slices=channel_slices,
                baselines=physical_baselines,
                metric_space='physical',
                depth_values=depth_values,
            )
            stratified_reports['by_climatology_period'] = compute_period_group_report(
                physical_predictions,
                physical_targets,
                np.asarray(provenance['target_period_ids']),
                self.target_variables,
                channel_slices=channel_slices,
                baselines=physical_baselines,
                metric_space='physical',
                depth_values=depth_values,
            )
        evaluation_timings['stratified_metrics'] = time.perf_counter() - phase_started
        print(
            '[评估阶段] 起报时次与气候月份分层指标完成: '
            f"{evaluation_timings['stratified_metrics']:.1f}s"
        )

        del physical_predictions, physical_targets, physical_baselines
        gc.collect()

        phase_started = time.perf_counter()
        try:
            normalized_baselines = eval_dataset.build_reference_forecasts(
                sample_indices=sample_indices,
                spaces=('normalized',),
            ).get('normalized', {})
        except Exception as exc:
            raise RuntimeError('标准化冻结基线构造失败；拒绝输出不完整指标') from exc

        normalized_report = compute_metric_report(
            predictions,
            targets_array,
            self.target_variables,
            channel_slices=channel_slices,
            baselines=normalized_baselines,
            metric_space='normalized',
            depth_values=depth_values,
            include_depth=False,
        )
        del normalized_baselines
        gc.collect()
        evaluation_timings['normalized_metrics'] = time.perf_counter() - phase_started
        evaluation_timings['total_before_serialization'] = (
            time.perf_counter() - evaluation_started
        )
        print(
            '[评估阶段] 标准化指标完成: '
            f"{evaluation_timings['normalized_metrics']:.1f}s; "
            '写盘前总计 '
            f"{evaluation_timings['total_before_serialization']:.1f}s"
        )

        normalized_overall = normalized_report.get('overall') or {}
        normalized_mae = normalized_overall.get('mae')
        normalized_rmse = normalized_overall.get('rmse')
        normalized_r2 = normalized_overall.get('r2')
        if normalized_mae is None or normalized_rmse is None:
            raise RuntimeError('标准化指标没有有限的 MAE/RMSE')

        # 多物理变量不能混合 °C/PSU；整体指标改用无量纲 normalized/anomaly 空间。
        physical_overall = physical_report.get('overall') if physical_units_available else None
        aggregate_metrics = physical_overall or normalized_report.get('overall') or {}
        aggregate_units = 'physical' if physical_overall is not None else 'normalized/anomaly'
        mae = aggregate_metrics.get('mae')
        rmse = aggregate_metrics.get('rmse')
        correlation = aggregate_metrics.get('correlation')
        r2 = aggregate_metrics.get('r2')

        results = {
            'evaluation_split': split,
            'evaluation_loss': float(avg_test_loss),
            'evaluation_data_loss': float(avg_evaluation_data_loss),
            'test_loss': float(avg_test_loss),
            'mae': float(mae) if mae is not None else None,
            'rmse': float(rmse) if rmse is not None else None,
            'metric_units': aggregate_units,
            'physical_mae': float(physical_overall['mae']) if physical_overall else None,
            'physical_rmse': float(physical_overall['rmse']) if physical_overall else None,
            'normalized_mae': float(normalized_mae),
            'normalized_rmse': float(normalized_rmse),
            'normalized_r2': float(normalized_r2) if normalized_r2 is not None and not np.isnan(normalized_r2) else None,
            'correlation': float(correlation) if correlation is not None else None,
            'r2': float(r2) if r2 is not None else None,
            'physical_report': physical_report if physical_units_available else None,
            'normalized_report': normalized_report,
            'baseline_reports': {
                'physical': physical_report.get('baselines', {}) if physical_units_available else {},
                'normalized': normalized_report.get('baselines', {}),
            },
            'baseline_comparison': {
                'physical': physical_report.get('comparison', {}) if physical_units_available else {},
                'normalized': normalized_report.get('comparison', {}),
            },
            'stratified_reports': stratified_reports,
            'evaluation_provenance': provenance,
            'evaluation_timings_seconds': {
                name: float(value) for name, value in evaluation_timings.items()
            },
            # uppercase aliases for ablation script compatibility
            'MAE': float(mae) if mae is not None else None,
            'RMSE': float(rmse) if rmse is not None else None,
            'R^2': float(r2) if r2 is not None else None,
        }

        # 计算分变量指标
        for var_name in self.target_variables:
            # Reuse the unit-safe report instead of rescanning the full arrays.
            primary_var = (
                physical_report if physical_units_available else normalized_report
            )['by_variable'][var_name]
            normalized_var = normalized_report['by_variable'][var_name]
            var_mae = primary_var.get('mae')
            var_rmse = primary_var.get('rmse')
            var_normalized_mae = normalized_var.get('mae')
            var_normalized_rmse = normalized_var.get('rmse')
            var_corr = primary_var.get('correlation')
            var_r2 = primary_var.get('r2')

            results[f'mae_{var_name}'] = float(var_mae) if var_mae is not None else None
            results[f'rmse_{var_name}'] = float(var_rmse) if var_rmse is not None else None
            results[f'physical_mae_{var_name}'] = float(var_mae) if physical_units_available and var_mae is not None else None
            results[f'physical_rmse_{var_name}'] = float(var_rmse) if physical_units_available and var_rmse is not None else None
            results[f'normalized_mae_{var_name}'] = float(var_normalized_mae) if var_normalized_mae is not None else None
            results[f'normalized_rmse_{var_name}'] = float(var_normalized_rmse) if var_normalized_rmse is not None else None
            results[f'correlation_{var_name}'] = float(var_corr) if var_corr is not None else None
            results[f'r2_{var_name}'] = float(var_r2) if var_r2 is not None else None

        print(f"{split_label}结果:")
        if not np.isnan(avg_test_loss):
            print(f"  测试损失: {avg_test_loss:.6f}")
        else:
            print(f"  测试损失: nan")
        if mae is not None and not np.isnan(mae):
            print(f"  MAE: {mae:.6f}")
        else:
            print(f"  MAE: nan")
        if rmse is not None and not np.isnan(rmse):
            print(f"  RMSE: {rmse:.6f} ({results['metric_units']})")
        else:
            print(f"  RMSE: nan")
        normalized_r2_text = f"{normalized_r2:.6f}" if normalized_r2 is not None and not np.isnan(normalized_r2) else "nan"
        print(f"  Normalized MAE/RMSE/R^2: {normalized_mae:.6f} / {normalized_rmse:.6f} / {normalized_r2_text}")
        if correlation is not None and not np.isnan(correlation):
            print(f"  相关系数: {correlation:.6f}")
        else:
            print(f"  相关系数: nan")
        if r2 is not None and not np.isnan(r2):
            print(f"  R^2: {r2:.6f}")
        else:
            print("  R^2: nan")

        # 打印分变量结果
        for var_name in self.target_variables:
            print(f"  [{var_name}] MAE: {results[f'mae_{var_name}']:.6f} | RMSE: {results[f'rmse_{var_name}']:.6f} | R^2: {results[f'r2_{var_name}'] if results[f'r2_{var_name}'] is not None else 'nan'}")

        primary_report = physical_report if physical_units_available else normalized_report
        macro_r2 = (
            primary_report.get('macro_field', {})
            .get('dimensionless_overall', {})
            .get('r2', {})
            .get('mean')
        )
        print("  去背景稳健指标:")
        print(f"    Macro-field R^2 mean: {macro_r2:.6f}" if macro_r2 is not None else "    Macro-field R^2 mean: nan")
        spatial_by_var = primary_report.get('spatial_mean_removed', {}).get('by_variable', {})
        clim_by_var = primary_report.get('climatology_residual', {}).get('by_variable', {})
        for var_name in self.target_variables:
            spatial_r2 = spatial_by_var.get(var_name, {}).get('r2')
            clim_metrics = clim_by_var.get(var_name, {})
            spatial_text = f"{spatial_r2:.6f}" if spatial_r2 is not None else "nan"
            clim_r2 = clim_metrics.get('r2')
            clim_corr = clim_metrics.get('correlation')
            clim_r2_text = f"{clim_r2:.6f}" if clim_r2 is not None else "nan"
            clim_corr_text = f"{clim_corr:.6f}" if clim_corr is not None else "nan"
            print(f"    [{var_name}] Spatial-mean-removed R^2: {spatial_text}")
            if clim_metrics:
                print(f"    [{var_name}] Climatology-residual Corr/R^2: {clim_corr_text} / {clim_r2_text}")

        print("  Baseline 对比 (按变量计算，skill > 0 表示优于基线):")
        for space_name, comparisons in results['baseline_comparison'].items():
            if not comparisons:
                continue
            print(f"    [{space_name}]")
            for baseline_name, comparison in comparisons.items():
                macro = comparison.get('macro', {})
                macro_skill = macro.get('mse_skill', {}).get('mean')
                macro_text = f"{macro_skill:.6f}" if macro_skill is not None else "nan"
                print(f"      {baseline_name}: macro MSE skill={macro_text}")
                for var_name, values in comparison.get('by_variable', {}).items():
                    skill = values.get('mse_skill')
                    improvement = values.get('rmse_improvement_pct')
                    skill_text = f"{skill:.6f}" if skill is not None else "nan"
                    improvement_text = f"{improvement:.2f}%" if improvement is not None else "nan"
                    print(f"        [{var_name}] skill={skill_text}, RMSE improvement={improvement_text}")

        # 保存评估结果
        evaluation_filename = (
            'evaluation_results.json' if split == 'test' else 'validation_results.json'
        )
        with open(os.path.join(self.result_dir, evaluation_filename), 'w') as f:
            json.dump(results, f, indent=4)

        return results


def main():
    """
    主函数
    """
    import argparse
    parser = argparse.ArgumentParser(description="海洋数据 ConvLSTM 训练脚本")
    parser.add_argument('--config', type=str, default=None,
                        help='实验配置 JSON；未提供字段继承默认配置')
    parser.add_argument('--note', type=str, default='',
                        help='本次训练备注（可选，追加到结果目录名 + 保存到 training_note.txt）')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖配置中的训练轮数')
    parser.add_argument('--lr', type=float, default=None,
                        help='覆盖配置中的学习率')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='覆盖配置中的批次大小')
    parser.add_argument('--resume_dir', type=str, default=None,
                        help='从指定结果目录的 latest checkpoint 继续训练')
    parser.add_argument('--seed', type=int, default=None,
                        help='覆盖随机种子')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='使用确定性的结果目录，供批量实验可靠续跑')
    parser.add_argument('--overrides_json', type=str, default=None,
                        help='JSON 对象形式的配置覆盖，供可审计实验矩阵使用')
    args = parser.parse_args()

    # 设置中文字体
    setup_chinese_fonts()

    # 使用统一配置文件
    config = DEFAULT_CONFIG.copy()
    if args.config:
        config = merge_configs(load_config(args.config), config)
    if args.overrides_json:
        overrides = json.loads(args.overrides_json)
        if not isinstance(overrides, dict):
            raise ValueError('--overrides_json 必须解析为 JSON 对象')
        config.update(overrides)

    # 命令行参数覆盖
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.lr is not None:
        config['learning_rate'] = args.lr
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.resume_dir:
        config['resume_dir'] = args.resume_dir
    if args.seed is not None:
        config['seed'] = args.seed
    if args.result_dir:
        config['explicit_result_dir'] = args.result_dir

    # 验证配置
    validate_config(config)

    model_type = str(config.get('model_type', 'convlstm')).lower()
    print("海洋温盐预测模型训练")
    print("=" * 50)
    print("使用统一配置文件:")
    print(f"  数据路径: {config['data_path']}")
    print(f"  序列长度: {config['sequence_length']}")
    print(f"  预测长度: {config['prediction_length']}")
    print(f"  隐藏维度: {config['hidden_dims']}")
    print(f"  学习率: {config['learning_rate']}")
    print(f"  批次大小: {config['batch_size']}")
    print(f"  训练轮数: {config['epochs']}")
    print(f"  Dropout: {config['dropout']}")
    print("-" * 50)
    print(f"模型类型: {model_type}")
    print(f"  [PosEncode] 位置编码: {'开启' if config.get('enable_positional_encoding', False) else '关闭'}")
    print(f"  [TimeEncode] 时间编码: {'开启' if config.get('enable_time_encoding', False) else '关闭'}")
    if model_type in {
        'tsc_fusion', 'tscglobal', 'tsc_global_axiom_ensemble',
        'tsc-spectrum-axiom-ensemble', 'tsc_spectrum_axiom_ensemble',
    }:
        print("  TSC-Fusion 组件状态:")
        print(f"    [TSC] 热盐结构记忆: {'关闭(消融)' if config.get('ablation_disable_tsc', False) else '开启'}")
        print(f"    [Spectral] 全局频谱分支: {'关闭(消融)' if config.get('ablation_disable_spectral', False) else '开启'}")
        print(f"    [3D] 三维结构分支: {'关闭(消融)' if config.get('ablation_disable_3d', False) else '开启'}")
        print(f"    [Ensemble] 门控集成: {'关闭(消融)' if config.get('ablation_disable_ensemble', False) else '开启'}")
        print(f"    [GTB] 跨窗口上下文: {'开启' if config.get('enable_global_token_bank', False) else '关闭'}")
        print(f"    [FusionTransformer] 层数: {int(config.get('tsc_fusion_transformer_layers', 0))}")
    elif model_type in {
        'ofb_fourcastnet', 'ofb-fourcastnet',
        'ofb_climax', 'ofb-climax',
        'ofb_swin', 'ofb-swin',
    }:
        provenance = config.get('baseline_provenance', {})
        print(
            "  [Baseline] OceanForecastBench architecture adapter: "
            f"{provenance.get('method_name', model_type)}"
        )
        print("  [Baseline] 本地 ORAS5 从头训练；不是官方权重或官方成绩")
    print(f"  [Persistence] 持久性残差: {'开启' if config.get('enable_persistence_residual', True) else '关闭'}")
    if config.get('enable_persistence_residual', True):
        print(f"  [Persistence] 模式: {config.get('persistence_residual_mode', 'learned_scale')}")
    print("=" * 50)

    training_note = args.note.strip()
    if training_note:
        print(f"本次训练备注: {training_note}")
        config['training_note'] = training_note
    else:
        print("未输入备注，使用默认配置继续训练。")

    trainer = None
    run_started = time.time()
    phase_timings = {}
    try:
        # 创建训练器
        phase_started = time.perf_counter()
        trainer = OceanModelTrainer(config)
        phase_timings['initialization'] = time.perf_counter() - phase_started

        # 训练模型（使用配置中的默认epochs）
        phase_started = time.perf_counter()
        trainer.train()
        phase_timings['training'] = time.perf_counter() - phase_started

        # 校准可只看训练期 validation loss；消融筛选输出 validation 报告；
        # 测试集仅在协议与候选模块冻结后运行。
        evaluation_scope = config.get('post_training_evaluation', 'validation')
        evaluation_lock_path = os.environ.get('TSC_POST_EVAL_LOCK')
        if evaluation_scope in {'validation', 'test'} and evaluation_lock_path:
            print(f'[EVAL-LOCK] 等待训练后评估锁: {evaluation_lock_path}')
        with interprocess_evaluation_lock(
            evaluation_lock_path if evaluation_scope in {'validation', 'test'} else None
        ) as evaluation_lock_wait:
            phase_timings['post_training_evaluation_lock_wait'] = evaluation_lock_wait
            if evaluation_scope in {'validation', 'test'} and evaluation_lock_path:
                print(f'[EVAL-LOCK] 已获取评估锁，等待 {evaluation_lock_wait:.1f}s')
            phase_started = time.perf_counter()
            results = (
                trainer.evaluate(split=evaluation_scope)
                if evaluation_scope in {'validation', 'test'} else {}
            )
            phase_timings['post_training_evaluation'] = time.perf_counter() - phase_started
        if evaluation_scope == 'none':
            print('按配置跳过训练后评估；本次运行仅可用于超参数校准')

        git_commit = None
        git_dirty = None
        synced_source_state = read_synced_source_state()
        source_hash = synced_source_state.get('source_hash')
        training_source_hash = synced_source_state.get('training_source_hash')
        try:
            git_commit = subprocess.run(
                ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True
            ).stdout.strip()
            git_dirty = bool(subprocess.run(
                ['git', 'status', '--porcelain'], capture_output=True, text=True, check=True
            ).stdout.strip())
        except (OSError, subprocess.SubprocessError):
            git_commit = synced_source_state.get('git_commit')
            git_dirty = synced_source_state.get('git_dirty')

        run_summary = {
            'status': 'completed',
            'started_at': datetime.fromtimestamp(run_started).isoformat(),
            'completed_at': datetime.now().isoformat(),
            'wall_time_seconds': time.time() - run_started,
            'result_dir': os.path.abspath(trainer.result_dir),
            'seed': config.get('seed'),
            'training_note': config.get('training_note', ''),
            'parameter_count': trainer.parameter_count,
            'data_protocol': trainer.data_protocol,
            'completed_epochs': len(trainer.train_losses),
            'training_objective_losses': [float(value) for value in trainer.train_losses],
            'validation_selection_losses': [float(value) for value in trainer.val_losses],
            'best_epoch': trainer.best_epoch,
            'best_val_loss': trainer.best_val_loss,
            'epoch_times_seconds': trainer.epoch_times,
            'mean_epoch_time_seconds': (
                float(np.mean(trainer.epoch_times)) if trainer.epoch_times else None
            ),
            'compile_requested': trainer.compile_requested,
            'compile_active': trainer.compile_active,
            'compile_fallback_used': trainer.compile_fallback_used,
            'peak_cuda_memory_bytes': (
                int(torch.cuda.max_memory_allocated(trainer.device))
                if trainer.device.type == 'cuda' else None
            ),
            'peak_host_memory_bytes': peak_host_memory_bytes(),
            'phase_timings_seconds': {
                name: float(value) for name, value in phase_timings.items()
            },
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'runtime_determinism': {
                'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
                'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
                'deterministic_algorithms_enabled': bool(
                    torch.are_deterministic_algorithms_enabled()
                ),
            },
            'gpu_name': (
                torch.cuda.get_device_name(trainer.device)
                if trainer.device.type == 'cuda' else None
            ),
            'git_commit': git_commit,
            'git_dirty': git_dirty,
            'source_hash': source_hash,
            'training_source_hash': training_source_hash,
            'training_config_fingerprint': training_config_fingerprint(config),
            'evaluation_scope': evaluation_scope,
            'evaluation_file': (
                'evaluation_results.json' if evaluation_scope == 'test'
                else 'validation_results.json' if evaluation_scope == 'validation'
                else None
            ),
            'primary_metrics': {
                'evaluation_loss': results.get('evaluation_loss'),
                'evaluation_data_loss': results.get('evaluation_data_loss'),
                'rmse_TEMP': results.get('physical_rmse_TEMP'),
                'rmse_SALT': results.get('physical_rmse_SALT'),
            },
        }
        with open(os.path.join(trainer.result_dir, 'run_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(run_summary, f, indent=2, ensure_ascii=False)
        with open(os.path.join(trainer.result_dir, '_SUCCESS'), 'w', encoding='utf-8') as f:
            f.write(run_summary['completed_at'] + '\n')

        print("训练和评估完成！")
        print(f"结果保存在: {trainer.result_dir}")

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        if trainer is not None and getattr(trainer, 'result_dir', None):
            failure = {
                'status': 'failed',
                'failed_at': datetime.now().isoformat(),
                'wall_time_seconds': time.time() - run_started,
                'peak_host_memory_bytes': peak_host_memory_bytes(),
                'phase_timings_seconds': {
                    name: float(value) for name, value in phase_timings.items()
                },
                'error_type': type(e).__name__,
                'error': str(e),
            }
            try:
                with open(os.path.join(trainer.result_dir, 'run_summary.json'), 'w', encoding='utf-8') as f:
                    json.dump(failure, f, indent=2, ensure_ascii=False)
            except OSError:
                pass
        raise
    finally:
        if trainer is not None:
            trainer.close()


if __name__ == "__main__":
    main()
