"""
海洋数据加载器
处理 NetCDF 海洋数据，用于 SEAF 与基线模型训练和预测
"""

import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import xarray as xr
import pickle
import hashlib
import json
import os
import random
from typing import Tuple, List, Dict, Optional, Sequence
from sklearn.preprocessing import StandardScaler


class OceanDataset(Dataset):
    """
    海洋数据集类
    用于加载和预处理NetCDF格式的海洋数据
    """

    def __init__(self, data_path: str, config: dict, mode: str = 'train',
                 train_ratio: Optional[float] = None, val_ratio: Optional[float] = None,
                 scalers: Optional[Dict] = None,
                 override_stride_lon: Optional[float] = None,
                 override_stride_lat: Optional[float] = None):
        """
        初始化数据集

        Args:
            data_path: NetCDF数据文件路径
            config: 配置字典
            mode: 模式 ('train', 'val', 'test')
            train_ratio: 训练集比例（减少到0.6）
            val_ratio: 验证集比例
            scalers: 预计算的标准化器（可选，用于验证集和测试集以防止数据泄露）
        """
        self.data_path = data_path
        self.config = config
        self.mode = mode
        self.train_ratio = config.get('train_ratio', 0.6) if train_ratio is None else train_ratio
        self.val_ratio = config.get('val_ratio', 0.2) if val_ratio is None else val_ratio
        self.test_ratio = 1.0 - self.train_ratio - self.val_ratio
        self.provided_scalers = scalers  # 存储外部提供的scalers

        self.input_variables = config['input_variables']
        self.target_variables = config['target_variables']
        self.sequence_length = config['sequence_length']
        self.prediction_length = config['prediction_length']
        self.enable_climatology_anomaly = config.get('enable_climatology_anomaly', False)
        self.enable_target_climatology_anomaly = config.get(
            'enable_target_climatology_anomaly',
            self.enable_climatology_anomaly,
        )
        self.climatology_period = max(1, int(config.get('climatology_period', 12)))
        self.anomaly_variables = set(config.get('anomaly_variables', self.target_variables))
        self.target_anomaly_variables = set(
            config.get('target_anomaly_variables', self.target_variables)
        )
        self.include_climatology_features = config.get('include_climatology_features', False)
        self.climatology_feature_variables = list(
            config.get('climatology_feature_variables', self.target_variables)
        )
        self.include_tendency_features = bool(
            config.get('include_tendency_features', False)
        )
        self.tendency_feature_variables = list(
            config.get('tendency_feature_variables', self.target_variables)
        )
        self.damped_persistence_coefficients = {}
        self.return_sample_index = bool(config.get('return_sample_index', False))
        # 数据范围设置（优先使用配置中的范围）
        self.lon_range = list(config.get('lon_range', [130.5, 162.5]))
        self.lat_range = list(config.get('lat_range', [6.5, 27.5]))
        self.depth_range = list(config.get('depth_range', [0.0, 5.0]))

        # 滑动窗口参数
        self.sliding_enabled = config.get(
            f'{mode}_sliding_enabled', config.get('sliding_enabled', False)
        )  # 是否启用滑动数据增强
        self.ocean_threshold = config.get('ocean_threshold', 1.0)   # 海洋面积占比阈值（1.0=仅纯海洋）
        self.lon_step = config.get('lon_step', 2.0)                # 经纬度滑动步长（兜底）
        self.sliding_regions = []        # 存储所有有效的100%海洋区域

        # 推理阶段可覆盖步长（用于 dense overlap-tile prediction）
        self.override_stride_lon = override_stride_lon
        self.override_stride_lat = override_stride_lat

        # 分模式滑动步长
        self._resolve_stride(config)

        # 在 _load_data 中从 NetCDF 动态解析，允许 ORAS5 等数据源增加物理输入。
        self.available_variables = []
        self.coord_variables = ['LONGITUDE', 'LATITUDE', 'LEVEL', 'TIME']
        self.preassembled_mmap_dir = config.get('preassembled_mmap_dir')
        self._preassembled_mmap_enabled = False

        if not self._try_load_preassembled_mmap():
            # 加载和预处理数据
            self._load_data()

            # 尝试从缓存加载预处理结果
            loaded_from_cache = self._try_load_from_cache()

            if loaded_from_cache:
                print(f"[{self.mode.upper()}] 从缓存加载预处理数据成功，跳过滑窗搜索/气候态/标准化")
                # 关闭 NetCDF，已不再需要
                if hasattr(self, 'dataset') and self.dataset is not None:
                    self.dataset.close()
                    self.dataset = None
            else:
                self._find_sliding_regions()  # 查找有效的滑动区域
                self._preprocess_data()
                # 预处理完成后关闭 NetCDF
                if hasattr(self, 'dataset') and self.dataset is not None:
                    self.dataset.close()
                    self.dataset = None
                self._save_cache()

        self._split_data()
        self._create_sequences()
        self.input_channel_slices = {}
        self.target_channel_slices = {}
        if getattr(self, '_preassembled_mmap_enabled', False):
            self._initialize_preassembled_channel_schema()
        else:
            self._initialize_channel_schema()
        self._validate_channel_schema()

    def can_share_preprocessed_with_mode(self, mode: str) -> bool:
        """Return whether ``mode`` has identical spatial preprocessing semantics.

        Temporal splits only change which sequence origins are exposed. When
        sliding enablement and strides are also identical, rebuilding the
        spatial arrays for validation/test is redundant and memory intensive.
        Regional-training experiments may intentionally use a different train
        grid, so those cases must still construct a separate payload.
        """
        target_sliding = self.config.get(
            f'{mode}_sliding_enabled', self.config.get('sliding_enabled', False)
        )
        target_stride_lon = self.config.get(f'{mode}_stride_lon')
        target_stride_lat = self.config.get(f'{mode}_stride_lat')
        if target_stride_lon is None or target_stride_lat is None:
            target_stride_lon = self.config.get('lon_step', 2.0)
            target_stride_lat = target_stride_lon
        return (
            bool(target_sliding) == bool(self.sliding_enabled)
            and float(target_stride_lon) == float(self.stride_lon)
            and float(target_stride_lat) == float(self.stride_lat)
        )

    def temporal_split_view(self, mode: str) -> 'OceanDataset':
        """Create a split-specific view over immutable preprocessed arrays.

        The returned dataset shares ``all_regions_data`` and fitted scalers,
        but owns its temporal indices, sequence list, mode, and sampler flag.
        This preserves independent-construction semantics without keeping three
        multi-gigabyte copies in memory.
        """
        if mode not in {'train', 'val', 'test'}:
            raise ValueError(f'未知数据集模式: {mode!r}')
        if not self.can_share_preprocessed_with_mode(mode):
            raise ValueError(
                f'{self.mode!r} 与 {mode!r} 的滑窗预处理语义不同，不能共享数组'
            )

        view = copy.copy(self)
        view.mode = mode
        view.sliding_enabled = self.config.get(
            f'{mode}_sliding_enabled', self.config.get('sliding_enabled', False)
        )
        view.override_stride_lon = None
        view.override_stride_lat = None
        view.return_sample_index = False
        view.provided_scalers = self.scalers
        view._resolve_stride(self.config)
        view._split_data()
        view._create_sequences()
        return view

    def _load_data(self):
        """
        加载NetCDF数据
        """
        print(f"正在加载数据文件: {self.data_path}")

        # 使用xarray加载NetCDF文件
        self.dataset = xr.open_dataset(self.data_path)

        missing_coords = [
            name for name in self.coord_variables if name not in self.dataset.coords
        ]
        if missing_coords:
            raise ValueError(
                f'数据文件缺少标准坐标 {missing_coords}；'
                '请先转换为 TIME/LEVEL/LATITUDE/LONGITUDE schema'
            )
        self.available_variables = list(self.dataset.data_vars)

        # 仅按深度切片，保留完整经纬度范围供2D滑动窗口使用
        self.dataset = self.dataset.sel(
            LEVEL=slice(self.depth_range[0], self.depth_range[1])
        )

        print(f"数据形状: {dict(self.dataset.dims)}")
        print(f"可用变量: {list(self.dataset.data_vars)}")

        # 获取坐标信息
        self.lons = self.dataset.LONGITUDE.values
        self.lats = self.dataset.LATITUDE.values
        self.levels = self.dataset.LEVEL.values
        self.times = self.dataset.TIME.values
        self.time_period_indices = self._extract_period_indices(self.times, self.climatology_period)

        lat_size = len(self.lats)
        lon_size = len(self.lons)

        print(f"时间序列长度: {len(self.times)}")
        print(f"空间维度: 经度{len(self.lons)} x 纬度{len(self.lats)} x 深度{len(self.levels)}")

    _PREASSEMBLED_MMAP_FORMAT_VERSION = 1

    def _try_load_preassembled_mmap(self) -> bool:
        """Load a read-only, channel-preassembled time-axis cache when requested."""
        if not self.preassembled_mmap_dir:
            return False

        cache_dir = os.path.realpath(str(self.preassembled_mmap_dir))
        manifest_path = os.path.join(cache_dir, 'manifest.json')
        success_path = os.path.join(cache_dir, '_SUCCESS')
        state_path = os.path.join(cache_dir, 'state.pkl')
        if not all(os.path.isfile(path) for path in (manifest_path, success_path, state_path)):
            raise FileNotFoundError(
                f'预组装 mmap 缓存不完整: {cache_dir}'
            )

        with open(manifest_path, 'r', encoding='utf-8') as handle:
            manifest = json.load(handle)
        if int(manifest.get('format_version', -1)) != self._PREASSEMBLED_MMAP_FORMAT_VERSION:
            raise ValueError(
                '预组装 mmap 格式版本不匹配: '
                f"{manifest.get('format_version')} != {self._PREASSEMBLED_MMAP_FORMAT_VERSION}"
            )

        data_stat = os.stat(self.data_path)
        expected_identity = manifest.get('data_identity', {})
        actual_identity = {
            'size': int(data_stat.st_size),
            'mtime_ns': int(data_stat.st_mtime_ns),
        }
        if expected_identity != actual_identity:
            raise ValueError(
                f'预组装 mmap 与当前数据文件不一致: '
                f'expected={expected_identity}, actual={actual_identity}'
            )
        if int(manifest.get('sequence_length', -1)) != int(self.sequence_length):
            raise ValueError('预组装 mmap 的 sequence_length 与当前配置不一致')
        if int(manifest.get('prediction_length', -1)) != int(self.prediction_length):
            raise ValueError('预组装 mmap 的 prediction_length 与当前配置不一致')

        input_path = os.path.join(cache_dir, manifest['input_file'])
        fullfield_target_path = os.path.join(cache_dir, manifest['fullfield_target_file'])
        climatology_path = os.path.join(cache_dir, manifest['target_climatology_file'])
        for path in (input_path, fullfield_target_path, climatology_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f'预组装 mmap 缺少数组: {path}')

        # Copy-on-write mappings are writable from PyTorch's perspective while
        # never persisting accidental tensor mutations back to the shared cache.
        self._mmap_inputs = np.load(input_path, mmap_mode='c', allow_pickle=False)
        self._mmap_fullfield_targets = np.load(
            fullfield_target_path, mmap_mode='c', allow_pickle=False
        )
        self._mmap_target_climatology = np.load(
            climatology_path, mmap_mode='c', allow_pickle=False
        )
        expected_shapes = {
            'inputs': tuple(manifest['input_shape']),
            'fullfield_targets': tuple(manifest['fullfield_target_shape']),
            'target_climatology': tuple(manifest['target_climatology_shape']),
        }
        actual_shapes = {
            'inputs': tuple(self._mmap_inputs.shape),
            'fullfield_targets': tuple(self._mmap_fullfield_targets.shape),
            'target_climatology': tuple(self._mmap_target_climatology.shape),
        }
        if actual_shapes != expected_shapes:
            raise ValueError(
                f'预组装 mmap shape 不匹配: expected={expected_shapes}, actual={actual_shapes}'
            )

        with open(state_path, 'rb') as handle:
            state = pickle.load(handle)
        self.scalers = state['scalers']
        self.damped_persistence_coefficients = state['damped_persistence_coefficients']

        self.available_variables = list(manifest['available_variables'])
        self.lons = np.asarray(manifest['lons'], dtype=np.float32)
        self.lats = np.asarray(manifest['lats'], dtype=np.float32)
        self.levels = np.asarray(manifest['levels'], dtype=np.float32)
        self.times = np.asarray(manifest['times'])
        self.time_period_indices = np.asarray(
            manifest['time_period_indices'], dtype=np.int64
        )
        self.dataset = None

        target_slices = {
            name: slice(int(bounds[0]), int(bounds[1]))
            for name, bounds in manifest['target_channel_slices'].items()
        }
        self._preassembled_target_source_slices = target_slices
        self._preassembled_fullfield_scaler_names = dict(
            manifest['fullfield_target_scalers']
        )
        input_slices = {
            name: slice(int(bounds[0]), int(bounds[1]))
            for name, bounds in manifest['input_channel_slices'].items()
        }
        self._preassembled_manifest_input_slices = input_slices
        actual_input_variables = list(self.input_variables)
        if self.include_climatology_features:
            actual_input_variables.extend(
                self._climatology_feature_name(name)
                for name in self.climatology_feature_variables
            )
        if self.include_tendency_features:
            actual_input_variables.extend(
                self._tendency_feature_name(name)
                for name in self.tendency_feature_variables
            )
        if self.config.get('enable_positional_encoding', False):
            actual_input_variables.append('SPATIAL_ENCODING')
        if self.config.get('enable_time_encoding', False):
            actual_input_variables.append('TIME_ENCODING')

        missing_inputs = [name for name in actual_input_variables if name not in input_slices]
        missing_targets = [name for name in self.target_variables if name not in target_slices]
        if missing_inputs or missing_targets:
            raise ValueError(
                '预组装 mmap 不覆盖当前配置通道: '
                f'inputs={missing_inputs}, targets={missing_targets}'
            )
        selected_channels = []
        for name in actual_input_variables:
            source_slice = input_slices[name]
            selected_channels.extend(range(source_slice.start, source_slice.stop))
        self.actual_input_variables = actual_input_variables
        if selected_channels == list(range(selected_channels[0], selected_channels[-1] + 1)):
            self._preassembled_input_selection = slice(
                selected_channels[0], selected_channels[-1] + 1
            )
        else:
            self._preassembled_input_selection = np.asarray(
                selected_channels, dtype=np.int64
            )
        selected_targets = []
        for name in self.target_variables:
            source_slice = target_slices[name]
            selected_targets.extend(range(source_slice.start, source_slice.stop))
        if selected_targets == list(range(selected_targets[0], selected_targets[-1] + 1)):
            self._preassembled_target_selection = slice(
                selected_targets[0], selected_targets[-1] + 1
            )
        else:
            self._preassembled_target_selection = np.asarray(
                selected_targets, dtype=np.int64
            )

        self.all_regions_data = []
        for region_idx, region_meta in enumerate(manifest['regions']):
            climatology = {}
            for variable, source_slice in target_slices.items():
                climatology[variable] = self._mmap_target_climatology[
                    region_idx, :, source_slice, :, :
                ]
            self.all_regions_data.append({
                'lon_range': region_meta['lon_range'],
                'lat_range': region_meta['lat_range'],
                'region_type': region_meta.get('region_type', 'sliding'),
                'coords': {
                    'lons': np.asarray(region_meta['lons'], dtype=np.float32),
                    'lats': np.asarray(region_meta['lats'], dtype=np.float32),
                    'levels': self.levels,
                    'times': self.times,
                },
                'climatology': climatology,
            })

        self._preassembled_mmap_enabled = True
        print(
            f'[{self.mode.upper()}] 使用只读预组装 mmap: {cache_dir}; '
            f'regions={len(self.all_regions_data)}, channels={len(selected_channels)}'
        )
        return True

    def _initialize_preassembled_channel_schema(self):
        """Initialize output channel slices without materializing mmap arrays."""
        offset = 0
        manifest_slices = self._preassembled_manifest_input_slices
        for variable in self.actual_input_variables:
            source_slice = manifest_slices[variable]
            length = int(source_slice.stop - source_slice.start)
            self.input_channel_slices[variable] = slice(offset, offset + length)
            offset += length

        offset = 0
        for variable in self.target_variables:
            source_slice = self._preassembled_target_source_slices[variable]
            length = int(source_slice.stop - source_slice.start)
            self.target_channel_slices[variable] = slice(offset, offset + length)
            offset += length

    # ========== 预处理缓存持久化 ==========

    # v5 introduced terminal anchors; v6 shares mode-independent payloads;
    # v7 adds train-scaled causal physical tendency features; v8 separates
    # input and target anomaly spaces for the strict target ablation.
    _CACHE_FORMAT_VERSION = 8
    _WINDOW_GRID_POLICY = 'regular_plus_terminal_anchor_v1'
    _CACHE_CONFIG_KEYS = frozenset({
        'lon_range', 'lat_range', 'depth_range',
        'ocean_threshold', 'ocean_coverage_depth', 'ocean_mask_variable',
        'input_variables',
        'anomaly_variables', 'target_anomaly_variables', 'climatology_period',
        'include_climatology_features', 'climatology_feature_variables',
        'tendency_feature_variables',
        'climatology_baseline_variables',
        'enable_climatology_anomaly', 'enable_target_climatology_anomaly',
    })

    def _compute_cache_key(self) -> str:
        """从数据文件身份和所有预处理语义生成缓存 key。"""
        data_stat = os.stat(self.data_path)
        relevant = {}
        for k in self._CACHE_CONFIG_KEYS:
            relevant[k] = self.config.get(k)
        relevant['cache_format_version'] = self._CACHE_FORMAT_VERSION
        relevant['window_grid_policy'] = self._WINDOW_GRID_POLICY
        relevant['data_identity'] = {
            'path': os.path.realpath(self.data_path),
            'size': int(data_stat.st_size),
            'mtime_ns': int(data_stat.st_mtime_ns),
        }
        # target_variables 本身不必拆分缓存，但缓存必须覆盖所有可能被读取的
        # 物理变量。以变量并集作为身份，既允许 TEMP/SALT 单任务共享联合缓存，
        # 又不会让一个缺少目标变量的缓存被错误复用。
        required_variables = set(self.config.get('input_variables', []))
        required_variables.update(self.config.get('target_variables', []))
        required_variables.update(self.config.get('anomaly_variables', []))
        required_variables.update(self.config.get('target_anomaly_variables', []))
        required_variables.update(self.config.get('climatology_feature_variables', []))
        required_variables.update(self.config.get('climatology_baseline_variables', []))
        required_variables.update(self.config.get('tendency_feature_variables', []))
        relevant['required_physical_variables'] = sorted(required_variables)
        relevant['sliding_enabled'] = self.sliding_enabled
        relevant['stride_lon'] = self.stride_lon
        relevant['stride_lat'] = self.stride_lat
        relevant['train_ratio'] = self.train_ratio
        relevant['val_ratio'] = self.val_ratio
        scaler_fingerprint = self._scalers_fingerprint(self.provided_scalers)
        if scaler_fingerprint is not None:
            relevant['provided_scalers'] = scaler_fingerprint
        raw = json.dumps(relevant, sort_keys=True, default=str)
        h = hashlib.sha256(raw.encode('utf-8'))
        return h.hexdigest()[:16]

    @staticmethod
    def _scalers_fingerprint(scalers) -> Optional[str]:
        """为外部训练 scaler 生成稳定指纹，避免误用其他 scaler 的缓存。"""
        if not scalers:
            return None
        summary = {}
        for name in sorted(scalers):
            scaler = scalers[name]
            summary[name] = {
                'mean': np.asarray(getattr(scaler, 'mean_', [])).tolist(),
                'scale': np.asarray(getattr(scaler, 'scale_', [])).tolist(),
                'var': np.asarray(getattr(scaler, 'var_', [])).tolist(),
            }
        raw = json.dumps(summary, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cache_dir(self) -> str:
        """缓存目录由完整预处理语义决定，与纯时间 split 名称无关。"""
        base = str(self.config.get('cache_preprocessed_dir', '.cache/preprocessed'))
        key = self._compute_cache_key()
        return os.path.join(base, key, 'payload')

    def _save_cache(self):
        """将预处理结果（scalers、normalized_data、climatology 等）保存到磁盘。"""
        cache_enabled = self.config.get('cache_preprocessed', True)
        if not cache_enabled or not self.all_regions_data:
            return
        cache_dir = self._cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        success_path = os.path.join(cache_dir, '_SUCCESS')
        if os.path.exists(success_path):
            os.remove(success_path)

        print(f"[CACHE] 保存预处理缓存到 {cache_dir} ...")

        # 1. scalers（pickle 序列化）
        with open(os.path.join(cache_dir, 'scalers.pkl'), 'wb') as f:
            pickle.dump(self.scalers, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 2. time_period_indices
        np.save(os.path.join(cache_dir, 'time_period_indices.npy'), self.time_period_indices)

        # 3. 元信息（滑动区域、变量列表等）
        metadata = {
            'cache_format_version': self._CACHE_FORMAT_VERSION,
            'window_grid_policy': self._WINDOW_GRID_POLICY,
            'mode_independent_payload': True,
            'sliding_regions': [
                {
                    'lon_range': r['lon_range'],
                    'lat_range': r['lat_range'],
                    'region_type': r.get('region_type', 'sliding'),
                }
                for r in self.all_regions_data
            ],
            'available_variables': self.available_variables,
            'lons': self.lons.tolist(),
            'lats': self.lats.tolist(),
            'levels': self.levels.tolist(),
            'times': [str(t) for t in self.times],
        }
        with open(os.path.join(cache_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # 4. 逐个区域保存数组
        for idx, region in enumerate(self.all_regions_data):
            # normalized_data
            norm_dict = {}
            for var_name, arr in region.get('normalized_data', {}).items():
                if (
                    var_name in {'SPATIAL_ENCODING', 'TIME_ENCODING'}
                    or var_name.startswith('TENDENCY_')
                ):
                    continue
                norm_dict[f'norm_{var_name}'] = arr
            for var_name, arr in region.get('normalized_target_data', {}).items():
                if self._target_scaler_name(var_name) != var_name:
                    norm_dict[f'targetnorm_{var_name}'] = arr
            # coords
            for coord_name in ('lons', 'lats', 'levels', 'times'):
                val = region.get('coords', {}).get(coord_name)
                if val is not None:
                    norm_dict[f'coord_{coord_name}'] = np.asarray(val)
            if norm_dict:
                np.savez_compressed(
                    os.path.join(cache_dir, f'region_{idx}.npz'),
                    **norm_dict,
                )

            # climatology（若启用 anomaly）
            clim_dict = {}
            for var_name, arr in region.get('climatology', {}).items():
                clim_dict[f'clim_{var_name}'] = arr
            if clim_dict:
                np.savez_compressed(
                    os.path.join(cache_dir, f'region_{idx}_clim.npz'),
                    **clim_dict,
                )

            # anomaly_data 只用于拟合/生成 normalized_data。缓存加载时可由
            # normalized_data + scaler + climatology 恢复目标物理量，因此不再
            # 重复落盘，避免每种数据消融额外保存一整份 TEMP/SALT anomaly。

        # 完成标记最后写入。进程若在此前中断，加载器会拒绝整个半成品缓存，
        # 而不是把缺文件的区域静默当作有效数据。
        marker_tmp = success_path + '.tmp'
        with open(marker_tmp, 'w', encoding='utf-8') as f:
            f.write(f'cache_format_version={self._CACHE_FORMAT_VERSION}\n')
        os.replace(marker_tmp, success_path)

        print(f"[CACHE] 缓存保存完成，共 {len(self.all_regions_data)} 个区域")

    def _try_load_from_cache(self) -> bool:
        """尝试从缓存加载预处理结果，成功返回 True。"""
        cache_enabled = self.config.get('cache_preprocessed', True)
        if not cache_enabled:
            return False
        cache_dir = self._cache_dir()
        meta_path = os.path.join(cache_dir, 'metadata.json')
        scalers_path = os.path.join(cache_dir, 'scalers.pkl')
        success_path = os.path.join(cache_dir, '_SUCCESS')
        if (
            not os.path.isfile(success_path)
            or not os.path.isfile(meta_path)
            or not os.path.isfile(scalers_path)
        ):
            return False

        try:
            # 1. scalers
            with open(scalers_path, 'rb') as f:
                self.scalers = pickle.load(f)

            # 2. time_period_indices
            tpi_path = os.path.join(cache_dir, 'time_period_indices.npy')
            if os.path.isfile(tpi_path):
                self.time_period_indices = np.load(tpi_path)
            else:
                # 降级：用已有 times 重新计算
                self.time_period_indices = self._extract_period_indices(
                    self.times, self.climatology_period
                )

            # 3. 元信息
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            # 4. 重建 all_regions_data
            self.all_regions_data = []
            sliding_list = metadata.get('sliding_regions', [])
            if not sliding_list:
                raise ValueError('缓存元信息不含任何空间区域')
            for idx, region_meta in enumerate(sliding_list):
                region = {
                    'lon_range': region_meta['lon_range'],
                    'lat_range': region_meta['lat_range'],
                    'region_type': region_meta.get('region_type', 'sliding'),
                }

                # 加载 normalized_data
                norm_path = os.path.join(cache_dir, f'region_{idx}.npz')
                if not os.path.isfile(norm_path):
                    raise FileNotFoundError(f'缓存缺少区域文件: {norm_path}')
                with np.load(norm_path, allow_pickle=False) as loaded:
                    norm_data = {}
                    target_norm_data = {}
                    coords = {}
                    for key in loaded.files:
                        if key.startswith('targetnorm_'):
                            var_name = key[11:]
                            target_norm_data[var_name] = loaded[key]
                        elif key.startswith('norm_'):
                            var_name = key[5:]
                            norm_data[var_name] = loaded[key]
                        elif key.startswith('coord_'):
                            coord_name = key[6:]
                            arr = loaded[key]
                            if arr.dtype.kind == 'S' or arr.dtype.kind == 'U':
                                arr = arr.astype(str)
                            coords[coord_name] = arr
                if not norm_data:
                    raise ValueError(f'缓存区域 {idx} 不含 normalized_data')
                missing_targets = [
                    name for name in self.target_variables if name not in norm_data
                ]
                if missing_targets:
                    raise ValueError(
                        f'缓存区域 {idx} 缺少目标变量: {missing_targets}'
                    )
                missing_separate_targets = [
                    name for name in self.target_variables
                    if self._target_scaler_name(name) != name
                    and name not in target_norm_data
                ]
                if missing_separate_targets:
                    raise ValueError(
                        f'缓存区域 {idx} 缺少独立目标空间变量: '
                        f'{missing_separate_targets}'
                    )
                region['normalized_data'] = norm_data
                region['normalized_target_data'] = target_norm_data
                region['coords'] = coords

                # 加载 climatology
                clim_path = os.path.join(cache_dir, f'region_{idx}_clim.npz')
                if os.path.isfile(clim_path):
                    with np.load(clim_path, allow_pickle=False) as clim_loaded:
                        clim_data = {}
                        for key in clim_loaded.files:
                            if key.startswith('clim_'):
                                clim_data[key[5:]] = clim_loaded[key]
                    region['climatology'] = clim_data

                # 兼容旧缓存中的 anomaly_data；v4 自身不再重复保存。
                anom_path = os.path.join(cache_dir, f'region_{idx}_anom.npz')
                if os.path.isfile(anom_path):
                    with np.load(anom_path, allow_pickle=False) as anom_loaded:
                        anom_data = {}
                        for key in anom_loaded.files:
                            if key.startswith('anom_'):
                                anom_data[key[5:]] = anom_loaded[key]
                    region['anomaly_data'] = anom_data

                # 辅助字段，后续惰性填充
                region['data'] = {}
                if 'climatology' not in region:
                    region['climatology'] = {}
                if 'anomaly_data' not in region:
                    region['anomaly_data'] = {}

                self.all_regions_data.append(region)

            # Reconstruct raw data from normalized data for baseline computation
            needed_vars = set(self.target_variables) | set(self.tendency_feature_variables)
            for region in self.all_regions_data:
                if not region.get('data'):
                    region['data'] = {}
                for var_name in needed_vars:
                    if var_name in region['data']:
                        continue
                    norm_arr = region.get('normalized_data', {}).get(var_name)
                    if norm_arr is None:
                        continue
                    scaler = self.scalers.get(var_name)
                    if scaler is not None:
                        orig_shape = norm_arr.shape
                        recon = scaler.inverse_transform(
                            norm_arr.reshape(-1, 1)
                        ).reshape(orig_shape).astype(np.float32)
                    else:
                        recon = norm_arr.astype(np.float32)
                    if self._is_anomaly_variable(var_name):
                        clim = region.get('climatology', {}).get(var_name)
                        if clim is not None:
                            nsteps = recon.shape[0]
                            recon = recon + clim[self.time_period_indices[:nsteps]]
                    region['data'][var_name] = recon

            self._estimate_damped_persistence_coefficients()

            # Auxiliary encodings are deterministic functions of coordinates
            # and configuration. Rebuild them after cache load so positional
            # and temporal ablations share one physical-data cache.
            for region in self.all_regions_data:
                normalized = region.get('normalized_data', {})
                for feature_name in tuple(normalized):
                    if (
                        feature_name in {'SPATIAL_ENCODING', 'TIME_ENCODING'}
                        or feature_name.startswith('TENDENCY_')
                    ):
                        normalized.pop(feature_name, None)
                self._add_auxiliary_encodings(region)

            if not self.all_regions_data:
                return False

            print(f"[CACHE] 从 {cache_dir} 加载了 {len(self.all_regions_data)} 个区域")
            return True

        except Exception as e:
            print(f"[CACHE] 缓存加载失败 ({e})，将重新预处理")
            return False

    def _resolve_stride(self, config: dict):
        """根据模式选择对应的经纬度步长，支持推理阶段覆盖"""
        if self.override_stride_lon is not None and self.override_stride_lat is not None:
            self.stride_lon = self.override_stride_lon
            self.stride_lat = self.override_stride_lat
        else:
            mode_key = self.mode
            stride_lon = config.get(f'{mode_key}_stride_lon')
            stride_lat = config.get(f'{mode_key}_stride_lat')
            if stride_lon is None or stride_lat is None:
                stride_lon = config.get('lon_step', 2.0)
                stride_lat = stride_lon
            self.stride_lon = stride_lon
            self.stride_lat = stride_lat
        print(f"[{self.mode.upper()}] 滑动步长: 经度={self.stride_lon}°, 纬度={self.stride_lat}°")

    def _find_sliding_regions(self):
        """
        2D滑动窗口搜索：在完整海洋数据范围内双向（经度+纬度）滑动，
        仅保留100%海洋覆盖的区域。
        """
        if not self.sliding_enabled:
            print("滑动窗口关闭，使用配置中的原始经纬度区域。")
            self.sliding_regions.append({
                'lon_range': self.lon_range,
                'lat_range': self.lat_range,
                'region_type': 'original'
            })
            return

        print("正在进行2D滑动窗口搜索（仅保留100%海洋覆盖区域）...")

        all_lons = self.dataset.LONGITUDE.values
        all_lats = self.dataset.LATITUDE.values

        # 窗口大小由配置的 lon_range/lat_range 跨度决定
        window_lon_size = self.lon_range[1] - self.lon_range[0]
        window_lat_size = self.lat_range[1] - self.lat_range[0]

        min_lon = float(all_lons.min())
        max_lon = float(all_lons.max())
        min_lat = float(all_lats.min())
        max_lat = float(all_lats.max())

        print(f"  数据范围: 经度 [{min_lon:.1f}, {max_lon:.1f}], 纬度 [{min_lat:.1f}, {max_lat:.1f}]")
        print(f"  窗口大小: {window_lon_size:.1f}° × {window_lat_size:.1f}°")
        print(f"  滑动步长: 经度={self.stride_lon}°, 纬度={self.stride_lat}°")
        print(f"  海洋阈值: {self.ocean_threshold} (100% 纯海洋)")

        # 2D 滑动: 规则步长之外显式加入终端锚点。否则当跨度不能被
        # stride 整除时，最东/最北边缘会留下从未被任何窗口覆盖的空带。
        lon_starts = self._window_starts(
            min_lon, max_lon, window_lon_size, self.stride_lon
        )
        lat_starts = self._window_starts(
            min_lat, max_lat, window_lat_size, self.stride_lat
        )
        for current_lon in lon_starts:
            for current_lat in lat_starts:
                new_lon_range = [current_lon, current_lon + window_lon_size]
                new_lat_range = [current_lat, current_lat + window_lat_size]

                if self._check_ocean_coverage(self.dataset, new_lon_range, new_lat_range):
                    if not self._region_exists(new_lon_range, new_lat_range):
                        self.sliding_regions.append({
                            'lon_range': new_lon_range,
                            'lat_range': new_lat_range,
                            'region_type': 'sliding'
                        })

        print(f"找到 {len(self.sliding_regions)} 个100%海洋覆盖的区域")
        for i, region in enumerate(self.sliding_regions):
            print(
                f"  区域 {i+1}: 经度 [{region['lon_range'][0]:.1f}, {region['lon_range'][1]:.1f}], "
                f"纬度 [{region['lat_range'][0]:.1f}, {region['lat_range'][1]:.1f}]"
            )

    @staticmethod
    def _window_starts(
        axis_min: float,
        axis_max: float,
        window_size: float,
        stride: float,
    ) -> List[float]:
        """Generate regular starts and always include the terminal anchor."""
        axis_min = float(axis_min)
        axis_max = float(axis_max)
        window_size = float(window_size)
        stride = float(stride)
        terminal = axis_max - window_size
        if stride <= 0 or window_size <= 0 or terminal < axis_min:
            return []
        count = int(np.floor((terminal - axis_min) / stride + 1e-10)) + 1
        starts = [axis_min + idx * stride for idx in range(count)]
        if not starts or not np.isclose(starts[-1], terminal):
            starts.append(terminal)
        return [float(value) for value in starts]

    def _region_exists(self, lon_range: List[float], lat_range: List[float]) -> bool:
        """检查指定经纬度范围是否已经存在于滑动区域列表中"""
        for region in self.sliding_regions:
            if (
                np.isclose(region['lon_range'][0], lon_range[0]) and
                np.isclose(region['lon_range'][1], lon_range[1]) and
                np.isclose(region['lat_range'][0], lat_range[0]) and
                np.isclose(region['lat_range'][1], lat_range[1])
            ):
                return True
        return False

    def _check_ocean_coverage(self, dataset, lon_range, lat_range):
        """
        检查指定区域的海洋覆盖率
        """
        try:
            # 选择该区域的数据
            region_data = dataset.sel(
                LONGITUDE=slice(lon_range[0], lon_range[1]),
                LATITUDE=slice(lat_range[0], lat_range[1]),
            )

            mask_variable = self.config.get(
                'ocean_mask_variable',
                self.target_variables[0] if self.target_variables else 'TEMP',
            )
            if mask_variable in region_data.data_vars:
                reference = region_data[mask_variable]
                if 'TIME' in reference.dims:
                    reference = reference.isel(TIME=0)
                if 'LEVEL' in reference.dims:
                    coverage_depth = self.config.get('ocean_coverage_depth')
                    if coverage_depth is None:
                        reference = reference.isel(LEVEL=0)
                    else:
                        reference = reference.sel(
                            LEVEL=float(coverage_depth), method='nearest'
                        )
                reference_data = np.asarray(reference.values)

                # 有限值表示该参考深度上确有海水；不能忽略 NaN 后再求平均。
                total_points = reference_data.size
                valid_points = np.sum(np.isfinite(reference_data))
                ocean_ratio = valid_points / total_points if total_points > 0 else 0

                return ocean_ratio >= self.ocean_threshold

        except Exception as e:
            print(f"检查区域 {lon_range} 时出错: {e}")
            return False

        return False

    def _preprocess_data(self):
        """
        预处理数据：对所有100%海洋窗口统一处理
        """
        print("正在预处理数据...")

        self.all_regions_data = []

        if len(self.sliding_regions) > 0:
            print(f"处理 {len(self.sliding_regions)} 个100%海洋区域...")

            for i, region in enumerate(self.sliding_regions):
                print(f"  处理区域 {i+1}/{len(self.sliding_regions)}: "
                      f"经度[{region['lon_range'][0]:.1f},{region['lon_range'][1]:.1f}] "
                      f"纬度[{region['lat_range'][0]:.1f},{region['lat_range'][1]:.1f}]")

                region_dataset = self.dataset.sel(
                    LONGITUDE=slice(region['lon_range'][0], region['lon_range'][1]),
                    LATITUDE=slice(region['lat_range'][0], region['lat_range'][1]),
                )

                region_data = self._process_single_region(region_dataset)
                if region_data:
                    self.all_regions_data.append({
                        'data': region_data['data'],
                        'coords': region_data['coords'],
                        'region_type': region.get('region_type', 'sliding'),
                        'lon_range': region['lon_range'],
                        'lat_range': region['lat_range']
                    })
        else:
            print("警告: 没有找到任何100%海洋覆盖的区域！请检查 ocean_threshold 或窗口大小设置。")

        print(f"总共处理了 {len(self.all_regions_data)} 个区域的数据")

        if len(self.all_regions_data) == 0:
            raise ValueError("没有有效的数据区域，无法创建数据集。请降低 ocean_threshold 或调整窗口大小。")

        self._merge_and_normalize_data()

    def _process_single_region(self, dataset):
        """
        处理单个区域的数据
        """
        region_data = {}

        # 过滤输入变量，只使用实际存在的变量
        valid_input_vars = []
        for var in self.input_variables:
            if var in self.available_variables and var in dataset.data_vars:
                valid_input_vars.append(var)

        # 处理所有有效变量
        for var in valid_input_vars + self.target_variables:
            if var in dataset.data_vars:
                # 获取变量数据
                data = dataset[var].values

                # 处理缺失值
                if not np.isfinite(data).all():
                    data = self._fill_missing_values(data, variable=var)

                region_data[var] = data

        if not region_data:
            return None

        coords = {
            'lons': dataset.LONGITUDE.values if 'LONGITUDE' in dataset.coords else self.lons,
            'lats': dataset.LATITUDE.values if 'LATITUDE' in dataset.coords else self.lats,
            'levels': dataset.LEVEL.values if 'LEVEL' in dataset.coords else getattr(self, 'levels', np.array([0.0])),
            'times': dataset.TIME.values if 'TIME' in dataset.coords else self.times
        }

        return {
            'data': region_data,
            'coords': coords
        }

    def _is_anomaly_variable(self, variable: str) -> bool:
        return self.enable_climatology_anomaly and variable in self.anomaly_variables

    def _is_target_anomaly_variable(self, variable: str) -> bool:
        enabled = getattr(
            self,
            'enable_target_climatology_anomaly',
            getattr(self, 'enable_climatology_anomaly', False),
        )
        variables = getattr(
            self,
            'target_anomaly_variables',
            set(getattr(self, 'target_variables', [])),
        )
        return (
            enabled
            and variable in variables
        )

    def _target_scaler_name(self, variable: str) -> str:
        if self._is_target_anomaly_variable(variable) == self._is_anomaly_variable(variable):
            return variable
        return f'TARGET_{variable}'

    @staticmethod
    def _climatology_feature_name(variable: str) -> str:
        return f"CLIMATOLOGY_{variable}"

    @staticmethod
    def _tendency_feature_name(variable: str) -> str:
        return f"TENDENCY_{variable}"

    @staticmethod
    def _build_tendency_feature(source: np.ndarray) -> np.ndarray:
        """Return a causal one-step backward difference with a zero first step."""
        values = np.asarray(source, dtype=np.float32)
        if values.ndim not in (3, 4):
            raise ValueError(f"tendency 源数据维度必须为 3 或 4，实际为 {values.shape}")
        tendency = np.empty_like(values, dtype=np.float32)
        tendency[0] = 0.0
        tendency[1:] = values[1:] - values[:-1]
        return tendency

    def _compute_region_climatology(self, region: Dict):
        """基于训练期计算月气候态；模型开关只决定是否使用 anomaly。"""
        region['climatology'] = {}
        region['anomaly_data'] = {}

        train_end_idx = int(len(self.times) * self.train_ratio)
        train_periods = self.time_period_indices[:train_end_idx]
        # 所有目标都需要月气候态来构造 climatology/anomaly-persistence 基线。
        # 这必须独立于模型是否在 anomaly 空间训练，否则不同消融的基线口径会变化。
        variables = set(
            getattr(self, 'config', {}).get(
                'climatology_baseline_variables', self.target_variables
            )
        )
        if self.enable_climatology_anomaly:
            variables.update(self.anomaly_variables)
        if getattr(
            self,
            'enable_target_climatology_anomaly',
            self.enable_climatology_anomaly,
        ):
            variables.update(getattr(
                self, 'target_anomaly_variables', set(self.target_variables)
            ))
        if self.include_climatology_features:
            variables.update(self.climatology_feature_variables)

        for var in variables:
            if var not in region['data']:
                continue

            data = region['data'][var].astype(np.float32, copy=False)
            train_data = data[:train_end_idx]
            if train_data.size == 0:
                continue

            fallback = np.nanmean(train_data, axis=0).astype(np.float32)
            fallback = np.nan_to_num(fallback, nan=0.0)
            climatology = np.empty((self.climatology_period,) + data.shape[1:], dtype=np.float32)

            for period_idx in range(self.climatology_period):
                mask = train_periods == period_idx
                if np.any(mask):
                    period_mean = np.nanmean(train_data[mask], axis=0).astype(np.float32)
                    climatology[period_idx] = np.where(np.isnan(period_mean), fallback, period_mean)
                else:
                    climatology[period_idx] = fallback

            region['climatology'][var] = climatology

            if self._is_anomaly_variable(var) or self._is_target_anomaly_variable(var):
                periods = self.time_period_indices[:data.shape[0]]
                region['anomaly_data'][var] = (data - climatology[periods]).astype(np.float32)

    def _estimate_damped_persistence_coefficients(self) -> None:
        """Estimate train-only lag regression coefficients for damped AP.

        Coefficients are pooled over the same canonical windows used by the
        evaluation metric and retained per target, lead, and depth channel.
        Clipping to [0, 1] makes this a genuine damping baseline rather than an
        unconstrained linear autoregression.
        """
        train_end_idx = int(len(self.times) * self.train_ratio)
        coefficients = {}
        for variable in self.target_variables:
            numerator = None
            denominator = None
            for region in self.all_regions_data:
                raw_data = region.get('data', {}).get(variable)
                climatology = region.get('climatology', {}).get(variable)
                if raw_data is None or climatology is None:
                    continue
                periods = self.time_period_indices[:train_end_idx]
                anomaly = (
                    raw_data[:train_end_idx] - climatology[periods]
                ).astype(np.float32, copy=False)
                anomaly = self._series_to_channel_first(anomaly)
                if numerator is None:
                    shape = (self.prediction_length, anomaly.shape[1])
                    numerator = np.zeros(shape, dtype=np.float64)
                    denominator = np.zeros(shape, dtype=np.float64)
                for lead in range(1, self.prediction_length + 1):
                    if anomaly.shape[0] <= lead:
                        continue
                    source = anomaly[:-lead]
                    future = anomaly[lead:]
                    numerator[lead - 1] += np.sum(
                        source * future,
                        axis=(0, 2, 3),
                        dtype=np.float64,
                    )
                    denominator[lead - 1] += np.sum(
                        source * source,
                        axis=(0, 2, 3),
                        dtype=np.float64,
                    )
            if numerator is None or denominator is None:
                raise ValueError(
                    f'无法从训练期数据估计 {variable} damped persistence 系数'
                )
            rho = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > np.finfo(np.float64).eps,
            )
            coefficients[variable] = np.clip(rho, 0.0, 1.0).astype(np.float32)
        self.damped_persistence_coefficients = coefficients

    def _build_climatology_series(self, region: Dict, variable: str) -> Optional[np.ndarray]:
        climatology = region.get('climatology', {}).get(variable)
        if climatology is None:
            return None
        time_steps = next(iter(region['data'].values())).shape[0]
        periods = self.time_period_indices[:time_steps]
        return climatology[periods]

    def _get_model_source_data(self, region: Dict, variable: str) -> Optional[np.ndarray]:
        if self._is_anomaly_variable(variable):
            anomaly_data = region.get('anomaly_data', {}).get(variable)
            if anomaly_data is not None:
                return anomaly_data
        return region['data'].get(variable)

    def _get_target_source_data(self, region: Dict, variable: str) -> Optional[np.ndarray]:
        if self._is_target_anomaly_variable(variable):
            anomaly_data = region.get('anomaly_data', {}).get(variable)
            if anomaly_data is not None:
                return anomaly_data
        return region['data'].get(variable)

    def _merge_and_normalize_data(self):
        """
        合并所有区域数据并进行标准化
        """
        print("合并和标准化所有区域数据...")

        # 获取所有变量名
        all_vars = set()
        for region in self.all_regions_data:
            all_vars.update(region['data'].keys())

        if self.enable_climatology_anomaly:
            print(
                "启用训练期月气候态 anomaly: "
                f"变量={sorted(self.anomaly_variables)}, period={self.climatology_period}"
            )
        for region in self.all_regions_data:
            self._compute_region_climatology(region)

        self._estimate_damped_persistence_coefficients()

        # 如果提供了scalers，直接使用
        if self.provided_scalers is not None:
            print("使用提供的标准化参数 (防止数据泄露)...")
            self.scalers = self.provided_scalers
        else:
            print("计算全局标准化参数 (仅使用训练集数据)...")
            # 注意：只使用训练集时间段的数据来计算scaler，防止数据泄露
            # 使用 partial_fit 逐区域增量计算，避免将所有区域数据拼成大数组引发内核内存问题

            total_time_steps = len(self.times)
            train_end_idx = int(total_time_steps * self.train_ratio)
            print(f"标准化参数计算范围: 时间步 0 - {train_end_idx}")

            self.scalers = {}
            for var in all_vars:
                scaler = StandardScaler()
                for region in self.all_regions_data:
                    source_data = self._get_model_source_data(region, var)
                    if source_data is not None:
                        train_data = source_data[:train_end_idx]
                        data_2d = train_data.reshape(-1, 1)
                        scaler.partial_fit(data_2d)
                self.scalers[var] = scaler
                print(f"变量 {var} 全局标准化参数: 均值={scaler.mean_[0]:.4f}, 标准差={scaler.scale_[0]:.4f}")

            for var in self.target_variables:
                scaler_name = self._target_scaler_name(var)
                if scaler_name == var:
                    continue
                scaler = StandardScaler()
                fitted = False
                for region in self.all_regions_data:
                    source_data = self._get_target_source_data(region, var)
                    if source_data is None:
                        continue
                    scaler.partial_fit(source_data[:train_end_idx].reshape(-1, 1))
                    fitted = True
                if not fitted:
                    raise ValueError(f'目标变量 {var!r} 缺少可拟合数据')
                self.scalers[scaler_name] = scaler
                print(
                    f"目标变量 {scaler_name} 全局标准化参数: "
                    f"均值={scaler.mean_[0]:.4f}, 标准差={scaler.scale_[0]:.4f}"
                )

            if self.include_climatology_features:
                for var in self.climatology_feature_variables:
                    feature_name = self._climatology_feature_name(var)
                    scaler = StandardScaler()
                    for region in self.all_regions_data:
                        clim_series = self._build_climatology_series(region, var)
                        if clim_series is not None:
                            clim_2d = clim_series[:train_end_idx].reshape(-1, 1)
                            scaler.partial_fit(clim_2d)
                    self.scalers[feature_name] = scaler
                    print(f"变量 {feature_name} 全局标准化参数: 均值={scaler.mean_[0]:.4f}, 标准差={scaler.scale_[0]:.4f}")

            for var in self.tendency_feature_variables:
                feature_name = self._tendency_feature_name(var)
                scaler = StandardScaler()
                fitted = False
                for region in self.all_regions_data:
                    raw_data = region.get('data', {}).get(var)
                    if raw_data is None:
                        continue
                    tendency = self._build_tendency_feature(raw_data)
                    scaler.partial_fit(tendency[:train_end_idx].reshape(-1, 1))
                    fitted = True
                if fitted:
                    self.scalers[feature_name] = scaler
                    print(
                        f"变量 {feature_name} 全局标准化参数: "
                        f"均值={scaler.mean_[0]:.4f}, 标准差={scaler.scale_[0]:.4f}"
                    )

        # 使用全局参数标准化各区域数据 (对所有数据进行变换)
        for region in self.all_regions_data:
            region['normalized_data'] = {}
            for var, data in region['data'].items():
                if var in self.scalers:
                    source_data = self._get_model_source_data(region, var)
                    if source_data is None:
                        continue
                    original_shape = source_data.shape
                    data_2d = source_data.reshape(-1, 1)
                    data_normalized = self.scalers[var].transform(data_2d)
                    region['normalized_data'][var] = data_normalized.reshape(original_shape)

            region['normalized_target_data'] = {}
            for var in self.target_variables:
                scaler_name = self._target_scaler_name(var)
                if scaler_name == var:
                    if var in region['normalized_data']:
                        region['normalized_target_data'][var] = region['normalized_data'][var]
                    continue
                source_data = self._get_target_source_data(region, var)
                scaler = self.scalers.get(scaler_name)
                if source_data is None or scaler is None:
                    continue
                original_shape = source_data.shape
                transformed = scaler.transform(source_data.reshape(-1, 1))
                region['normalized_target_data'][var] = transformed.reshape(original_shape)

            if self.include_climatology_features:
                for var in self.climatology_feature_variables:
                    feature_name = self._climatology_feature_name(var)
                    clim_series = self._build_climatology_series(region, var)
                    if clim_series is None or feature_name not in self.scalers:
                        continue
                    original_shape = clim_series.shape
                    clim_2d = clim_series.reshape(-1, 1)
                    clim_normalized = self.scalers[feature_name].transform(clim_2d)
                    region['normalized_data'][feature_name] = clim_normalized.reshape(original_shape)

        for region in self.all_regions_data:
            self._add_auxiliary_encodings(region)

        # 设置主要数据（用于向后兼容）
        if self.all_regions_data:
            main_region = self.all_regions_data[0]  # 原始区域
            self.data_arrays = main_region['data']
            self.normalized_data = main_region['normalized_data']

    def _add_auxiliary_encodings(self, region: Dict):
        """为区域附加位置和时间编码等辅助特征"""
        normalized = region.get('normalized_data', {})
        coords = region.get('coords', {})
        if not normalized:
            return

        if getattr(self, 'include_tendency_features', False):
            for variable in self.tendency_feature_variables:
                feature_name = self._tendency_feature_name(variable)
                source = region.get('data', {}).get(variable)
                scaler = self.scalers.get(feature_name)
                if source is not None and scaler is not None:
                    tendency = self._build_tendency_feature(source)
                    normalized[feature_name] = scaler.transform(
                        tendency.reshape(-1, 1)
                    ).reshape(tendency.shape).astype(np.float32)

        sample_array = next(iter(region['data'].values()))
        time_steps = sample_array.shape[0]
        lat_size = len(coords.get('lats', self.lats))
        lon_size = len(coords.get('lons', self.lons))

        if self.config.get('enable_positional_encoding', False):
            spatial_features = self._build_spatial_encoding(coords, time_steps, lat_size, lon_size)
            if spatial_features is not None:
                normalized['SPATIAL_ENCODING'] = spatial_features.astype(np.float32)

        if self.config.get('enable_time_encoding', False):
            time_features = self._build_time_encoding(coords, time_steps, lat_size, lon_size)
            if time_features is not None:
                normalized['TIME_ENCODING'] = time_features.astype(np.float32)

    def _build_spatial_encoding(self, coords: Dict, time_steps: int, lat_size: int, lon_size: int) -> Optional[np.ndarray]:
        num_freq = int(self.config.get('positional_encoding_frequencies', 0))
        if num_freq <= 0:
            return None

        lons = coords.get('lons', self.lons)
        lats = coords.get('lats', self.lats)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        lon_grid = np.deg2rad(lon_grid.astype(np.float32))
        lat_grid = np.deg2rad(lat_grid.astype(np.float32))

        features = []
        # Integer angular harmonics preserve the 0°/360° longitude seam and
        # provide genuinely distinct spatial scales. Dividing radian values by
        # Transformer-style 10000**k produced near-constant, non-periodic maps.
        for harmonic in range(1, num_freq + 1):
            features.append(np.sin(harmonic * lon_grid))
            features.append(np.cos(harmonic * lon_grid))
        for harmonic in range(1, num_freq + 1):
            features.append(np.sin(harmonic * lat_grid))
            features.append(np.cos(harmonic * lat_grid))

        if not features:
            return None

        # Static coordinates are stored once and broadcast only for the sampled
        # 12-step sequence. Repeating them for all 121 times and 151 windows
        # wastes GiB without changing a single model value.
        return np.stack(features, axis=0)[np.newaxis, ...]

    def _extract_period_indices(self, times: np.ndarray, period: int) -> np.ndarray:
        """将 TIME 坐标转换为 0-based 周期索引，兼容 YYYYMM 整数和 datetime64。"""
        period = max(1, int(period))
        indices = []
        for idx, raw_time in enumerate(np.asarray(times)):
            month = None
            scalar = raw_time.item() if hasattr(raw_time, 'item') else raw_time

            if isinstance(scalar, np.datetime64):
                try:
                    month = int(np.datetime_as_string(scalar, unit='M')[-2:])
                except Exception:
                    month = None
            elif hasattr(scalar, 'month'):
                month = int(scalar.month)
            else:
                try:
                    value = int(float(scalar))
                    if value >= 10000:
                        month = value % 100
                    elif 1 <= value <= 12:
                        month = value
                except (TypeError, ValueError, OverflowError):
                    text = str(scalar)
                    if len(text) >= 6 and text[-2:].isdigit():
                        month = int(text[-2:])

            if month is None or month < 1 or month > 12:
                month = (idx % period) + 1
            indices.append((month - 1) % period)

        return np.asarray(indices, dtype=np.int64)

    def _extract_year_values(self, times: np.ndarray) -> Optional[np.ndarray]:
        years = []
        for raw_time in np.asarray(times):
            scalar = raw_time.item() if hasattr(raw_time, 'item') else raw_time
            if isinstance(scalar, np.datetime64):
                try:
                    years.append(float(np.datetime_as_string(scalar, unit='Y')[:4]))
                    continue
                except Exception:
                    return None
            if hasattr(scalar, 'year'):
                years.append(float(scalar.year))
                continue
            try:
                value = int(float(scalar))
                years.append(float(value // 100 if value >= 10000 else value))
            except (TypeError, ValueError, OverflowError):
                text = str(scalar)
                if len(text) >= 4 and text[:4].isdigit():
                    years.append(float(text[:4]))
                else:
                    return None
        return np.asarray(years, dtype=np.float32)

    def _build_time_encoding(self, coords: Dict, time_steps: int, lat_size: int, lon_size: int) -> Optional[np.ndarray]:
        num_freq = int(self.config.get('time_encoding_frequencies', 0))
        include_trend = self.config.get('include_year_trend', False)
        if num_freq <= 0 and not include_trend:
            return None

        period = max(1, int(self.config.get('time_encoding_period', 12)))
        times = np.array(coords.get('times', self.times))

        feature_maps = []

        if times.shape[0] != time_steps:
            raise ValueError(
                f'时间坐标长度 {times.shape[0]} 与数据时间长度 {time_steps} 不一致'
            )

        # 时间特征只随时间变化，空间轴保留为 singleton，并在取样时广播。
        # 预先复制到每个网格点会把几个 KiB 的信号放大成数 GiB。
        if num_freq > 0:
            time_channels = np.zeros((time_steps, num_freq * 2, 1, 1), dtype=np.float32)
            months = self._extract_period_indices(times, period).astype(np.float32)

            for idx, freq in enumerate(range(1, num_freq + 1)):
                angle = 2 * np.pi * months * freq / period
                time_channels[:, 2 * idx, 0, 0] = np.sin(angle)
                time_channels[:, 2 * idx + 1, 0, 0] = np.cos(angle)
            feature_maps.append(time_channels)

        # 年份趋势
        if include_trend:
            years = self._extract_year_values(times)

            if years is not None:
                year_span = float(np.max(years) - np.min(years))
                if year_span > 0:
                    year_norm = (years - np.min(years)) / year_span
                else:
                    year_norm = np.zeros_like(years)
                trend_values = ((year_norm - 0.5) * 2).astype(np.float32)
                feature_maps.append(trend_values[:, None, None, None])

        if not feature_maps:
            return None

        return np.concatenate(feature_maps, axis=1)

    def _fallback_month_extraction(self, times: np.ndarray, period: int) -> np.ndarray:
        """从时间戳中提取月份索引的兜底方法"""
        return self._extract_period_indices(times, period).astype(np.float32)

    def _fallback_year_extraction(self, times: np.ndarray) -> Optional[np.ndarray]:
        return self._extract_year_values(times)

    def _fill_missing_values(self, data: np.ndarray, variable: str = "unknown") -> np.ndarray:
        """
        填充缺失值。

        全局兜底均值只使用训练时间段计算，避免验证/测试时间段的信息
        进入预处理统计量。逐时间片仍优先使用该片已有空间值补洞；如果
        某片全为空，再使用训练段兜底均值。
        """
        # 获取有效数据的索引（NaN/Inf 都不得进入 scaler 或模型）
        valid_mask = np.isfinite(data)

        if valid_mask.sum() == 0:
            # 如果全部是缺失值，用0填充
            print(f"警告: 数据全部为缺失值，使用0填充")
            return np.zeros_like(data)

        train_end_idx = int(len(self.times) * self.train_ratio)
        train_slice = data[:train_end_idx]
        finite_train = train_slice[np.isfinite(train_slice)]
        if finite_train.size:
            global_mean = float(np.mean(finite_train, dtype=np.float64))
        else:
            # Never inspect validation/test values to repair training-time
            # preprocessing statistics. A zero sentinel is explicit and can be
            # detected by the input ablations/data audit.
            print(f"警告: 变量 {variable} 的训练段无有效值，使用0填充")
            global_mean = 0.0

        # 根据数据维度进行不同的处理
        if data.ndim == 4:  # 4D数据: (time, level, lat, lon)
            for t in range(data.shape[0]):
                for d in range(data.shape[1]):
                    slice_data = data[t, d, :, :]
                    if not np.isfinite(slice_data).all():
                        # 首先尝试用该切片的有效值均值填充
                        finite_slice = slice_data[np.isfinite(slice_data)]
                        slice_mean = float(np.mean(finite_slice)) if finite_slice.size else np.nan
                        if np.isfinite(slice_mean):
                            slice_data[~np.isfinite(slice_data)] = slice_mean
                        else:
                            # 如果该切片全为NaN，使用全局均值
                            slice_data[~np.isfinite(slice_data)] = global_mean
                        data[t, d, :, :] = slice_data
        elif data.ndim == 3:  # 3D数据: (time, lat, lon)
            for t in range(data.shape[0]):
                slice_data = data[t, :, :]
                if not np.isfinite(slice_data).all():
                    # 首先尝试用该时间步的有效值均值填充
                    finite_slice = slice_data[np.isfinite(slice_data)]
                    slice_mean = float(np.mean(finite_slice)) if finite_slice.size else np.nan
                    if np.isfinite(slice_mean):
                        slice_data[~np.isfinite(slice_data)] = slice_mean
                    else:
                        # 如果该时间步全为NaN，使用全局均值
                        slice_data[~np.isfinite(slice_data)] = global_mean
                    data[t, :, :] = slice_data

        # 最终检查：如果还有NaN，用全局均值填充
        if not np.isfinite(data).all():
            print(f"警告: 插值后仍有非有限值，使用全局均值 {global_mean:.6f} 填充")
            data[~np.isfinite(data)] = global_mean

        return data

    def _split_data(self):
        """
        分割数据集
        """
        total_time_steps = len(self.times)

        train_end = int(total_time_steps * self.train_ratio)
        val_end = int(total_time_steps * (self.train_ratio + self.val_ratio))

        if self.mode == 'train':
            self.time_indices = list(range(0, train_end))
        elif self.mode == 'val':
            self.time_indices = list(range(train_end, val_end))
        else:  # test
            self.time_indices = list(range(val_end, total_time_steps))

        print(f"{self.mode.upper()}集时间范围: {self.time_indices[0]} - {self.time_indices[-1]} (共{len(self.time_indices)}个时步)")

    def _create_sequences(self):
        """
        创建时序序列（包括所有区域的数据）。

        ``carry_history`` 下以预测目标所属分段为准：验证/测试可以使用
        分界前已经观测到的历史，但所有预测目标仍严格留在当前分段。
        """
        self.sequences = []

        # 获取该模式对应的时间范围
        if not hasattr(self, 'time_indices') or len(self.time_indices) == 0:
            print(f"警告: {self.mode}模式的时间索引为空")
            return

        segment_start = min(self.time_indices)
        segment_end = max(self.time_indices)

        # 确保有足够的时间步来创建序列
        required_length = self.sequence_length + self.prediction_length
        available_length = segment_end - segment_start + 1

        context_policy = self.config.get('split_context_policy', 'carry_history')
        minimum_segment_length = (
            required_length if self.mode == 'train' or context_policy == 'strict_segment'
            else self.prediction_length
        )
        if available_length < minimum_segment_length:
            print(
                f"警告: {self.mode}集时间步不足。需要{minimum_segment_length}，"
                f"可用{available_length}"
            )
            return

        if self.mode == 'train' or context_policy == 'strict_segment':
            earliest_start = segment_start
        else:
            earliest_start = max(0, segment_start - self.sequence_length)
        latest_start = segment_end - required_length + 1

        # 为每个区域创建序列（train/val/test 均包含所有100%海洋窗口）
        for region_idx, region_info in enumerate(self.all_regions_data):
            region_type = region_info['region_type']

            region_sequences = []
            for t in range(earliest_start, latest_start + 1):
                target_start = t + self.sequence_length
                target_end = target_start + self.prediction_length - 1
                if target_start >= segment_start and target_end <= segment_end:
                    region_sequences.append((t, region_idx))

            self.sequences.extend(region_sequences)

            print(f"区域 {region_idx} ({region_type}): {len(region_sequences)} 个序列")

        print(f"{self.mode.upper()}集总序列数量: {len(self.sequences)}")
        if len(self.sequences) > 0:
            first_start = min(start for start, _ in self.sequences)
            last_start = max(start for start, _ in self.sequences)
            print(
                f"  目标时间范围: {segment_start} 到 {segment_end}; "
                f"历史起点范围: {first_start} 到 {last_start}"
            )
        else:
            print(
                f"  警告: 无法创建序列！目标范围: {segment_start}-{segment_end}, "
                f"需要预测长度: {self.prediction_length}"
            )

    def __len__(self) -> int:
        """
        返回数据集大小
        """
        return len(self.sequences)

    def build_sample_provenance(self, sample_indices: Optional[List[int]] = None) -> Dict:
        """Describe the exact temporal origins and spatial windows in an evaluation.

        The returned arrays follow ``sample_indices`` order, so they can be
        paired directly with collected predictions even when a grouped batch
        sampler changes iteration order.
        """
        if sample_indices is None:
            sample_indices = list(range(len(self.sequences)))
        else:
            sample_indices = [int(value) for value in sample_indices]

        samples = []
        origins = {}
        regions = {}
        origin_ids = []
        region_ids = []
        target_time_indices = []
        target_period_ids = []

        def time_label(index: int) -> str:
            return str(self.times[index])

        for sample_idx in sample_indices:
            if sample_idx < 0 or sample_idx >= len(self.sequences):
                raise IndexError(f'sample index out of range: {sample_idx}')
            start_idx, region_idx = self.sequences[sample_idx]
            start_idx = int(start_idx)
            region_idx = int(region_idx)
            history_end = start_idx + self.sequence_length - 1
            target_start = history_end + 1
            future_indices = list(range(target_start, target_start + self.prediction_length))
            future_periods = [int(self.time_period_indices[index]) for index in future_indices]

            origin_key = str(start_idx)
            if origin_key not in origins:
                origins[origin_key] = {
                    'history_start_index': start_idx,
                    'history_end_index': history_end,
                    'target_start_index': target_start,
                    'target_end_index': future_indices[-1],
                    'history_start_time': time_label(start_idx),
                    'history_end_time': time_label(history_end),
                    'target_times': [time_label(index) for index in future_indices],
                    'target_period_ids': future_periods,
                }

            region_key = str(region_idx)
            if region_key not in regions:
                region = self.all_regions_data[region_idx]
                coords = region.get('coords', {})
                lons = np.asarray(coords.get('lons', []), dtype=np.float64)
                lats = np.asarray(coords.get('lats', []), dtype=np.float64)
                regions[region_key] = {
                    'region_type': region.get('region_type', 'unknown'),
                    'lon_range': (
                        [float(value) for value in region.get('lon_range', [])]
                        if region.get('lon_range') is not None else []
                    ),
                    'lat_range': (
                        [float(value) for value in region.get('lat_range', [])]
                        if region.get('lat_range') is not None else []
                    ),
                    'center_lon': float(np.mean(lons)) if lons.size else None,
                    'center_lat': float(np.mean(lats)) if lats.size else None,
                }

            origin_ids.append(start_idx)
            region_ids.append(region_idx)
            target_time_indices.append(future_indices)
            target_period_ids.append(future_periods)
            samples.append({
                'sample_index': sample_idx,
                'origin_id': start_idx,
                'region_id': region_idx,
            })

        return {
            'samples': samples,
            'origin_ids': origin_ids,
            'region_ids': region_ids,
            'target_time_indices': target_time_indices,
            'target_period_ids': target_period_ids,
            'origins': origins,
            'regions': regions,
            'period_definition': (
                f'zero-based phase within climatology_period={self.climatology_period}; '
                'for monthly data, 0=January and 11=December'
            ),
        }

    def _initialize_channel_schema(self):
        """在主进程中确定通道布局，避免 DataLoader worker 内惰性初始化丢失。"""
        if not self.all_regions_data:
            return
        normalized_data = self.all_regions_data[0].get('normalized_data', {})

        actual_input_variables = [
            var for var in self.input_variables if var in normalized_data
        ]
        if self.include_climatology_features:
            for var in self.climatology_feature_variables:
                feature_name = self._climatology_feature_name(var)
                if feature_name in normalized_data:
                    actual_input_variables.append(feature_name)
        if getattr(self, 'include_tendency_features', False):
            for var in self.tendency_feature_variables:
                feature_name = self._tendency_feature_name(var)
                if feature_name in normalized_data:
                    actual_input_variables.append(feature_name)
        if self.config.get('enable_positional_encoding', False):
            if 'SPATIAL_ENCODING' in normalized_data:
                actual_input_variables.append('SPATIAL_ENCODING')
        if self.config.get('enable_time_encoding', False) and 'TIME_ENCODING' in normalized_data:
            actual_input_variables.append('TIME_ENCODING')

        self.actual_input_variables = actual_input_variables
        offset = 0
        for var_name in self.actual_input_variables:
            arr = normalized_data[var_name]
            channels = int(arr.shape[1]) if arr.ndim == 4 else 1
            self.input_channel_slices[var_name] = slice(offset, offset + channels)
            offset += channels

        offset = 0
        for var_name in self.target_variables:
            arr = normalized_data.get(var_name)
            if arr is None:
                continue
            channels = int(arr.shape[1]) if arr.ndim == 4 else 1
            self.target_channel_slices[var_name] = slice(offset, offset + channels)
            offset += channels

    def _validate_channel_schema(self):
        """Fail when configured variables/features did not become real channels."""
        missing_inputs = [
            name for name in self.input_variables
            if name not in self.input_channel_slices
        ]
        missing_targets = [
            name for name in self.target_variables
            if name not in self.target_channel_slices
        ]
        expected_features = []
        if self.include_climatology_features:
            expected_features.extend(
                self._climatology_feature_name(name)
                for name in self.climatology_feature_variables
            )
        if getattr(self, 'include_tendency_features', False):
            expected_features.extend(
                self._tendency_feature_name(name)
                for name in self.tendency_feature_variables
            )
        if (
            self.config.get('enable_positional_encoding', False)
            and int(self.config.get('positional_encoding_frequencies', 0)) > 0
        ):
            expected_features.append('SPATIAL_ENCODING')
        if (
            self.config.get('enable_time_encoding', False)
            and (
                int(self.config.get('time_encoding_frequencies', 0)) > 0
                or self.config.get('include_year_trend', False)
            )
        ):
            expected_features.append('TIME_ENCODING')
        missing_features = [
            name for name in expected_features
            if name not in self.input_channel_slices
        ]
        if missing_inputs or missing_targets or missing_features:
            raise ValueError(
                '数据通道 schema 与配置不一致；拒绝静默删除变量。'
                f'缺失输入={missing_inputs}, 缺失目标={missing_targets}, '
                f'缺失特征={missing_features}'
            )

    @staticmethod
    def _slice_and_broadcast_input(
        data: np.ndarray,
        start_idx: int,
        length: int,
        height: int,
        width: int,
        *,
        variable: str,
        allow_static_time: bool = False,
        allow_spatial_broadcast: bool = False,
    ) -> np.ndarray:
        """严格切片输入，并只在语义允许的 singleton 轴上零拷贝广播。"""
        array = np.asarray(data)
        if array.ndim not in (3, 4):
            raise ValueError(f'{variable} 输入维度必须为 3 或 4，实际为 {array.shape}')
        if start_idx < 0 or length <= 0:
            raise ValueError(f'{variable} 的切片参数非法: start={start_idx}, length={length}')

        if array.shape[0] == 1 and allow_static_time:
            window = array
        else:
            end_idx = start_idx + length
            if end_idx > array.shape[0]:
                raise IndexError(
                    f'{variable} 时间长度不足: 请求 [{start_idx}:{end_idx}]，'
                    f'实际长度={array.shape[0]}'
                )
            window = array[start_idx:end_idx]

        if window.shape[0] == 1 and length != 1:
            if not allow_static_time:
                raise ValueError(f'{variable} 不允许沿时间轴广播')
            window = np.broadcast_to(window, (length, *window.shape[1:]))
        elif window.shape[0] != length:
            raise ValueError(
                f'{variable} 时间窗口长度不一致: expected={length}, actual={window.shape[0]}'
            )

        source_height, source_width = window.shape[-2:]
        if (source_height, source_width) != (height, width):
            if (
                not allow_spatial_broadcast
                or source_height not in (1, height)
                or source_width not in (1, width)
            ):
                raise ValueError(
                    f'{variable} 空间形状不兼容: expected={(height, width)}, '
                    f'actual={(source_height, source_width)}'
                )
            window = np.broadcast_to(
                window,
                (*window.shape[:-2], height, width),
            )
        return window

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            输入序列和目标序列的元组
        """
        if getattr(self, '_preassembled_mmap_enabled', False):
            return self._get_preassembled_item(idx)

        # 解析序列信息：(时间起始索引, 区域索引)
        start_idx, region_idx = self.sequences[idx]

        # 获取对应区域的数据
        region_info = self.all_regions_data[region_idx]
        normalized_data = region_info['normalized_data']
        normalized_target_data = (
            region_info.get('normalized_target_data') or normalized_data
        )

        # 更新输入变量列表（第一次调用时）
        if not hasattr(self, 'actual_input_variables'):
            valid_input_vars = []
            for var in self.input_variables:
                if var in normalized_data:
                    valid_input_vars.append(var)
            if self.include_climatology_features:
                for var in self.climatology_feature_variables:
                    feature_name = self._climatology_feature_name(var)
                    if feature_name in normalized_data:
                        valid_input_vars.append(feature_name)
            if getattr(self, 'include_tendency_features', False):
                for var in self.tendency_feature_variables:
                    feature_name = self._tendency_feature_name(var)
                    if feature_name in normalized_data:
                        valid_input_vars.append(feature_name)
            if self.config.get('enable_positional_encoding', False):
                if 'SPATIAL_ENCODING' in normalized_data:
                    valid_input_vars.append('SPATIAL_ENCODING')
            if self.config.get('enable_time_encoding', False) and 'TIME_ENCODING' in normalized_data:
                valid_input_vars.append('TIME_ENCODING')
            self.actual_input_variables = valid_input_vars

        reference_variable = next(
            (name for name in self.input_variables if name in normalized_data),
            None,
        )
        if reference_variable is None:
            raise ValueError('没有可用于确定样本空间形状的物理输入变量')
        reference_data = np.asarray(normalized_data[reference_variable])
        if reference_data.ndim not in (3, 4):
            raise ValueError(
                f'{reference_variable} 输入维度必须为 3 或 4，实际为 {reference_data.shape}'
            )
        height, width = reference_data.shape[-2:]

        # 构建输入序列
        input_sequence = []
        for var in self.actual_input_variables:
            if var in normalized_data:
                var_data = self._slice_and_broadcast_input(
                    normalized_data[var],
                    start_idx,
                    self.sequence_length,
                    height,
                    width,
                    variable=var,
                    allow_static_time=(var == 'SPATIAL_ENCODING'),
                    allow_spatial_broadcast=(var == 'TIME_ENCODING'),
                )
                input_sequence.append(var_data)

        # 构建目标序列
        target_sequence = []
        for var in self.target_variables:
            if var in normalized_target_data:
                var_data = normalized_target_data[var][start_idx + self.sequence_length:start_idx + self.sequence_length + self.prediction_length]
                target_sequence.append(var_data)

        # 统一为 channel-first，避免先转置到 channel-last、拼接后再转回去。
        # 预处理缓存中的物理变量约定为 (time, channels, height, width)。
        input_arrays = []
        input_channel_lengths = []
        for var_data in input_sequence:
            if var_data.ndim == 3:  # (seq_len, lat, lon)
                var_data = var_data[:, np.newaxis, :, :]
            elif var_data.ndim != 4:
                raise ValueError(f'输入变量样本维度必须为 3 或 4，实际为 {var_data.shape}')
            input_arrays.append(np.asarray(var_data, dtype=np.float32))
            input_channel_lengths.append(var_data.shape[1])

        if not self.input_channel_slices:
            channel_offset = 0
            for var_name, channel_len in zip(self.actual_input_variables, input_channel_lengths):
                self.input_channel_slices[var_name] = slice(channel_offset, channel_offset + channel_len)
                channel_offset += channel_len

        target_arrays = []
        target_channel_lengths = []
        for var_data in target_sequence:
            if var_data.ndim == 3:  # (pred_len, lat, lon)
                var_data = var_data[:, np.newaxis, :, :]
            elif var_data.ndim != 4:
                raise ValueError(f'目标变量样本维度必须为 3 或 4，实际为 {var_data.shape}')
            target_arrays.append(np.asarray(var_data, dtype=np.float32))
            target_channel_lengths.append(var_data.shape[1])

        if not self.target_channel_slices:
            channel_offset = 0
            for var_name, channel_len in zip(self.target_variables, target_channel_lengths):
                self.target_channel_slices[var_name] = slice(channel_offset, channel_offset + channel_len)
                channel_offset += channel_len

        # 在 channel 维连接不同变量；from_numpy 避免再次复制完整样本。
        input_seq = np.ascontiguousarray(np.concatenate(input_arrays, axis=1))
        target_seq = np.ascontiguousarray(np.concatenate(target_arrays, axis=1))

        # 数据已经是 (time, channels, height, width)，无需再 permute。
        input_tensor = torch.from_numpy(input_seq)
        target_tensor = torch.from_numpy(target_seq)

        if self.return_sample_index:
            return input_tensor, target_tensor, idx
        return input_tensor, target_tensor

    def _get_preassembled_item(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Slice one sample from the shared time-axis mmap cache."""
        start_idx, region_idx = self.sequences[idx]
        input_values = self._mmap_inputs[
            region_idx, start_idx:start_idx + self.sequence_length
        ][:, self._preassembled_input_selection, :, :]
        target_start = start_idx + self.sequence_length
        target_source = (
            self._mmap_inputs
            if self.enable_target_climatology_anomaly
            else self._mmap_fullfield_targets
        )
        target_values = target_source[
            region_idx, target_start:target_start + self.prediction_length
        ][:, self._preassembled_target_selection, :, :]
        input_tensor = torch.from_numpy(np.asarray(input_values, dtype=np.float32))
        target_tensor = torch.from_numpy(np.asarray(target_values, dtype=np.float32))
        if self.return_sample_index:
            return input_tensor, target_tensor, idx
        return input_tensor, target_tensor

    @staticmethod
    def _series_to_channel_first(data: np.ndarray) -> np.ndarray:
        """将 (T, LEVEL, H, W) 或 (T, H, W) 转为 (T, C, H, W)。"""
        if data.ndim == 4:
            return data
        if data.ndim == 3:
            return data[:, np.newaxis, :, :]
        raise ValueError(f"不支持的序列维度: {data.shape}")

    def _resolve_target_slice(self, variable: str, var_idx: int, total_channels: int) -> slice:
        del var_idx
        ch_slice = self.target_channel_slices.get(variable)
        if ch_slice is None:
            raw_slices = self.config.get('target_channel_slices', {})
            raw_slice = raw_slices.get(variable) if isinstance(raw_slices, dict) else None
            if isinstance(raw_slice, (list, tuple)) and len(raw_slice) >= 2:
                ch_slice = slice(int(raw_slice[0]), int(raw_slice[1]))
        if ch_slice is None or ch_slice.start is None or ch_slice.stop is None:
            raise ValueError(f'目标变量 {variable!r} 缺少显式 channel slice')
        start, stop = int(ch_slice.start), int(ch_slice.stop)
        if not 0 <= start < stop <= int(total_channels):
            raise ValueError(
                f'目标变量 {variable!r} 的 channel slice [{start}, {stop}) '
                f'超出总通道数 {total_channels}'
            )
        return slice(start, stop)

    def _validated_sample_indices(
        self,
        sample_indices: Optional[List[int]],
        sample_count: int,
    ) -> List[int]:
        indices = (
            list(range(sample_count))
            if sample_indices is None else [int(index) for index in sample_indices]
        )
        if len(indices) != sample_count:
            raise ValueError(
                f'sample_indices 数量 {len(indices)} 与样本数 {sample_count} 不一致'
            )
        invalid = [index for index in indices if index < 0 or index >= len(self.sequences)]
        if invalid:
            raise IndexError(f'sample_indices 超出数据集范围: {invalid[:10]}')
        return indices

    def _target_climatology_channels(
        self,
        region: Dict,
        variable: str,
        start_idx: int,
        length: int
    ) -> Optional[np.ndarray]:
        climatology = region.get('climatology', {}).get(variable)
        if climatology is None:
            return None
        period_indices = self.time_period_indices[start_idx:start_idx + length]
        if len(period_indices) < length:
            return None
        return self._series_to_channel_first(climatology[period_indices])

    def _preassembled_raw_target_channels(
        self,
        region_idx: int,
        variable: str,
        start_idx: int,
        length: int,
    ) -> np.ndarray:
        """Restore a short physical target slice from the full-field mmap."""
        source_slice = self._preassembled_target_source_slices[variable]
        normalized = np.asarray(
            self._mmap_fullfield_targets[
                region_idx,
                start_idx:start_idx + length,
                source_slice,
                :, :,
            ],
            dtype=np.float32,
        )
        scaler_name = self._preassembled_fullfield_scaler_names[variable]
        scaler = self.scalers[scaler_name]
        return scaler.inverse_transform(
            normalized.reshape(-1, 1)
        ).reshape(normalized.shape).astype(np.float32, copy=False)

    def inverse_transform_targets(
        self,
        data: np.ndarray,
        sample_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        将模型目标空间从标准化值恢复到物理量。

        若启用了 climatology anomaly，先执行 scaler inverse 得到 anomaly，
        再按样本对应的未来月份把训练期月气候态加回去。
        """
        array = np.asarray(data)
        squeeze_sample = False
        if array.ndim == 4:
            array = array[np.newaxis, ...]
            squeeze_sample = True
        if array.ndim != 5:
            raise ValueError(f"目标反标准化期望 4D/5D 数组，收到: {array.shape}")

        working_dtype = np.float64 if array.dtype == np.float64 else np.float32
        restored = array.astype(working_dtype, copy=True)
        sample_count, pred_steps, total_channels = restored.shape[:3]

        sample_indices = self._validated_sample_indices(sample_indices, sample_count)

        for var_idx, var_name in enumerate(self.target_variables):
            ch_slice = self._resolve_target_slice(var_name, var_idx, total_channels)
            if ch_slice.stop <= ch_slice.start:
                continue

            scaler = self.scalers.get(self._target_scaler_name(var_name))
            if scaler is None:
                raise ValueError(f'目标变量 {var_name!r} 缺少训练期 scaler')
            var_block = restored[:, :, ch_slice, :, :]
            original_shape = var_block.shape
            var_block = scaler.inverse_transform(var_block.reshape(-1, 1)).reshape(original_shape)

            if self._is_target_anomaly_variable(var_name):
                for out_idx, sample_idx in enumerate(sample_indices):
                    start_idx, region_idx = self.sequences[int(sample_idx)]
                    target_start = start_idx + self.sequence_length
                    region = self.all_regions_data[region_idx]
                    clim = self._target_climatology_channels(
                        region,
                        var_name,
                        target_start,
                        pred_steps
                    )
                    if clim is None or clim.shape != var_block[out_idx].shape:
                        raise ValueError(
                            f'样本 {sample_idx} 的 {var_name} 气候态形状不完整'
                        )
                    var_block[out_idx] = var_block[out_idx] + clim

            restored[:, :, ch_slice, :, :] = var_block

        return restored[0] if squeeze_sample else restored

    def transform_targets_to_model_space(
        self,
        data: np.ndarray,
        sample_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """将物理量目标转换为当前模型训练空间。"""
        array = np.asarray(data)
        squeeze_sample = False
        if array.ndim == 4:
            array = array[np.newaxis, ...]
            squeeze_sample = True
        if array.ndim != 5:
            raise ValueError(f"目标标准化期望 4D/5D 数组，收到: {array.shape}")

        working_dtype = np.float64 if array.dtype == np.float64 else np.float32
        transformed = array.astype(working_dtype, copy=True)
        sample_count, pred_steps, total_channels = transformed.shape[:3]

        sample_indices = self._validated_sample_indices(sample_indices, sample_count)

        for var_idx, var_name in enumerate(self.target_variables):
            ch_slice = self._resolve_target_slice(var_name, var_idx, total_channels)
            if ch_slice.stop <= ch_slice.start:
                continue

            var_block = transformed[:, :, ch_slice, :, :]
            if self._is_target_anomaly_variable(var_name):
                for out_idx, sample_idx in enumerate(sample_indices):
                    start_idx, region_idx = self.sequences[int(sample_idx)]
                    target_start = start_idx + self.sequence_length
                    region = self.all_regions_data[region_idx]
                    clim = self._target_climatology_channels(
                        region,
                        var_name,
                        target_start,
                        pred_steps
                    )
                    if clim is None or clim.shape != var_block[out_idx].shape:
                        raise ValueError(
                            f'样本 {sample_idx} 的 {var_name} 气候态形状不完整'
                        )
                    var_block[out_idx] = var_block[out_idx] - clim

            scaler = self.scalers.get(self._target_scaler_name(var_name))
            if scaler is None:
                raise ValueError(f'目标变量 {var_name!r} 缺少训练期 scaler')
            original_shape = var_block.shape
            var_block = scaler.transform(var_block.reshape(-1, 1)).reshape(original_shape)

            transformed[:, :, ch_slice, :, :] = var_block

        return transformed[0] if squeeze_sample else transformed

    def build_reference_forecasts(
        self,
        sample_indices: Optional[List[int]] = None,
        spaces: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        构造 Climatology / Persistence / Anomaly Persistence / Damped AP baseline。

        Returns:
            {
              'physical': {name: (N, T, C, H, W)},
              'normalized': {name: (N, T, C, H, W)}
            }
        """
        requested_spaces = set(spaces or ('physical', 'normalized'))
        unknown_spaces = requested_spaces - {'physical', 'normalized'}
        if unknown_spaces or not requested_spaces:
            raise ValueError(
                f'基线空间必须是 physical/normalized 的非空子集，实际={sorted(requested_spaces)}'
            )

        if sample_indices is None:
            sample_indices = list(range(len(self.sequences)))
        else:
            sample_indices = [int(index) for index in sample_indices]
        sample_indices = self._validated_sample_indices(
            sample_indices,
            len(sample_indices),
        )

        if len(sample_indices) == 0:
            empty = np.empty((0, self.prediction_length, 0, 0, 0), dtype=np.float32)
            return {'physical': {}, 'normalized': {}}

        if not self.target_channel_slices and len(self.sequences) > 0:
            _ = self[0]

        physical = None

        for output_idx, sample_idx in enumerate(sample_indices):
            start_idx, region_idx = self.sequences[int(sample_idx)]
            target_start = start_idx + self.sequence_length
            last_hist_idx = target_start - 1
            region = self.all_regions_data[region_idx]

            clim_parts = []
            persistence_parts = []
            anomaly_persistence_parts = []
            damped_anomaly_persistence_parts = []

            for var_name in self.target_variables:
                raw_data = region.get('data', {}).get(var_name)
                future_clim = self._target_climatology_channels(
                    region,
                    var_name,
                    target_start,
                    self.prediction_length
                )
                if future_clim is None:
                    if raw_data is None:
                        raise ValueError(
                            f'样本 {sample_idx} 的 {var_name} 缺少气候态和原始数据'
                        )
                    train_end_idx = int(len(self.times) * self.train_ratio)
                    fallback = np.nanmean(raw_data[:train_end_idx], axis=0).astype(np.float32)
                    fallback = np.nan_to_num(fallback, nan=0.0)
                    future_clim = np.repeat(
                        self._series_to_channel_first(fallback[np.newaxis, ...]),
                        self.prediction_length,
                        axis=0
                    )

                if raw_data is None and getattr(self, '_preassembled_mmap_enabled', False):
                    last_raw = self._preassembled_raw_target_channels(
                        region_idx, var_name, last_hist_idx, 1
                    )[0]
                elif raw_data is not None:
                    last_raw = self._series_to_channel_first(
                        raw_data[last_hist_idx:last_hist_idx + 1]
                    )[0]
                else:
                    raise ValueError(f'样本 {sample_idx} 的 {var_name} 缺少原始数据')
                persistence = np.repeat(last_raw[np.newaxis, ...], self.prediction_length, axis=0)

                hist_clim = self._target_climatology_channels(
                    region,
                    var_name,
                    last_hist_idx,
                    1
                )
                if hist_clim is None:
                    hist_clim = future_clim[:1]
                last_anomaly = last_raw - hist_clim[0]
                anomaly_persistence = future_clim + last_anomaly[np.newaxis, ...]
                coefficients = self.damped_persistence_coefficients.get(var_name)
                if coefficients is None:
                    raise RuntimeError(f'{var_name} 缺少训练期 damped persistence 系数')
                expected_shape = (self.prediction_length, last_anomaly.shape[0])
                if coefficients.shape != expected_shape:
                    raise ValueError(
                        f'{var_name} damped persistence 系数形状错误: '
                        f'{coefficients.shape} != {expected_shape}'
                    )
                damped_anomaly_persistence = (
                    future_clim
                    + coefficients[:, :, np.newaxis, np.newaxis]
                    * last_anomaly[np.newaxis, ...]
                )

                clim_parts.append(future_clim)
                persistence_parts.append(persistence)
                anomaly_persistence_parts.append(anomaly_persistence)
                damped_anomaly_persistence_parts.append(damped_anomaly_persistence)

            sample_forecasts = {
                'climatology': np.concatenate(clim_parts, axis=1),
                'persistence': np.concatenate(persistence_parts, axis=1),
                'anomaly_persistence': np.concatenate(
                    anomaly_persistence_parts, axis=1
                ),
                'damped_anomaly_persistence': np.concatenate(
                    damped_anomaly_persistence_parts, axis=1
                ),
            }
            if physical is None:
                physical = {
                    name: np.empty(
                        (len(sample_indices), *values.shape),
                        dtype=np.float32,
                    )
                    for name, values in sample_forecasts.items()
                }
            for name, values in sample_forecasts.items():
                if values.shape != physical[name].shape[1:]:
                    raise ValueError(
                        f'{name} 基线形状在样本间不一致: '
                        f'{values.shape} != {physical[name].shape[1:]}'
                    )
                physical[name][output_idx] = values

        if physical is None:
            raise RuntimeError('非空 sample_indices 未生成任何基线')

        normalized = {}
        if 'normalized' in requested_spaces:
            keep_physical = 'physical' in requested_spaces
            for name in tuple(physical):
                values = physical[name] if keep_physical else physical.pop(name)
                normalized[name] = self.transform_targets_to_model_space(
                    values,
                    sample_indices=sample_indices,
                ).astype(np.float32, copy=False)

        return {
            'physical': physical if 'physical' in requested_spaces else {},
            'normalized': normalized,
        }

    def get_scaler(self, variable: str) -> Optional[StandardScaler]:
        """
        获取指定变量的标准化器
        """
        return self.scalers.get(variable, None)

    def inverse_transform(self, data: np.ndarray, variable: str) -> np.ndarray:
        """
        反标准化数据
        """
        if variable in self.scalers:
            original_shape = data.shape
            data_2d = data.reshape(-1, 1)
            data_inversed = self.scalers[variable].inverse_transform(data_2d)
            return data_inversed.reshape(original_shape)
        return data


class TimeGroupedBatchSampler(Sampler[List[int]]):
    """
    将同一输入起点时间的不同空间窗口放入同一批次。

    Global Token Bank 依赖 batch 内样本代表同一个历史时间段，
    否则跨窗口 attention 会混入不同时间的状态。
    """

    def __init__(
        self,
        dataset: OceanDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = shuffle
        self.drop_last = drop_last

        grouped: Dict[int, List[int]] = {}
        for sample_idx, (start_idx, _) in enumerate(dataset.sequences):
            grouped.setdefault(int(start_idx), []).append(sample_idx)
        self.groups = list(grouped.values())

    @property
    def max_group_size(self) -> int:
        return max((len(group) for group in self.groups), default=0)

    def __iter__(self):
        groups = [list(group) for group in self.groups]
        if self.shuffle:
            random.shuffle(groups)

        for group in groups:
            if self.shuffle or self.drop_last:
                if self.shuffle:
                    random.shuffle(group)
                batches = [group[offset:offset + self.batch_size] for offset in range(0, len(group), self.batch_size)]
            else:
                batch_count = (len(group) + self.batch_size - 1) // self.batch_size
                batches = [group[offset::batch_count] for offset in range(batch_count)]
            for batch in batches:
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    yield batch

    def __len__(self) -> int:
        total = 0
        for group in self.groups:
            if self.drop_last:
                total += len(group) // self.batch_size
            else:
                total += (len(group) + self.batch_size - 1) // self.batch_size
        return total


def validate_expected_canonical_window_count(datasets, expected_count):
    """Fail fast when the frozen spatial protocol no longer matches the data."""
    if expected_count is None:
        return
    actual = {
        name: len(dataset.all_regions_data)
        for name, dataset in datasets.items()
    }
    if isinstance(expected_count, dict):
        missing = sorted(set(actual) - set(expected_count))
        if missing:
            raise ValueError(
                'expected_canonical_windows_per_origin 缺少 split: '
                f'{missing}'
            )
        expected = {name: int(expected_count[name]) for name in actual}
    else:
        expected = {name: int(expected_count) for name in actual}
    if any(count <= 0 for count in expected.values()):
        raise ValueError('expected_canonical_windows_per_origin 必须为正整数或 split 映射')
    mismatched = {
        name: {'expected': expected[name], 'actual': count}
        for name, count in actual.items()
        if count != expected[name]
    }
    if mismatched:
        raise ValueError(
            'canonical 空间窗口协议已变化；请先运行 '
            'scripts/audit_dataset_protocol.py 审计数据，并显式更新冻结配置。'
            f'不匹配={mismatched}'
        )


def create_data_loaders(
    data_path: str,
    config: dict,
    batch_size: int = 4,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器

    Args:
        data_path: 数据文件路径
        config: 配置字典
        batch_size: 批次大小
        num_workers: 工作进程数

    Returns:
        训练、验证、测试数据加载器的元组
    """
    # 创建数据集，使用配置中的分割比例
    train_ratio = config.get('train_ratio', 0.6)
    val_ratio = config.get('val_ratio', 0.2)

    # 首先创建训练集，计算并获取scalers
    print("初始化训练集...")
    train_dataset = OceanDataset(data_path, config, mode='train',
                                train_ratio=train_ratio, val_ratio=val_ratio)

    # 获取训练集计算出的scalers
    scalers = train_dataset.scalers

    # 时间 split 不应复制整套空间数组。只有模式对应的滑窗开关/步长不同
    # （例如区域训练、全球验证）时，才构建独立的预处理 payload。
    print("初始化验证集 (使用训练集标准化参数)...")
    if train_dataset.can_share_preprocessed_with_mode('val'):
        print("[MEMORY] 验证集复用训练集只读预处理数组")
        val_dataset = train_dataset.temporal_split_view('val')
    else:
        val_dataset = OceanDataset(
            data_path, config, mode='val',
            train_ratio=train_ratio, val_ratio=val_ratio,
            scalers=scalers,
        )

    print("初始化测试集 (使用训练集标准化参数)...")
    if val_dataset.can_share_preprocessed_with_mode('test'):
        print("[MEMORY] 测试集复用验证集只读预处理数组")
        test_dataset = val_dataset.temporal_split_view('test')
    else:
        test_dataset = OceanDataset(
            data_path, config, mode='test',
            train_ratio=train_ratio, val_ratio=val_ratio,
            scalers=scalers,
        )

    # 检查数据集大小
    print(f"数据集大小检查:")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    validate_expected_canonical_window_count(
        {
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset,
        },
        config.get('expected_canonical_windows_per_origin'),
    )

    # 验证数据集大小
    if len(val_dataset) == 0:
        print("⚠️  警告: 验证集为空！这可能是因为:")
        print("   1. 时间序列长度(sequence_length)或预测长度(prediction_length)太长")
        print("   2. 验证集分割比例太小")
        print("   3. 数据总时间步数不足")
        print("   建议: 减少sequence_length/prediction_length或增加val_ratio")

    if len(train_dataset) == 0:
        raise ValueError("训练集为空，无法进行训练!")

    # 创建数据加载器
    # 仅在多进程时启用预取与持久化，以避免 PyTorch 对 num_workers=0 的限制
    use_prefetch = prefetch_factor if num_workers > 0 else None
    use_persistent = persistent_workers and num_workers > 0
    use_pin_memory = bool(config.get('pin_memory', True))
    group_batches = bool(config.get('group_batches_by_time', False))

    # 评估需要真实样本索引用于溯源与物理量恢复，因此所有 loader 都返回索引。
    train_dataset.return_sample_index = True
    val_dataset.return_sample_index = True
    test_dataset.return_sample_index = True

    if group_batches:
        print("启用同起报时次 batch sampler")

        train_sampler = TimeGroupedBatchSampler(train_dataset, batch_size, shuffle=True)
        val_sampler = TimeGroupedBatchSampler(val_dataset, batch_size, shuffle=False)
        test_sampler = TimeGroupedBatchSampler(test_dataset, batch_size, shuffle=False)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            prefetch_factor=use_prefetch,
            persistent_workers=use_persistent,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            prefetch_factor=use_prefetch,
            persistent_workers=use_persistent,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            prefetch_factor=use_prefetch,
            persistent_workers=use_persistent,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            prefetch_factor=use_prefetch,
            persistent_workers=use_persistent,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            prefetch_factor=use_prefetch,
            persistent_workers=use_persistent,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            prefetch_factor=use_prefetch,
            persistent_workers=use_persistent,
        )

    print(f"数据加载器创建完成:")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")

    return train_loader, val_loader, test_loader


def get_data_info(data_path: str) -> Dict:
    """
    获取数据基本信息

    Args:
        data_path: 数据文件路径

    Returns:
        数据信息字典
    """
    try:
        dataset = xr.open_dataset(data_path)

        info = {
            'dimensions': dict(dataset.dims),
            'variables': list(dataset.data_vars),
            'coordinates': list(dataset.coords),
            'time_range': [str(dataset.TIME.min().values), str(dataset.TIME.max().values)],
            'spatial_range': {
                'longitude': [float(dataset.LONGITUDE.min()), float(dataset.LONGITUDE.max())],
                'latitude': [float(dataset.LATITUDE.min()), float(dataset.LATITUDE.max())],
                'level': [float(dataset.LEVEL.min()), float(dataset.LEVEL.max())]
            }
        }

        dataset.close()
        return info

    except Exception as e:
        print(f"读取数据信息时出错: {e}")
        return {}


if __name__ == "__main__":
    # 测试数据加载器
    from config import DEFAULT_CONFIG

    data_path = "Data/FullData_preprocessed.nc"
    config = DEFAULT_CONFIG.copy()

    try:
        # 获取数据信息
        print("数据文件信息:")
        data_info = get_data_info(data_path)
        for key, value in data_info.items():
            print(f"  {key}: {value}")

        print("\n创建数据加载器...")
        train_loader, val_loader, test_loader = create_data_loaders(
            data_path, config, batch_size=2, num_workers=0
        )

        # 测试数据加载
        print("\n测试数据加载...")
        for i, (inputs, targets) in enumerate(train_loader):
            print(f"批次 {i}:")
            print(f"  输入形状: {inputs.shape}")
            print(f"  目标形状: {targets.shape}")
            if i >= 2:  # 只测试前几个批次
                break

        print("数据加载器测试完成！")

    except Exception as e:
        print(f"测试失败: {e}")
        print("请确保数据文件存在且格式正确")
