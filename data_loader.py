"""
海洋数据加载器
处理NetCDF格式的海洋数据，用于ConvLSTM模型训练和预测
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


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
        self.climatology_period = max(1, int(config.get('climatology_period', 12)))
        self.anomaly_variables = set(config.get('anomaly_variables', self.target_variables))
        self.include_climatology_features = config.get('include_climatology_features', False)
        self.climatology_feature_variables = list(
            config.get('climatology_feature_variables', self.target_variables)
        )
        # 数据范围设置（优先使用配置中的范围）
        self.lon_range = list(config.get('lon_range', [130.5, 162.5]))
        self.lat_range = list(config.get('lat_range', [6.5, 27.5]))
        self.depth_range = list(config.get('depth_range', [0.0, 5.0]))
        
        # 滑动窗口参数
        self.sliding_enabled = config.get('sliding_enabled', False) # 是否启用滑动数据增强
        self.ocean_threshold = config.get('ocean_threshold', 1.0)   # 海洋面积占比阈值（1.0=仅纯海洋）
        self.lon_step = config.get('lon_step', 2.0)                # 经纬度滑动步长（兜底）
        self.sliding_regions = []        # 存储所有有效的100%海洋区域

        # 推理阶段可覆盖步长（用于 dense overlap-tile prediction）
        self.override_stride_lon = override_stride_lon
        self.override_stride_lat = override_stride_lat

        # 分模式滑动步长
        self._resolve_stride(config)

        # 实际可用的变量（根据NetCDF文件描述）
        self.available_variables = ['TEMP', 'SALT', 'PTEMP', 'PDEN', 'ADDEP', 'SPICE', 'SSHA', 'UWND', 'VWND', 'SSW']
        self.coord_variables = ['LONGITUDE', 'LATITUDE', 'LEVEL', 'TIME']
        
        # 加载和预处理数据
        self._load_data()
        self._find_sliding_regions()  # 查找有效的滑动区域
        self._preprocess_data()
        self._split_data()
        self._create_sequences()
        self.input_channel_slices = {}
        self.target_channel_slices = {}
        
    def _load_data(self):
        """
        加载NetCDF数据
        """
        print(f"正在加载数据文件: {self.data_path}")
        
        # 使用xarray加载NetCDF文件
        self.dataset = xr.open_dataset(self.data_path)
        
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

        # 2D 滑动: 经度 + 纬度双向滑动
        current_lon = min_lon
        while current_lon + window_lon_size <= max_lon:
            current_lat = min_lat
            while current_lat + window_lat_size <= max_lat:
                new_lon_range = [current_lon, current_lon + window_lon_size]
                new_lat_range = [current_lat, current_lat + window_lat_size]

                if self._check_ocean_coverage(self.dataset, new_lon_range, new_lat_range):
                    if not self._region_exists(new_lon_range, new_lat_range):
                        self.sliding_regions.append({
                            'lon_range': new_lon_range,
                            'lat_range': new_lat_range,
                            'region_type': 'sliding'
                        })

                current_lat += self.stride_lat

            current_lon += self.stride_lon

        print(f"找到 {len(self.sliding_regions)} 个100%海洋覆盖的区域")
        for i, region in enumerate(self.sliding_regions):
            print(
                f"  区域 {i+1}: 经度 [{region['lon_range'][0]:.1f}, {region['lon_range'][1]:.1f}], "
                f"纬度 [{region['lat_range'][0]:.1f}, {region['lat_range'][1]:.1f}]"
            )

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
                LEVEL=slice(0, 5.0)  # 只检查表层
            )
            
            # 使用温度数据来判断海洋覆盖（有数据的地方认为是海洋）
            if 'TEMP' in region_data.data_vars:
                temp_data = region_data.TEMP.isel(TIME=0, LEVEL=0).values  # 第一个时间步的表层数据
                
                # 计算有效数据比例（非NaN的部分认为是海洋）
                total_points = temp_data.size
                valid_points = np.sum(~np.isnan(temp_data))
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
                if np.isnan(data).any():
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

    @staticmethod
    def _climatology_feature_name(variable: str) -> str:
        return f"CLIMATOLOGY_{variable}"

    def _compute_region_climatology(self, region: Dict):
        """基于训练时间段为单个窗口计算月气候态和 anomaly 源数据。"""
        region['climatology'] = {}
        region['anomaly_data'] = {}
        if not self.enable_climatology_anomaly:
            return

        train_end_idx = int(len(self.times) * self.train_ratio)
        train_periods = self.time_period_indices[:train_end_idx]
        variables = set(self.anomaly_variables)
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

            if var in self.anomaly_variables:
                periods = self.time_period_indices[:data.shape[0]]
                region['anomaly_data'][var] = (data - climatology[periods]).astype(np.float32)

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
        
        # 如果提供了scalers，直接使用
        if self.provided_scalers is not None:
            print("使用提供的标准化参数 (防止数据泄露)...")
            self.scalers = self.provided_scalers
        else:
            print("计算全局标准化参数 (仅使用训练集数据)...")
            # 为每个变量收集所有区域的数据用于计算全局统计量
            # 注意：只使用训练集时间段的数据来计算scaler，防止数据泄露
            global_data = {}
            
            # 计算训练集结束的时间索引
            total_time_steps = len(self.times)
            train_end_idx = int(total_time_steps * self.train_ratio)
            print(f"标准化参数计算范围: 时间步 0 - {train_end_idx}")
            
            for var in all_vars:
                var_data_list = []
                for region in self.all_regions_data:
                    source_data = self._get_model_source_data(region, var)
                    if source_data is not None:
                        # 只取训练时间段的数据
                        train_data = source_data[:train_end_idx]
                        var_data_list.append(train_data)
                
                if var_data_list:
                    # 合并所有区域的该变量数据
                    global_data[var] = np.concatenate(var_data_list, axis=0)

            if self.include_climatology_features:
                for var in self.climatology_feature_variables:
                    feature_name = self._climatology_feature_name(var)
                    var_data_list = []
                    for region in self.all_regions_data:
                        clim_series = self._build_climatology_series(region, var)
                        if clim_series is not None:
                            var_data_list.append(clim_series[:train_end_idx])
                    if var_data_list:
                        global_data[feature_name] = np.concatenate(var_data_list, axis=0)
            
            # 计算全局标准化参数
            self.scalers = {}
            for var, data in global_data.items():
                original_shape = data.shape
                data_2d = data.reshape(-1, 1)
                
                scaler = StandardScaler()
                scaler.fit(data_2d)
                self.scalers[var] = scaler
                
                print(f"变量 {var} 全局标准化参数: 均值={scaler.mean_[0]:.4f}, 标准差={scaler.scale_[0]:.4f}")
        
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

        sample_array = next(iter(region['data'].values()))
        time_steps = sample_array.shape[0]
        lat_size = len(coords.get('lats', self.lats))
        lon_size = len(coords.get('lons', self.lons))

        if self.config.get('enable_positional_encoding', False):
            spatial_features = self._build_spatial_encoding(coords, time_steps, lat_size, lon_size)
            if spatial_features is not None:
                normalized['SPATIAL_ENCODING'] = spatial_features.astype(np.float32)

            depth_features = self._build_depth_encoding(coords, time_steps, lat_size, lon_size)
            if depth_features is not None:
                normalized['DEPTH_ENCODING'] = depth_features.astype(np.float32)

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

        div_terms = np.power(10000.0, np.arange(num_freq, dtype=np.float32) / max(num_freq, 1))
        features = []
        for div in div_terms:
            features.append(np.sin(lon_grid / div))
            features.append(np.cos(lon_grid / div))
        for div in div_terms:
            features.append(np.sin(lat_grid / div))
            features.append(np.cos(lat_grid / div))

        if not features:
            return None

        spatial_encoding = np.stack(features, axis=0)
        spatial_encoding = np.repeat(spatial_encoding[np.newaxis, ...], time_steps, axis=0)
        return spatial_encoding

    def _build_depth_encoding(self, coords: Dict, time_steps: int, lat_size: int, lon_size: int) -> Optional[np.ndarray]:
        num_freq = int(self.config.get('depth_encoding_frequencies', 0))
        levels = coords.get('levels', self.levels)
        if num_freq <= 0 or levels is None or len(levels) == 0:
            return None

        levels = np.array(levels, dtype=np.float32)
        div_terms = np.power(10000.0, np.arange(num_freq, dtype=np.float32) / max(num_freq, 1))
        feature_maps = []
        for depth in levels:
            for div in div_terms:
                feature_maps.append(np.full((lat_size, lon_size), np.sin(depth / div), dtype=np.float32))
                feature_maps.append(np.full((lat_size, lon_size), np.cos(depth / div), dtype=np.float32))

        if not feature_maps:
            return None

        depth_encoding = np.stack(feature_maps, axis=0)
        depth_encoding = np.repeat(depth_encoding[np.newaxis, ...], time_steps, axis=0)
        return depth_encoding

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

        # 傅里叶时间编码
        if num_freq > 0:
            time_channels = np.zeros((time_steps, num_freq * 2, lat_size, lon_size), dtype=np.float32)
            months = self._extract_period_indices(times, period).astype(np.float32)

            for idx, freq in enumerate(range(1, num_freq + 1)):
                angle = 2 * np.pi * months * freq / period
                time_channels[:, 2 * idx, :, :] = np.sin(angle)[:, None, None]
                time_channels[:, 2 * idx + 1, :, :] = np.cos(angle)[:, None, None]
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
                trend_channel = np.broadcast_to(
                    trend_values[:, None, None],
                    (time_steps, lat_size, lon_size)
                )
                feature_maps.append(trend_channel[:, None, :, :])

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
        # 获取有效数据的索引
        valid_mask = ~np.isnan(data)
        
        if valid_mask.sum() == 0:
            # 如果全部是缺失值，用0填充
            print(f"警告: 数据全部为缺失值，使用0填充")
            return np.zeros_like(data)
        
        train_end_idx = int(len(self.times) * self.train_ratio)
        train_slice = data[:train_end_idx]
        global_mean = np.nanmean(train_slice)
        if np.isnan(global_mean):
            global_mean = np.nanmean(data)
        if np.isnan(global_mean):
            print(f"警告: 变量 {variable} 无法计算有效均值，使用0填充")
            global_mean = 0.0
        
        # 根据数据维度进行不同的处理
        if data.ndim == 4:  # 4D数据: (time, level, lat, lon)
            for t in range(data.shape[0]):
                for d in range(data.shape[1]):
                    slice_data = data[t, d, :, :]
                    if np.isnan(slice_data).any():
                        # 首先尝试用该切片的有效值均值填充
                        slice_mean = np.nanmean(slice_data)
                        if not np.isnan(slice_mean):
                            slice_data[np.isnan(slice_data)] = slice_mean
                        else:
                            # 如果该切片全为NaN，使用全局均值
                            slice_data[np.isnan(slice_data)] = global_mean
                        data[t, d, :, :] = slice_data
        elif data.ndim == 3:  # 3D数据: (time, lat, lon)
            for t in range(data.shape[0]):
                slice_data = data[t, :, :]
                if np.isnan(slice_data).any():
                    # 首先尝试用该时间步的有效值均值填充
                    slice_mean = np.nanmean(slice_data)
                    if not np.isnan(slice_mean):
                        slice_data[np.isnan(slice_data)] = slice_mean
                    else:
                        # 如果该时间步全为NaN，使用全局均值
                        slice_data[np.isnan(slice_data)] = global_mean
                    data[t, :, :] = slice_data
        
        # 最终检查：如果还有NaN，用全局均值填充
        if np.isnan(data).any():
            print(f"警告: 插值后仍有NaN值，使用全局均值 {global_mean:.6f} 填充")
            data[np.isnan(data)] = global_mean
        
        return data
    
    def _handle_surface_variables(self):
        """
        处理表面变量（SSHA, UWND, VWND）的深度扩展
        这些变量只有时间、纬度、经度三个维度，需要扩展到四个维度以匹配其他变量
        """
        surface_vars = ['SSHA', 'UWND', 'VWND', 'SSW']
        
        # 获取深度层数（从其他4D变量中获取）
        depth_levels = None
        for var in self.data_arrays:
            if self.data_arrays[var].ndim == 4:
                depth_levels = self.data_arrays[var].shape[1]
                break
        
        if depth_levels is None:
            print("无法确定深度层数，使用默认值27")
            depth_levels = 27
        
        for var in surface_vars:
            if var in self.data_arrays:
                data = self.data_arrays[var]
                if data.ndim == 3:  # (time, lat, lon)
                    # 扩展到4D: (time, level, lat, lon)
                    time_steps, lat_size, lon_size = data.shape
                    expanded_data = np.zeros((time_steps, depth_levels, lat_size, lon_size))
                    
                    # 将表面数据复制到所有深度层
                    for d in range(depth_levels):
                        expanded_data[:, d, :, :] = data
                    
                    self.data_arrays[var] = expanded_data
                    print(f"变量 {var} 已从3D扩展到4D，所有深度层使用表面值")
                else:
                    print(f"变量 {var} 已经是4D，无需扩展")
    
    def _normalize_data(self):
        """
        数据标准化
        """
        print("正在标准化数据...")
        
        self.scalers = {}
        self.normalized_data = {}
        
        for var in self.data_arrays.keys():
            data = self.data_arrays[var]
            
            # 将数据重塑为2D进行标准化
            original_shape = data.shape
            data_2d = data.reshape(-1, 1)
            
            # 创建标准化器
            scaler = StandardScaler()
            data_normalized = scaler.fit_transform(data_2d)
            
            # 恢复原始形状
            self.normalized_data[var] = data_normalized.reshape(original_shape)
            self.scalers[var] = scaler
            
            print(f"变量 {var} 标准化完成，均值: {scaler.mean_[0]:.4f}, 标准差: {scaler.scale_[0]:.4f}")
    
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
        创建时序序列（包括所有区域的数据）
        """
        self.sequences = []
        
        # 获取该模式对应的时间范围
        if not hasattr(self, 'time_indices') or len(self.time_indices) == 0:
            print(f"警告: {self.mode}模式的时间索引为空")
            return
        
        min_time_idx = min(self.time_indices)
        max_time_idx = max(self.time_indices)
        
        # 确保有足够的时间步来创建序列
        required_length = self.sequence_length + self.prediction_length
        available_length = max_time_idx - min_time_idx + 1
        
        if available_length < required_length:
            print(f"警告: {self.mode}集时间步不足。需要{required_length}，可用{available_length}")
            return
        
        # 为每个区域创建序列（train/val/test 均包含所有100%海洋窗口）
        for region_idx, region_info in enumerate(self.all_regions_data):
            region_type = region_info['region_type']

            region_sequences = []
            for t in range(min_time_idx, max_time_idx - required_length + 2):
                if (t + self.sequence_length + self.prediction_length - 1) <= max_time_idx:
                    region_sequences.append((t, region_idx))

            self.sequences.extend(region_sequences)

            print(f"区域 {region_idx} ({region_type}): {len(region_sequences)} 个序列")
        
        print(f"{self.mode.upper()}集总序列数量: {len(self.sequences)}")
        if len(self.sequences) > 0:
            print(f"  时间范围: {min_time_idx} 到 {max_time_idx}")
        else:
            print(f"  警告: 无法创建序列！时间范围: {min_time_idx}-{max_time_idx}, 需要长度: {required_length}")
    
    def __len__(self) -> int:
        """
        返回数据集大小
        """
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            输入序列和目标序列的元组
        """
        # 解析序列信息：(时间起始索引, 区域索引)
        start_idx, region_idx = self.sequences[idx]
        
        # 获取对应区域的数据
        region_info = self.all_regions_data[region_idx]
        normalized_data = region_info['normalized_data']
        
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
            if self.config.get('enable_positional_encoding', False):
                if 'SPATIAL_ENCODING' in normalized_data:
                    valid_input_vars.append('SPATIAL_ENCODING')
                if 'DEPTH_ENCODING' in normalized_data:
                    valid_input_vars.append('DEPTH_ENCODING')
            if self.config.get('enable_time_encoding', False) and 'TIME_ENCODING' in normalized_data:
                valid_input_vars.append('TIME_ENCODING')
            self.actual_input_variables = valid_input_vars
        
        # 构建输入序列
        input_sequence = []
        for var in self.actual_input_variables:
            if var in normalized_data:
                var_data = normalized_data[var][start_idx:start_idx + self.sequence_length]
                input_sequence.append(var_data)
        
        # 构建目标序列
        target_sequence = []
        for var in self.target_variables:
            if var in normalized_data:
                var_data = normalized_data[var][start_idx + self.sequence_length:start_idx + self.sequence_length + self.prediction_length]
                target_sequence.append(var_data)
        
        # 转换为numpy数组并调整维度
        # 数据形状: (seq_len, level, lat, lon) -> 需要转换为 (seq_len, lat, lon, level * channels)
        input_arrays = []
        input_channel_lengths = []
        for var_data in input_sequence:
            if var_data.ndim == 4:  # (seq_len, level, lat, lon)
                # 重排维度: (seq_len, level, lat, lon) -> (seq_len, lat, lon, level)
                var_data = var_data.transpose(0, 2, 3, 1)
            elif var_data.ndim == 3:  # 如果是3D数据，添加深度维度
                var_data = var_data[:, :, :, np.newaxis]
            input_arrays.append(var_data)
            input_channel_lengths.append(var_data.shape[-1])

        if not self.input_channel_slices:
            channel_offset = 0
            for var_name, channel_len in zip(self.actual_input_variables, input_channel_lengths):
                self.input_channel_slices[var_name] = slice(channel_offset, channel_offset + channel_len)
                channel_offset += channel_len
        
        target_arrays = []
        target_channel_lengths = []
        for var_data in target_sequence:
            if var_data.ndim == 4:  # (pred_len, level, lat, lon)
                # 重排维度: (pred_len, level, lat, lon) -> (pred_len, lat, lon, level)
                var_data = var_data.transpose(0, 2, 3, 1)
            elif var_data.ndim == 3:  # 如果是3D数据，添加深度维度
                var_data = var_data[:, :, :, np.newaxis]
            target_arrays.append(var_data)
            target_channel_lengths.append(var_data.shape[-1])

        if not self.target_channel_slices:
            channel_offset = 0
            for var_name, channel_len in zip(self.target_variables, target_channel_lengths):
                self.target_channel_slices[var_name] = slice(channel_offset, channel_offset + channel_len)
                channel_offset += channel_len
        
        # 在深度维度上连接不同变量
        input_seq = np.concatenate(
            [arr.astype(np.float32, copy=False) for arr in input_arrays],
            axis=-1
        )  # (seq_len, lat, lon, total_channels)
        target_seq = np.concatenate(
            [arr.astype(np.float32, copy=False) for arr in target_arrays],
            axis=-1
        )  # (pred_len, lat, lon, total_channels)
        
        # 转换为PyTorch张量并调整维度顺序 (seq_len, channels, height, width)
        input_tensor = torch.FloatTensor(input_seq).permute(0, 3, 1, 2)
        target_tensor = torch.FloatTensor(target_seq).permute(0, 3, 1, 2)

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
        ch_slice = self.target_channel_slices.get(variable)
        if ch_slice is None:
            raw_slices = self.config.get('target_channel_slices', {})
            raw_slice = raw_slices.get(variable) if isinstance(raw_slices, dict) else None
            if isinstance(raw_slice, (list, tuple)) and len(raw_slice) >= 2:
                ch_slice = slice(int(raw_slice[0]), int(raw_slice[1]))
        if ch_slice is not None:
            return slice(ch_slice.start or 0, ch_slice.stop or total_channels)

        channels_per_var = total_channels // max(1, len(self.target_variables))
        return slice(var_idx * channels_per_var, (var_idx + 1) * channels_per_var)

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

        restored = array.astype(np.float64, copy=True)
        sample_count, pred_steps, total_channels = restored.shape[:3]

        if sample_indices is None:
            sample_indices = list(range(sample_count))
        else:
            sample_indices = list(sample_indices)

        for var_idx, var_name in enumerate(self.target_variables):
            ch_slice = self._resolve_target_slice(var_name, var_idx, total_channels)
            if ch_slice.stop <= ch_slice.start:
                continue

            scaler = self.scalers.get(var_name)
            var_block = restored[:, :, ch_slice, :, :]
            if scaler is not None:
                original_shape = var_block.shape
                var_block = scaler.inverse_transform(var_block.reshape(-1, 1)).reshape(original_shape)

            if self._is_anomaly_variable(var_name):
                for out_idx, sample_idx in enumerate(sample_indices[:sample_count]):
                    if sample_idx >= len(self.sequences):
                        continue
                    start_idx, region_idx = self.sequences[int(sample_idx)]
                    target_start = start_idx + self.sequence_length
                    region = self.all_regions_data[region_idx]
                    clim = self._target_climatology_channels(
                        region,
                        var_name,
                        target_start,
                        pred_steps
                    )
                    if clim is not None and clim.shape == var_block[out_idx].shape:
                        var_block[out_idx] = var_block[out_idx] + clim

            restored[:, :, ch_slice, :, :] = var_block

        return restored[0] if squeeze_sample else restored
    
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
    
    # 使用训练集的scalers初始化验证集和测试集
    print("初始化验证集 (使用训练集标准化参数)...")
    val_dataset = OceanDataset(data_path, config, mode='val',
                              train_ratio=train_ratio, val_ratio=val_ratio,
                              scalers=scalers)
    
    print("初始化测试集 (使用训练集标准化参数)...")
    test_dataset = OceanDataset(data_path, config, mode='test',
                               train_ratio=train_ratio, val_ratio=val_ratio,
                               scalers=scalers)
    
    # 检查数据集大小
    print(f"数据集大小检查:")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=use_prefetch,
        persistent_workers=use_persistent,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=use_prefetch,
        persistent_workers=use_persistent,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
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
    from convlstm_model import DEFAULT_CONFIG
    
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
