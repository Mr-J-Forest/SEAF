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
                 train_ratio: float = 0.6, val_ratio: float = 0.2):
        """
        初始化数据集
        
        Args:
            data_path: NetCDF数据文件路径
            config: 配置字典
            mode: 模式 ('train', 'val', 'test')
            train_ratio: 训练集比例（减少到0.6）
            val_ratio: 验证集比例
        """
        self.data_path = data_path
        self.config = config
        self.mode = mode
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio  # 现在是0.2，给测试集更多数据
        
        self.input_variables = config['input_variables']
        self.target_variables = config['target_variables']
        self.sequence_length = config['sequence_length']
        self.prediction_length = config['prediction_length']
        
        # 数据范围设置
        self.lon_range = [130.5, 162.5]  # 经度范围
        self.lat_range = [6.5, 27.5]     # 纬度范围
        self.depth_range = [0, 5.0]      # 只使用表层数据（0-5米）
        
        # 滑动窗口参数
        self.sliding_enabled = True       # 是否启用滑动数据增强
        self.ocean_threshold = 0.8        # 海洋面积占比阈值
        self.lon_step = 2.0              # 经度滑动步长
        self.sliding_regions = []        # 存储所有有效的滑动区域
        
        # 实际可用的变量（根据NetCDF文件描述）
        self.available_variables = ['TEMP', 'SALT', 'PTEMP', 'PDEN', 'ADDEP', 'SPICE', 'SSHA', 'UWND', 'VWND', 'SSW']
        self.coord_variables = ['LONGITUDE', 'LATITUDE', 'LEVEL', 'TIME']
        
        # 加载和预处理数据
        self._load_data()
        self._find_sliding_regions()  # 查找有效的滑动区域
        self._preprocess_data()
        self._split_data()
        self._create_sequences()
        
    def _load_data(self):
        """
        加载NetCDF数据
        """
        print(f"正在加载数据文件: {self.data_path}")
        
        # 使用xarray加载NetCDF文件
        self.dataset = xr.open_dataset(self.data_path)
        
        # 选择指定的经纬度和深度范围
        self.dataset = self.dataset.sel(
            LONGITUDE=slice(self.lon_range[0], self.lon_range[1]),
            LATITUDE=slice(self.lat_range[0], self.lat_range[1]),
            LEVEL=slice(self.depth_range[0], self.depth_range[1])
        )
        
        print(f"数据形状: {dict(self.dataset.dims)}")
        print(f"可用变量: {list(self.dataset.data_vars)}")
        
        # 获取坐标信息
        self.lons = self.dataset.LONGITUDE.values
        self.lats = self.dataset.LATITUDE.values
        self.levels = self.dataset.LEVEL.values
        self.times = self.dataset.TIME.values
        
        print(f"时间序列长度: {len(self.times)}")
        print(f"空间维度: 经度{len(self.lons)} x 纬度{len(self.lats)} x 深度{len(self.levels)}")
        
    def _find_sliding_regions(self):
        """
        查找有效的滑动区域（海洋面积占比>80%）
        """
        if not self.sliding_enabled or self.mode == 'test':
            return
            
        print("正在查找有效的滑动区域...")
        
        # 获取原始经度范围
        original_dataset = xr.open_dataset(self.data_path)
        all_lons = original_dataset.LONGITUDE.values
        
        # 计算滑动窗口大小
        window_lon_size = self.lon_range[1] - self.lon_range[0]  # 32度
        window_lat_size = self.lat_range[1] - self.lat_range[0]  # 21度
        
        # 在全球经度范围内滑动
        min_lon, max_lon = float(all_lons.min()), float(all_lons.max())
        
        # 计算可能的滑动位置
        current_lon = min_lon
        while current_lon + window_lon_size <= max_lon:
            new_lon_range = [current_lon, current_lon + window_lon_size]
            
            # 检查这个区域的海洋覆盖率
            if self._check_ocean_coverage(original_dataset, new_lon_range, self.lat_range):
                # 避免与原始区域完全重叠
                if not (abs(new_lon_range[0] - self.lon_range[0]) < 1.0 and 
                       abs(new_lon_range[1] - self.lon_range[1]) < 1.0):
                    self.sliding_regions.append({
                        'lon_range': new_lon_range,
                        'lat_range': self.lat_range.copy()
                    })
            
            current_lon += self.lon_step
        
        original_dataset.close()
        print(f"找到 {len(self.sliding_regions)} 个有效的滑动区域")
        for i, region in enumerate(self.sliding_regions):
            print(f"  区域 {i+1}: 经度 {region['lon_range'][0]:.1f}-{region['lon_range'][1]:.1f}")
    
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
                
                return ocean_ratio > self.ocean_threshold
            
        except Exception as e:
            print(f"检查区域 {lon_range} 时出错: {e}")
            return False
        
        return False
        
    def _preprocess_data(self):
        """
        预处理数据（包括原始区域和滑动区域）
        """
        print("正在预处理数据...")
        
        # 存储所有区域的数据
        self.all_regions_data = []
        
        # 处理原始区域
        print("处理原始区域...")
        original_data = self._process_single_region(self.dataset)
        if original_data:
            self.all_regions_data.append({
                'data': original_data,
                'region_type': 'original',
                'lon_range': self.lon_range,
                'lat_range': self.lat_range
            })
        
        # 处理滑动区域（仅用于训练）
        if self.sliding_enabled and self.mode == 'train' and len(self.sliding_regions) > 0:
            print("处理滑动区域...")
            original_dataset = xr.open_dataset(self.data_path)
            
            for i, region in enumerate(self.sliding_regions):
                print(f"  处理滑动区域 {i+1}/{len(self.sliding_regions)}")
                
                # 为每个滑动区域加载数据
                region_dataset = original_dataset.sel(
                    LONGITUDE=slice(region['lon_range'][0], region['lon_range'][1]),
                    LATITUDE=slice(region['lat_range'][0], region['lat_range'][1]),
                    LEVEL=slice(self.depth_range[0], self.depth_range[1])
                )
                
                region_data = self._process_single_region(region_dataset)
                if region_data:
                    self.all_regions_data.append({
                        'data': region_data,
                        'region_type': 'sliding',
                        'lon_range': region['lon_range'],
                        'lat_range': region['lat_range']
                    })
            
            original_dataset.close()
        
        print(f"总共处理了 {len(self.all_regions_data)} 个区域的数据")
        
        # 合并所有区域的数据用于标准化
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
                    data = self._fill_missing_values(data)
                
                region_data[var] = data
        
        return region_data if region_data else None
    
    def _merge_and_normalize_data(self):
        """
        合并所有区域数据并进行标准化
        """
        print("合并和标准化所有区域数据...")
        
        # 获取所有变量名
        all_vars = set()
        for region in self.all_regions_data:
            all_vars.update(region['data'].keys())
        
        # 为每个变量收集所有区域的数据用于计算全局统计量
        global_data = {}
        for var in all_vars:
            var_data_list = []
            for region in self.all_regions_data:
                if var in region['data']:
                    var_data_list.append(region['data'][var])
            
            if var_data_list:
                # 合并所有区域的该变量数据
                global_data[var] = np.concatenate(var_data_list, axis=0)
        
        # 计算全局标准化参数
        self.scalers = {}
        for var, data in global_data.items():
            original_shape = data.shape
            data_2d = data.reshape(-1, 1)
            
            scaler = StandardScaler()
            scaler.fit(data_2d)
            self.scalers[var] = scaler
            
            print(f"变量 {var} 全局标准化参数: 均值={scaler.mean_[0]:.4f}, 标准差={scaler.scale_[0]:.4f}")
        
        # 使用全局参数标准化各区域数据
        for region in self.all_regions_data:
            region['normalized_data'] = {}
            for var, data in region['data'].items():
                if var in self.scalers:
                    original_shape = data.shape
                    data_2d = data.reshape(-1, 1)
                    data_normalized = self.scalers[var].transform(data_2d)
                    region['normalized_data'][var] = data_normalized.reshape(original_shape)
        
        # 设置主要数据（用于向后兼容）
        if self.all_regions_data:
            main_region = self.all_regions_data[0]  # 原始区域
            self.data_arrays = main_region['data']
            self.normalized_data = main_region['normalized_data']
        
    def _fill_missing_values(self, data: np.ndarray) -> np.ndarray:
        """
        填充缺失值
        """
        # 获取有效数据的索引
        valid_mask = ~np.isnan(data)
        
        if valid_mask.sum() == 0:
            # 如果全部是缺失值，用0填充
            print(f"警告: 数据全部为缺失值，使用0填充")
            return np.zeros_like(data)
        
        # 计算全局有效值的均值作为备用填充值
        global_mean = np.nanmean(data)
        
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
        
        # 为每个区域创建序列
        for region_idx, region_info in enumerate(self.all_regions_data):
            region_type = region_info['region_type']
            
            # 测试集只使用原始区域
            if self.mode == 'test' and region_type != 'original':
                continue
            
            # 创建该区域的序列
            region_sequences = []
            for t in range(min_time_idx, max_time_idx - required_length + 2):
                if (t + self.sequence_length + self.prediction_length - 1) <= max_time_idx:
                    # 存储序列信息：(时间起始索引, 区域索引)
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
        for var_data in input_sequence:
            if var_data.ndim == 4:  # (seq_len, level, lat, lon)
                # 重排维度: (seq_len, level, lat, lon) -> (seq_len, lat, lon, level)
                var_data = var_data.transpose(0, 2, 3, 1)
            elif var_data.ndim == 3:  # 如果是3D数据，添加深度维度
                var_data = var_data[:, :, :, np.newaxis]
            input_arrays.append(var_data)
        
        target_arrays = []
        for var_data in target_sequence:
            if var_data.ndim == 4:  # (pred_len, level, lat, lon)
                # 重排维度: (pred_len, level, lat, lon) -> (pred_len, lat, lon, level)
                var_data = var_data.transpose(0, 2, 3, 1)
            elif var_data.ndim == 3:  # 如果是3D数据，添加深度维度
                var_data = var_data[:, :, :, np.newaxis]
            target_arrays.append(var_data)
        
        # 在深度维度上连接不同变量
        input_seq = np.concatenate(input_arrays, axis=-1)  # (seq_len, lat, lon, total_channels)
        target_seq = np.concatenate(target_arrays, axis=-1)  # (pred_len, lat, lon, total_channels)
        
        # 转换为PyTorch张量并调整维度顺序 (seq_len, channels, height, width)
        input_tensor = torch.FloatTensor(input_seq).permute(0, 3, 1, 2)
        target_tensor = torch.FloatTensor(target_seq).permute(0, 3, 1, 2)
        
        return input_tensor, target_tensor
    
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


def create_data_loaders(data_path: str, config: dict, batch_size: int = 4, 
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
    
    train_dataset = OceanDataset(data_path, config, mode='train', 
                                train_ratio=train_ratio, val_ratio=val_ratio)
    val_dataset = OceanDataset(data_path, config, mode='val',
                              train_ratio=train_ratio, val_ratio=val_ratio)
    test_dataset = OceanDataset(data_path, config, mode='test',
                               train_ratio=train_ratio, val_ratio=val_ratio)
    
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
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
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