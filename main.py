"""SEAF ocean anomaly forecasting entry point."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def train_model(config_path: Optional[str] = None):
    try:
        print("开始训练模型...")
        command = [sys.executable, str(Path(__file__).resolve().parent / 'train.py')]
        if config_path:
            command.extend(['--config', config_path])
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f'train.py exited with code {completed.returncode}')
    except Exception as e:
        print(f"训练失败: {e}")
        return False
    return True


def test_data_loading(config_path: Optional[str] = None):
    try:
        from data_loader import get_data_info, create_data_loaders
        from config import DEFAULT_CONFIG, load_config, merge_configs, validate_config

        config = DEFAULT_CONFIG.copy()
        if config_path:
            config = merge_configs(load_config(config_path), config)
        validate_config(config)
        data_path = config.get('data_path', 'Data/FullData_preprocessed.nc')

        if not os.path.exists(data_path):
            print(f"数据文件不存在: {data_path}")
            return False

        print("测试数据加载功能...")
        print("1. 获取数据信息...")
        data_info = get_data_info(data_path)
        print("数据信息:")
        for key, value in data_info.items():
            print(f"  {key}: {value}")

        print("\n2. 创建数据加载器...")
        train_loader, val_loader, test_loader = create_data_loaders(
            data_path, config, batch_size=config['batch_size'], num_workers=0
        )

        print("\n3. 测试数据批次...")
        for i, batch in enumerate(train_loader):
            inputs, targets = batch[:2]
            print(f"批次 {i}:")
            print(f"  输入形状: {inputs.shape}")
            print(f"  目标形状: {targets.shape}")
            if i >= 1:
                break

        print("数据加载测试成功！")
        for loader in (train_loader, val_loader, test_loader):
            source = getattr(getattr(loader, 'dataset', None), 'dataset', None)
            if source is not None:
                source.close()
        return True

    except Exception as e:
        print(f"数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model(config_path: Optional[str] = None):
    try:
        from model_factory import create_ocean_model
        from data_loader import OceanDataset
        from config import DEFAULT_CONFIG, load_config, merge_configs, validate_config
        import torch
        from torch.utils.data import DataLoader

        print("测试模型结构...")
        config = DEFAULT_CONFIG.copy()
        if config_path:
            config = merge_configs(load_config(config_path), config)
        validate_config(config)

        data_path = config.get('data_path', 'Data/FullData_preprocessed.nc')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"找不到数据文件: {data_path}")

        train_dataset = OceanDataset(
            data_path, config, mode='train',
            train_ratio=config.get('train_ratio', 0.6),
            val_ratio=config.get('val_ratio', 0.2)
        )
        sample_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
        sample_input, sample_target = next(iter(sample_loader))
        actual_input_channels = sample_input.shape[2]
        actual_output_channels = sample_target.shape[2]
        height, width = sample_input.shape[-2], sample_input.shape[-1]

        config['actual_input_channels'] = actual_input_channels
        config['actual_output_channels'] = actual_output_channels
        config['input_channel_slices'] = {
            name: [value.start, value.stop]
            for name, value in train_dataset.input_channel_slices.items()
        }
        config['target_channel_slices'] = {
            name: [value.start, value.stop]
            for name, value in train_dataset.target_channel_slices.items()
        }

        model = create_ocean_model(config)
        batch_size = 2
        seq_len = config['sequence_length']
        test_input = torch.randn(batch_size, seq_len, actual_input_channels, height, width)
        print(f"测试输入形状: {test_input.shape}")

        model.eval()
        with torch.no_grad():
            output = model(test_input)

        print(f"模型输出形状: {output.shape}")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print("模型结构测试成功！")
        train_dataset.dataset.close()
        return True

    except Exception as e:
        print(f"模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="SEAF 海洋异常预测模型")
    parser.add_argument('--mode', type=str, choices=['train', 'test_data', 'test_model'],
                       default='test_data', help='运行模式')
    parser.add_argument('--config', type=str, help='配置文件路径（可选）')

    args = parser.parse_args()

    print("=" * 60)
    print("SEAF 海洋异常预测模型 - 温度盐度多步预报")
    print("=" * 60)

    if args.mode == 'train':
        print("模式: 训练模型")
        success = train_model(args.config)
    elif args.mode == 'test_data':
        print("模式: 测试数据加载")
        success = test_data_loading(args.config)
    elif args.mode == 'test_model':
        print("模式: 测试模型结构")
        success = test_model(args.config)
    else:
        print(f"未知模式: {args.mode}")
        success = False

    print("=" * 60)
    if success:
        print("执行成功！")
    else:
        print("执行失败！")
    print("=" * 60)


if __name__ == "__main__":
    main()
