"""
TSC-Fusion 海洋预测模型主运行脚本
提供统一的接口来训练模型和进行预测
"""

import argparse
import os
import sys
from typing import Optional


def train_model(config_path: Optional[str] = None):
    try:
        from train import main as train_main
        print("开始训练模型...")
        train_main()
    except Exception as e:
        print(f"训练失败: {e}")
        return False
    return True


def test_data_loading():
    try:
        from data_loader import get_data_info, create_data_loaders
        from config import DEFAULT_CONFIG

        data_path = DEFAULT_CONFIG.get('data_path', 'Data/FullData_preprocessed.nc')

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
        config = DEFAULT_CONFIG.copy()
        train_loader, val_loader, test_loader = create_data_loaders(
            data_path, config, batch_size=2, num_workers=0
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
        return True

    except Exception as e:
        print(f"数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    try:
        from convlstm_model import create_ocean_model
        from data_loader import OceanDataset
        from config import DEFAULT_CONFIG
        import torch
        from torch.utils.data import DataLoader

        print("测试模型结构...")
        config = DEFAULT_CONFIG.copy()

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
        return True

    except Exception as e:
        print(f"模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="TSC-Fusion 海洋预测模型")
    parser.add_argument('--mode', type=str, choices=['train', 'test_data', 'test_model'],
                       default='test_data', help='运行模式')
    parser.add_argument('--config', type=str, help='配置文件路径（可选）')

    args = parser.parse_args()

    print("=" * 60)
    print("TSC-Fusion 海洋预测模型 - 温度盐度反演")
    print("=" * 60)

    if args.mode == 'train':
        print("模式: 训练模型")
        success = train_model(args.config)
    elif args.mode == 'test_data':
        print("模式: 测试数据加载")
        success = test_data_loading()
    elif args.mode == 'test_model':
        print("模式: 测试模型结构")
        success = test_model()
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
