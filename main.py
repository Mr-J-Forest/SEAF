"""
海洋数据ConvLSTM模型主运行脚本
提供统一的接口来训练模型和进行预测
"""

import argparse
import os
import sys
from typing import Optional

def train_model(config_path: Optional[str] = None):
    """
    训练模型
    
    Args:
        config_path: 配置文件路径（可选）
    """
    try:
        from train import main as train_main
        print("开始训练模型...")
        train_main()
    except Exception as e:
        print(f"训练失败: {e}")
        return False
    return True

def test_data_loading():
    """
    测试数据加载功能
    """
    try:
        from data_loader import get_data_info, create_data_loaders
        from convlstm_model import DEFAULT_CONFIG
        
        data_path = "Data/FullData_preprocessed.nc"
        
        if not os.path.exists(data_path):
            print(f"数据文件不存在: {data_path}")
            return False
        
        print("测试数据加载功能...")
        
        # 获取数据信息
        print("1. 获取数据信息...")
        data_info = get_data_info(data_path)
        print("数据信息:")
        for key, value in data_info.items():
            print(f"  {key}: {value}")
        
        # 创建数据加载器
        print("\n2. 创建数据加载器...")
        config = DEFAULT_CONFIG.copy()
        train_loader, val_loader, test_loader = create_data_loaders(
            data_path, config, batch_size=2, num_workers=0
        )
        
        # 测试一个批次
        print("\n3. 测试数据批次...")
        for i, (inputs, targets) in enumerate(train_loader):
            print(f"批次 {i}:")
            print(f"  输入形状: {inputs.shape}")
            print(f"  目标形状: {targets.shape}")
            if i >= 1:  # 只测试前两个批次
                break
        
        print("数据加载测试成功！")
        return True
        
    except Exception as e:
        print(f"数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model():
    """
    测试模型结构
    """
    try:
        from convlstm_model import create_ocean_model, DEFAULT_CONFIG
        import torch
        
        print("测试模型结构...")
        
        config = DEFAULT_CONFIG.copy()
        model = create_ocean_model(config)
        
        # 创建测试数据
        batch_size = 2
        seq_len = config['sequence_length']
        channels = len(config['input_variables'])
        height, width = 21, 32  # 根据数据范围计算
        
        test_input = torch.randn(batch_size, seq_len, channels, height, width)
        print(f"测试输入形状: {test_input.shape}")
        
        # 前向传播测试
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

def predict_with_model(model_path: str, sample_idx: int = 0):
    """
    使用训练好的模型进行预测
    
    Args:
        model_path: 模型文件路径
        sample_idx: 样本索引
    """
    try:
        from predict_visualize import OceanPredictor, OceanVisualizer
        from data_loader import OceanDataset
        from convlstm_model import DEFAULT_CONFIG
        
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return False
        
        print("开始模型预测...")
        
        # 创建预测器
        predictor = OceanPredictor(model_path)
        
        # 创建测试数据集
        data_path = "Data/FullData_preprocessed.nc"
        config = DEFAULT_CONFIG.copy()
        test_dataset = OceanDataset(data_path, config, mode='test')
        
        # 进行预测
        input_data, true_data, pred_data = predictor.predict_from_dataset(
            test_dataset, sample_idx, future_steps=5
        )
        
        print(f"输入数据形状: {input_data.shape}")
        print(f"真实数据形状: {true_data.shape}")
        print(f"预测数据形状: {pred_data.shape}")
        
        # 创建可视化器
        visualizer = OceanVisualizer(
            lon_range=[130.5, 162.5],
            lat_range=[6.5, 27.5]
        )
        
        # 可视化结果
        os.makedirs("prediction_results", exist_ok=True)
        
        visualizer.plot_prediction_comparison(
            true_data.numpy(), pred_data.numpy(), 
            variable_name="温度", time_step=0,
            save_dir="prediction_results"
        )
        
        print("预测和可视化完成！结果保存在 prediction_results 目录中")
        return True
        
    except Exception as e:
        print(f"预测失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="海洋数据ConvLSTM模型")
    parser.add_argument('--mode', type=str, choices=['train', 'test_data', 'test_model', 'predict'], 
                       default='test_data', help='运行模式')
    parser.add_argument('--model_path', type=str, help='模型文件路径（预测模式使用）')
    parser.add_argument('--sample_idx', type=int, default=0, help='预测样本索引')
    parser.add_argument('--config', type=str, help='配置文件路径（可选）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("海洋数据ConvLSTM模型 - 温度和盐度反演")
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
        
    elif args.mode == 'predict':
        print("模式: 模型预测")
        if not args.model_path:
            print("错误: 预测模式需要指定 --model_path")
            success = False
        else:
            success = predict_with_model(args.model_path, args.sample_idx)
    
    print("=" * 60)
    if success:
        print("执行成功！")
    else:
        print("执行失败！")
    print("=" * 60)

if __name__ == "__main__":
    main()