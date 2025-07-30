#!/usr/bin/env python3
"""
测试数据集加载是否正常
"""

import sys
import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from my_dataset import MyDataset
from configs.config_loader import load_training_config

sys.path.append(os.path.dirname(os.path.abspath(__file__)))



def main():
    parser = argparse.ArgumentParser(description='测试数据集加载和结构')
    parser.add_argument('--sd_version', type=str, default='sd15', choices=['sd15', 'sd21'],
                       help='Stable Diffusion版本 (sd15 或 sd21)')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_training_config("./configs/train/sd15_config.yaml", "./configs/dataset_config_raining.yaml")
    dataset_config = config['dataset']

    print("=" * 60)
    print(f"ControlNet Dataloader 详细结构分析 - {args.sd_version.upper()}")
    print("=" * 60)
    
    print(f"数据集配置:")
    print(f"  数据根目录: {dataset_config['data_root']}")
    print(f"  JSON文件: {dataset_config['json_file']}")
    print(f"  源图像归一化: {dataset_config['source_normalization']}")
    print(f"  目标图像归一化: {dataset_config['target_normalization']}")
    print()
    
    dataset = MyDataset(config=dataset_config)
    print(f"数据集大小: {len(dataset)}")

    dataloader = DataLoader(
        dataset, 
        num_workers=0, 
        batch_size=args.batch_size, 
        shuffle=True
    )

    print(f"\n批次大小: {args.batch_size}")
    print("=" * 60)

    # 获取第一个批次
    for batch in dataloader:
        print("DataLoader 输出的批次结构:")
        print("-" * 40)
        
        for key, value in batch.items():
            print(f"\n键名: '{key}'")
            print(f"类型: {type(value)}")
            
            if isinstance(value, torch.Tensor):
                print(f"张量形状: {value.shape}")
                print(f"张量类型: {value.dtype}")
                print(f"数值范围: [{value.min().item():.3f}, {value.max().item():.3f}]")
                print(f"设备: {value.device}")
                
                # 如果是图像数据，显示更多信息
                if len(value.shape) == 4:  # [batch, height, width, channels]
                    print(f"批次维度: {value.shape[0]} 个样本")
                    print(f"图像尺寸: {value.shape[1]}x{value.shape[2]}")
                    print(f"通道数: {value.shape[3]}")
                    
            elif isinstance(value, list):
                print(f"列表长度: {len(value)}")
                print("列表内容:")
                for i, item in enumerate(value):
                    print(f"  [{i}]: {item}")
                    
            else:
                print(f"值: {value}")
        
        break

    print("\n" + "=" * 60)
    print("单个样本详细分析")
    print("=" * 60)

    # 分析单个样本
    item = dataset[1234]
    print(f"样本索引: 1234")

    print("\n单个样本结构:")
    print("-" * 40)

    for key, value in item.items():
        print(f"\n键名: '{key}'")
        print(f"类型: {type(value)}")
        
        if isinstance(value, np.ndarray):
            print(f"数组形状: {value.shape}")
            print(f"数组类型: {value.dtype}")
            print(f"数值范围: [{value.min():.3f}, {value.max():.3f}]")
            
            if len(value.shape) == 3:  # [height, width, channels]
                print(f"图像尺寸: {value.shape[0]}x{value.shape[1]}")
                print(f"通道数: {value.shape[2]}")
                
        elif isinstance(value, str):
            print(f"字符串长度: {len(value)}")
            print(f"内容: '{value}'")
            
        else:
            print(f"值: {value}")

    print("\n" + "=" * 60)
    print("数据用途说明")
    print("=" * 60)

    print("""
ControlNet 数据说明:

1. 'jpg' (目标图像):
   - 形状: [batch_size, height, width, 3] 或 [height, width, 3]
   - 范围: [-1, 1] (归一化到扩散模型所需范围)
   - 用途: 作为扩散模型的目标图像，模型学习生成这样的图像

2. 'txt' (文本提示):
   - 类型: str 或 list[str]
   - 内容: 描述目标图像的文本
   - 用途: 通过CLIP编码器转换为文本特征，作为条件控制

3. 'hint' (控制信号):
   - 形状: [batch_size, height, width, 3] 或 [height, width, 3]
   - 范围: [0, 1] (归一化到控制信号范围)
   - 用途: 作为ControlNet的输入，提供额外的控制信息

在训练过程中:
- 'jpg' 被编码到潜在空间作为扩散目标
- 'txt' 被CLIP编码为文本条件
- 'hint' 被ControlNet处理为控制特征
- 所有条件在扩散过程中融合，指导图像生成
""")

    print("=" * 60)


def test_dataset():
    """测试数据集加载"""
    
    print("开始测试数据集加载...")
    
    # 加载配置
    config = load_training_config("./configs/train/sd15_config.yaml", "./configs/dataset_config_raining.yaml")
    dataset_config = config['dataset']
    
    print(f"数据集配置: {dataset_config}")
    
    try:
        # 创建数据集
        dataset = MyDataset(config=dataset_config)
        print(f"数据集创建成功，包含 {len(dataset)} 个样本")
        
        # 测试加载第一个样本
        print("\n测试加载第一个样本...")
        sample = dataset[0]
        
        print(f"样本键: {list(sample.keys())}")
        print(f"jpg shape: {sample['jpg'].shape}")
        print(f"hint shape: {sample['hint'].shape}")
        print(f"txt: {sample['txt']}")
        
        # 测试加载多个样本
        print("\n测试加载多个样本...")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"样本 {i}: jpg={sample['jpg'].shape}, hint={sample['hint'].shape}")
        
        print("\n✓ 数据集测试通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_dataset()
    main()