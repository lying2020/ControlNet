#!/usr/bin/env python3
"""
测试新的日志目录结构
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config_loader import load_training_config
from train import create_log_dirs


def test_new_log_structure():
    """测试新的日志目录结构"""
    
    print("测试新的日志目录结构...")
    
    # 加载配置
    config = load_training_config("./configs/train/sd15_config.yaml", "./configs/datasets/dataset_config_raining.yaml")
    dataset_config = config['dataset']
    training_config = config['training']

    print("原始配置:")
    print(f"  log_dir: {training_config.get('log_dir')}")
    print(f"  image_log_dir: {training_config.get('image_log_dir')}")
    
    # 测试创建日志目录
    updated_config = create_log_dirs('sd15', 'raining', training_config, dataset_config)
    
    print("\n更新后的配置:")
    print(f"  experiment_dir: {updated_config.get('experiment_dir')}")
    print(f"  log_dir: {updated_config.get('log_dir')}")
    print(f"  image_log_dir: {updated_config.get('image_log_dir')}")
    print(f"  experiment_name: {updated_config.get('experiment_name')}")
    
    # 检查目录是否创建成功
    experiment_dir = updated_config.get('experiment_dir')
    log_dir = updated_config.get('log_dir')
    image_log_dir = updated_config.get('image_log_dir')
    
    print(f"\n检查目录创建:")
    print(f"  实验目录存在: {os.path.exists(experiment_dir)}")
    print(f"  日志目录存在: {os.path.exists(log_dir)}")
    print(f"  图像日志目录存在: {os.path.exists(image_log_dir)}")
    
    # 检查实验信息文件
    info_file = os.path.join(experiment_dir, 'experiment_info.json')
    print(f"  实验信息文件存在: {os.path.exists(info_file)}")
    
    if os.path.exists(info_file):
        import json
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
        print(f"  实验信息: {info}")
    
    # 显示目录结构
    print(f"\n目录结构:")
    if os.path.exists(experiment_dir):
        for root, dirs, files in os.walk(experiment_dir):
            level = root.replace(experiment_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for d in dirs:
                print(f"{subindent}{d}/")
            for f in files:
                print(f"{subindent}{f}")
    
    print("\n✓ 新的日志目录结构测试完成！")


if __name__ == "__main__":
    test_new_log_structure() 