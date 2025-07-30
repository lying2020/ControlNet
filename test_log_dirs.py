#!/usr/bin/env python3
"""
测试日志目录创建功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config_loader import load_training_config


def test_log_dirs():
    """测试日志目录创建"""
    
    print("测试日志目录创建功能...")
    
    # 加载配置
    config = load_training_config("./configs/train/sd15_config.yaml", "./configs/dataset_config_raining.yaml")
    dataset_config = config['dataset']
    training_config = config['training']
    
    print("原始配置:")
    print(f"  log_dir: {training_config.get('log_dir')}")
    print(f"  image_log_dir: {training_config.get('image_log_dir')}")
    
    # 导入并测试create_log_dirs函数
    from train import create_log_dirs
    
    # 测试创建日志目录
    updated_config = create_log_dirs('sd15', 'raining', training_config, dataset_config)
    
    print("\n更新后的配置:")
    print(f"  log_dir: {updated_config.get('log_dir')}")
    print(f"  image_log_dir: {updated_config.get('image_log_dir')}")
    print(f"  experiment_name: {updated_config.get('experiment_name')}")
    
    # 检查目录是否创建成功
    log_dir = updated_config.get('log_dir')
    image_log_dir = updated_config.get('image_log_dir')
    
    print(f"\n检查目录创建:")
    print(f"  日志目录存在: {os.path.exists(log_dir)}")
    print(f"  图像日志目录存在: {os.path.exists(image_log_dir)}")
    
    # 检查实验信息文件
    info_file = os.path.join(log_dir, 'experiment_info.json')
    print(f"  实验信息文件存在: {os.path.exists(info_file)}")
    
    if os.path.exists(info_file):
        import json
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
        print(f"  实验信息: {info}")
    
    print("\n✓ 日志目录创建测试完成！")


if __name__ == "__main__":
    test_log_dirs() 