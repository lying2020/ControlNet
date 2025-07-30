"""
配置加载器
用于加载和合并主配置与数据集配置
"""

import os
import sys
import yaml
from typing import Dict, Any

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(main_config: Dict[str, Any], dataset_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并主配置和数据集配置
    
    Args:
        main_config: 主配置字典
        dataset_config: 数据集配置字典
    
    Returns:
        合并后的配置字典
    """
    merged_config = main_config.copy()
    
    # 将数据集配置合并到主配置中
    if 'dataset' in dataset_config:
        merged_config['dataset'] = dataset_config['dataset']
    
    return merged_config


def load_training_config(main_config_path: str, dataset_config_path: str = None) -> Dict[str, Any]:
    """
    加载完整的训练配置（包括数据集配置）
    
    Args:
        main_config_path: 主配置文件路径
        dataset_config_path: 可选的数据集配置文件路径，如果为None则使用主配置中的默认路径
    
    Returns:
        完整的配置字典
    """
    # 加载主配置
    main_config = load_config(main_config_path)

    # 确定数据集配置文件路径
    if dataset_config_path is None:
        # 使用主配置中的默认路径
        if 'dataset' in main_config:
            dataset_config_path = main_config['dataset']
        else:
            print("警告: 主配置中没有指定数据集配置文件")
            return main_config

    # 确保路径是相对于主配置文件的
    if not os.path.isabs(dataset_config_path):
        main_config_dir = os.path.dirname(main_config_path)
        dataset_config_path = os.path.join(main_config_dir, '../..', dataset_config_path)
        dataset_config_path = os.path.normpath(dataset_config_path)

    # 加载数据集配置
    if os.path.exists(dataset_config_path):
        print(f"正在加载数据集配置: {dataset_config_path}")
        dataset_config = load_config(dataset_config_path)
        # 合并配置
        merged_config = merge_configs(main_config, dataset_config)
        return merged_config
    else:
        print(f"错误: 数据集配置文件不存在: {dataset_config_path}")
        return main_config


def save_config(config: Dict[str, Any], output_path: str):
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)


def print_config(config: Dict[str, Any], title: str = "配置信息"):
    """
    打印配置信息
    
    Args:
        config: 配置字典
        title: 标题
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    def print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{'  ' * indent}{key}:")
                print_dict(value, indent + 1)
            else:
                print(f"{'  ' * indent}{key}: {value}")
    
    print_dict(config)


def list_available_dataset_configs(configs_dir: str = "./configs") -> list:
    """
    列出可用的数据集配置文件
    
    Args:
        configs_dir: 配置文件目录
    
    Returns:
        可用配置文件列表
    """
    dataset_configs = []
    for file in os.listdir(configs_dir):
        if file.startswith("dataset_config_") and file.endswith(".yaml"):
            dataset_configs.append(file)
    return sorted(dataset_configs)


def get_dataset_config_path(config_name: str, configs_dir: str = "./configs") -> str:
    """
    根据配置名称获取完整路径
    
    Args:
        config_name: 配置名称（如 'test', 'high_res'）
        configs_dir: 配置文件目录
    
    Returns:
        完整的配置文件路径
    """
    if config_name == "default":
        return os.path.join(configs_dir, "dataset_config.yaml")
    else:
        return os.path.join(configs_dir, f"dataset_config_{config_name}.yaml")


def validate_dataset_config(config_path: str) -> bool:
    """
    验证数据集配置文件的有效性
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        是否有效
    """
    try:
        config = load_config(config_path)
        required_keys = ['dataset']
        dataset_keys = ['data_root', 'json_file', 'image_size', 'batch_size']
        
        # 检查顶层结构
        for key in required_keys:
            if key not in config:
                print(f"错误: 缺少必需的顶层键 '{key}'")
                return False
        
        # 检查数据集配置
        dataset_config = config['dataset']
        for key in dataset_keys:
            if key not in dataset_config:
                print(f"错误: 缺少必需的数据集配置键 '{key}'")
                return False
        
        print(f"✓ 配置文件 {config_path} 验证通过")
        return True
        
    except Exception as e:
        print(f"错误: 配置文件 {config_path} 验证失败: {e}")
        return False


def main():
    """主函数 - 演示多数据集配置系统的使用"""
    
    # 1. 列出可用的数据集配置
    print("可用的数据集配置文件:")
    available_configs = list_available_dataset_configs()
    for i, config_file in enumerate(available_configs, 1):
        print(f"  {i}. {config_file}")
    
    print("\n" + "="*50)
    
    # 2. 验证所有配置文件
    print("验证数据集配置文件:")
    for config_file in available_configs:
        config_path_full = os.path.join("./configs", config_file)
        validate_dataset_config(config_path_full)
    
    print("\n" + "="*50)

    # 3. 测试加载默认配置
    print("加载默认配置:")
    config = load_training_config("./configs/train/sd15_config.yaml")
    print_config(config, "默认训练配置")

    print("\n" + "="*50)

    # 4. 测试加载测试配置
    print("加载测试配置:")
    test_dataset_path = get_dataset_config_path("raining")
    config_raining = load_training_config("./configs/train/sd15_config.yaml", test_dataset_path)
    print_config(config_raining, "测试训练配置")

    print("\n" + "="*50)

    # 6. 演示配置参数对比
    print("配置参数对比:")
    configs_to_compare = [
        ("默认配置", config),
        ("雨天降噪配置", config_raining)
    ]

    for name, cfg in configs_to_compare:
        dataset_cfg = cfg['dataset']
        print(f"\n{name}:")
        print(f"  数据根目录: {dataset_cfg['data_root']}")
        print(f"  图像尺寸: {dataset_cfg['image_size']}")
        print(f"  批次大小: {dataset_cfg['batch_size']}")
        print(f"  工作进程数: {dataset_cfg['num_workers']}")



if __name__ == "__main__":
    # 测试配置加载
    main()