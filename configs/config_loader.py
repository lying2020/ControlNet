"""
简单的配置加载器
"""

import yaml
import os


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 确保数值类型正确
    if 'model' in config and 'learning_rate' in config['model']:
        lr = config['model']['learning_rate']
        if isinstance(lr, str):
            # 处理科学计数法字符串
            if 'e' in lr.lower():
                config['model']['learning_rate'] = float(lr)
            else:
                config['model']['learning_rate'] = float(lr)
    
    return config


def get_config_path(sd_version):
    """获取配置文件路径"""
    return f"configs/train/{sd_version}_config.yaml"


def load_sd_config(sd_version):
    """加载指定SD版本的配置"""
    config_path = get_config_path(sd_version)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    return load_config(config_path) 