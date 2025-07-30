from share import *

import os
import sys
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from my_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from configs.config_loader import load_training_config, get_dataset_config_path, print_config, list_available_dataset_configs
from datetime import datetime


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


def add_control_net(model_config, sd_version):

    # 从配置文件获取路径
    config_path = model_config['config_path']
    input_path = model_config['sd_model_path']
    output_path = model_config['resume_path']

    print(f"Creating ControlNet for {sd_version.upper()}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Config: {config_path}")

    # 检查输入文件存在性
    assert os.path.exists(input_path), f'Input model does not exist: {input_path}'
    assert os.path.exists(os.path.dirname(output_path)), f'Output path is not valid: {output_path}'

    # 如果输出文件已存在，跳过创建
    if os.path.exists(output_path):
        print(f"Output file already exists: {output_path}")
        print("Skipping ControlNet creation...")
        return None

    # 创建模型
    model = create_model(config_path)

    # 加载预训练权重
    pretrained_weights = torch.load(input_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']

    scratch_dict = model.state_dict()

    # 复制权重
    target_dict = {}
    for k in scratch_dict.keys():
        is_control, name = get_node_name(k, 'control_')
        if is_control:
            copy_k = 'model.diffusion_' + name
        else:
            copy_k = k
        if copy_k in pretrained_weights:
            target_dict[k] = pretrained_weights[copy_k].clone()
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'These weights are newly added: {k}')

    model.load_state_dict(target_dict, strict=True)
    torch.save(model.state_dict(), output_path)

    print('======== add_control_net, Done. =========')

    return model


def load_model(model_config):
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(model_config['config_path']).cpu()
    model.load_state_dict(load_state_dict(model_config['resume_path'], location='cpu'))

    # 确保学习率是数值类型
    lr = model_config['learning_rate']
    if isinstance(lr, str):
        lr = float(lr)
    model.learning_rate = lr

    model.sd_locked = model_config['sd_locked']
    model.only_mid_control = model_config['only_mid_control']

    return model


def load_dataset(dataset_config):
    dataset = MyDataset(config=dataset_config)
    dataloader = DataLoader(
        dataset, 
        num_workers=dataset_config['num_workers'], 
        batch_size=dataset_config['batch_size'], 
        shuffle=dataset_config['shuffle'],
        pin_memory=dataset_config['pin_memory']
    )

    return dataloader


def create_log_dirs(sd_version, dataset_name, training_config, dataset_config):
    """
    创建带有详细信息的日志目录

    Args:
        sd_version: SD版本 (sd15/sd21)
        dataset_name: 数据集名称
        training_config: 训练配置
        dataset_config: 数据集配置

    Returns:
        更新后的训练配置
    """
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 获取训练参数
    batch_size = dataset_config.get('batch_size', 8)
    image_size = dataset_config.get('image_size', 512)
    learning_rate = training_config.get('learning_rate', 8e-6)

    # 构建实验名称，包含更多信息
    experiment_name = f"{sd_version}_{dataset_name}_bs{batch_size}_size{image_size}_lr{learning_rate}_{timestamp}"

    # 更新日志路径 - 使用统一的实验目录
    base_log_dir = training_config.get('log_dir', './lightning_logs')
    experiment_dir = os.path.join(base_log_dir, experiment_name)

    # 在实验目录下创建子目录
    log_dir = os.path.join(experiment_dir, 'logs')  # PyTorch Lightning日志
    image_log_dir = os.path.join(experiment_dir, 'image_log')  # 图像日志
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')  # 模型检查点

    # 确保目录存在
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(image_log_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # 创建实验信息文件
    experiment_info = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'sd_version': sd_version,
        'dataset_name': dataset_name,
        'batch_size': batch_size,
        'image_size': image_size,
        'learning_rate': learning_rate,
        'max_epochs': training_config.get('max_epochs', 10),
        'gpus': training_config.get('gpus', 1),
        'precision': training_config.get('precision', 32),
        'directories': {
            'experiment_dir': experiment_dir,
            'logs_dir': log_dir,
            'image_log_dir': image_log_dir,
            'checkpoints_dir': checkpoints_dir
        }
    }

    # 保存实验信息到JSON文件
    import json
    info_file = os.path.join(experiment_dir, 'experiment_info.json')
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_info, f, indent=2, ensure_ascii=False)

    # 更新配置
    training_config['log_dir'] = log_dir
    training_config['image_log_dir'] = image_log_dir
    training_config['experiment_dir'] = experiment_dir
    training_config['experiment_name'] = experiment_name

    print(f"实验名称: {experiment_name}")
    print(f"实验目录: {experiment_dir}")
    print(f"  ├── logs/ (PyTorch Lightning日志)")
    print(f"  ├── image_log/ (图像日志)")
    print(f"  ├── checkpoints/ (模型检查点)")
    print(f"  └── experiment_info.json (实验信息)")
    print(f"实验信息已保存到: {info_file}")

    return training_config


def train(model, dataloader, training_config):

    # 创建ImageLogger，使用配置中的图像日志目录
    image_log_dir = training_config.get('image_log_dir', './image_log')
    logger = ImageLogger(batch_frequency=training_config['logger_freq'])

    # 设置logger的save_dir为图像日志目录的父目录
    logger.save_dir = os.path.dirname(image_log_dir)

    trainer = pl.Trainer(
        gpus=training_config['gpus'], 
        precision=training_config['precision'], 
        callbacks=[logger],
        max_epochs=training_config['max_epochs'],
        default_root_dir=training_config['log_dir'],  # 设置默认根目录
        version=None  # 禁用版本号，避免创建version_0目录
    )

    # Train!
    trainer.fit(model, dataloader)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ControlNet model')
    parser.add_argument('--sd_version', type=str, default='sd15', choices=['sd15', 'sd21'],
                       help='Stable Diffusion version (sd15 or sd21)')
    parser.add_argument('--dataset', type=str, default='raining', 
                       help='Dataset configuration name (default, raining, etc.)')
    parser.add_argument('--list_datasets', action='store_true',
                       help='List available dataset configurations')
    parser.add_argument('--show_config', action='store_true',
                       help='Show loaded configuration without training')

    args = parser.parse_args()

    # 如果请求列出可用数据集配置
    if args.list_datasets:
        print("可用的数据集配置文件:")
        available_configs = list_available_dataset_configs()
        for i, config_file in enumerate(available_configs, 1):
            print(f"  {i}. {config_file}")
        return

    # 确定主配置文件路径
    main_config_path = f"./configs/train/{args.sd_version}_config.yaml"
    
    # 确定数据集配置文件路径
    if args.dataset == 'default':
        dataset_config_path = None  # 使用主配置中的默认路径
    else:
        dataset_config_path = get_dataset_config_path(args.dataset)

    # 加载完整配置
    print(f"正在加载配置...")
    print(f"主配置文件: {main_config_path}")
    if dataset_config_path:
        print(f"数据集配置文件: {dataset_config_path}")

    config = load_training_config(main_config_path, dataset_config_path)

    # 如果只是显示配置，不进行训练
    if args.show_config:
        print_config(config, f"训练配置 (SD{args.sd_version.upper()}, 数据集: {args.dataset})")
        return

    # 提取配置部分
    model_config = config['model']
    dataset_config = config['dataset']
    training_config = config['training']

    # 创建带详细信息的日志目录
    training_config = create_log_dirs(args.sd_version, args.dataset, training_config, dataset_config)

    # 显示配置信息
    print_config(config, f"开始训练 (SD{args.sd_version.upper()}, 数据集: {args.dataset})")

    # 添加控制网络
    add_control_net(model_config, args.sd_version)

    # 加载模型
    print("正在加载模型...")
    model = load_model(model_config)

    # 加载数据集
    print("正在加载数据集...")
    dataloader = load_dataset(dataset_config)

    # 训练
    print("开始训练...")
    train(model, dataloader, training_config)


if __name__ == "__main__":
    main() 

