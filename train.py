from share import *

import os
import sys
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from my_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from configs.config_loader import load_sd_config


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


def train(model, dataloader, training_config):

    # Misc
    logger = ImageLogger(batch_frequency=training_config['logger_freq'])
    trainer = pl.Trainer(
        gpus=training_config['gpus'], 
        precision=training_config['precision'], 
        callbacks=[logger],
        max_epochs=training_config['max_epochs']
    )

    # Train!
    trainer.fit(model, dataloader)


def main():
    parser = argparse.ArgumentParser(description='Create ControlNet model from Stable Diffusion')
    parser.add_argument('--sd_version', type=str, default='sd15', choices=['sd15', 'sd21'],
                       help='Stable Diffusion version (sd15 or sd21)')
    
    args = parser.parse_args()

    # 加载配置
    config = load_sd_config(args.sd_version)
    model_config = config['model']
    dataset_config = config['dataset']
    training_config = config['training']

    # 添加控制网络
    add_control_net(model_config, args.sd_version)

    # 加载模型
    model = load_model(model_config)

    # 加载数据集
    dataloader = load_dataset(dataset_config)

    # 训练
    train(model, dataloader, training_config)



if __name__ == "__main__":
    import argparse
    main() 

