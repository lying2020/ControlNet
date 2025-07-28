import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import argparse

from deblur_dataset import DeblurDataset, DeblurDatasetWithAugmentation
from deblur_controlnet import DeblurControlNet, DeblurControlNetWithAttention
from cldm.logger import ImageLogger
from cldm.model import load_state_dict


def create_deblur_model(config_path, resume_path, learning_rate=1e-5, 
                       deblur_loss_weight=1.0, perceptual_loss_weight=0.1):
    """创建去模糊ControlNet模型"""
    
    # 加载配置
    config = OmegaConf.load(config_path)
    
    # 修改配置以使用我们的去模糊模型
    config.model.target = "deblur_controlnet.DeblurControlNet"
    config.model.params.deblur_loss_weight = deblur_loss_weight
    config.model.params.perceptual_loss_weight = perceptual_loss_weight
    
    # 创建模型
    from ldm.util import instantiate_from_config
    model = instantiate_from_config(config.model).cpu()
    
    # 加载预训练权重
    if resume_path:
        model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    
    # 设置学习率
    model.learning_rate = learning_rate
    
    return model


def create_attention_model(config_path, resume_path, learning_rate=1e-5,
                          deblur_loss_weight=1.0, perceptual_loss_weight=0.1):
    """创建带注意力机制的去模糊ControlNet模型"""
    
    # 加载配置
    config = OmegaConf.load(config_path)
    
    # 修改配置以使用带注意力的去模糊模型
    config.model.target = "deblur_controlnet.DeblurControlNetWithAttention"
    config.model.params.deblur_loss_weight = deblur_loss_weight
    config.model.params.perceptual_loss_weight = perceptual_loss_weight
    
    # 创建模型
    from ldm.util import instantiate_from_config
    model = instantiate_from_config(config.model).cpu()
    
    # 加载预训练权重
    if resume_path:
        model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    
    # 设置学习率
    model.learning_rate = learning_rate
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Deblur ControlNet')
    parser.add_argument('--config', type=str, default='./models/cldm_v15.yaml',
                       help='Path to model config file')
    parser.add_argument('--resume', type=str, default='./models/control_sd15_ini.ckpt',
                       help='Path to resume checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to dataset root directory')
    parser.add_argument('--json_file', type=str, default='blur_types.json',
                       help='Name of JSON file containing dataset info')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--deblur_loss_weight', type=float, default=1.0,
                       help='Weight for deblur loss')
    parser.add_argument('--perceptual_loss_weight', type=float, default=0.1,
                       help='Weight for perceptual loss')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of training epochs')
    parser.add_argument('--logger_freq', type=int, default=300,
                       help='Frequency of logging images')
    parser.add_argument('--use_attention', action='store_true',
                       help='Use attention mechanism')
    parser.add_argument('--use_augmentation', action='store_true',
                       help='Use data augmentation')
    parser.add_argument('--sd_locked', action='store_true', default=True,
                       help='Lock Stable Diffusion weights')
    parser.add_argument('--only_mid_control', action='store_true',
                       help='Only control middle layers')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    parser.add_argument('--precision', type=int, default=32,
                       help='Training precision (16 or 32)')
    
    args = parser.parse_args()
    
    # 创建数据集
    if args.use_augmentation:
        dataset = DeblurDatasetWithAugmentation(
            data_root=args.data_root,
            json_file=args.json_file,
            image_size=args.image_size,
            augment=True
        )
    else:
        dataset = DeblurDataset(
            data_root=args.data_root,
            json_file=args.json_file,
            image_size=args.image_size
        )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    if args.use_attention:
        model = create_attention_model(
            config_path=args.config,
            resume_path=args.resume,
            learning_rate=args.learning_rate,
            deblur_loss_weight=args.deblur_loss_weight,
            perceptual_loss_weight=args.perceptual_loss_weight
        )
    else:
        model = create_deblur_model(
            config_path=args.config,
            resume_path=args.resume,
            learning_rate=args.learning_rate,
            deblur_loss_weight=args.deblur_loss_weight,
            perceptual_loss_weight=args.perceptual_loss_weight
        )
    
    # 设置模型参数
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control
    
    # 创建日志记录器
    logger = ImageLogger(batch_frequency=args.logger_freq)
    
    # 创建训练器
    trainer = pl.Trainer(
        gpus=args.gpus,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[logger],
        log_every_n_steps=10,
        val_check_interval=1000,
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        strategy='ddp' if args.gpus > 1 else None,
    )
    
    # 开始训练
    print(f"Starting training with {len(dataset)} samples")
    print(f"Model: {'Attention' if args.use_attention else 'Standard'} DeblurControlNet")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Deblur loss weight: {args.deblur_loss_weight}")
    print(f"Perceptual loss weight: {args.perceptual_loss_weight}")
    
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main() 