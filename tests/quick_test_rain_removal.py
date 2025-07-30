#!/usr/bin/env python3
"""
快速去雨效果测试脚本
简化版本，用于快速验证模型效果
支持原始Stable Diffusion和ControlNet两种模式
"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import argparse
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from share import *
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


def load_model(sd_version='sd15', use_controlnet=True):
    """加载模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if use_controlnet:
        # 使用ControlNet模型
        if sd_version == 'sd15':
            config_path = './models/cldm_v15.yaml'
            model_path = './models/controlnet/control_sd15_init.ckpt'
        elif sd_version == 'sd21':
            config_path = './models/cldm_v21.yaml'
            model_path = './models/controlnet/control_sd21_ini.ckpt'
        else:
            raise ValueError(f"不支持的SD版本: {sd_version}")

        print(f"加载 {sd_version.upper()} ControlNet 模型...")

        # 检查文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 创建模型
        model = create_model(config_path).to(device)
        model.load_state_dict(load_state_dict(model_path, location=device))
        model.eval()
    else:
        # 使用原始Stable Diffusion模型
        if sd_version == 'sd15':
            model_path = './models/stable-diffusion/v1-5-pruned.ckpt'
        elif sd_version == 'sd21':
            model_path = './models/stable-diffusion/v2-1_512-ema-pruned.ckpt'
        else:
            raise ValueError(f"不支持的SD版本: {sd_version}")

        print(f"加载 {sd_version.upper()} 原始 Stable Diffusion 模型...")

        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 创建原始SD配置文件
        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config
        
        # 使用与ControlNet相同的配置，但不包含control_stage_config
        config = OmegaConf.create({
            "model": {
                "target": "ldm.models.diffusion.ddpm.LatentDiffusion",
                "params": {
                    "linear_start": 0.00085,
                    "linear_end": 0.0120,
                    "num_timesteps_cond": 1,
                    "log_every_t": 200,
                    "timesteps": 1000,
                    "first_stage_key": "jpg",
                    "cond_stage_key": "txt",
                    "image_size": 64,
                    "channels": 4,
                    "cond_stage_trainable": False,
                    "conditioning_key": "crossattn",
                    "monitor": "val/loss_simple_ema",
                    "scale_factor": 0.18215,
                    "use_ema": False,
                    
                    "first_stage_config": {
                        "target": "ldm.models.autoencoder.AutoencoderKL",
                        "params": {
                            "embed_dim": 4,
                            "monitor": "val/rec_loss",
                            "ddconfig": {
                                "double_z": True,
                                "z_channels": 4,
                                "resolution": 256,
                                "in_channels": 3,
                                "out_ch": 3,
                                "ch": 128,
                                "ch_mult": [1, 2, 4, 4],
                                "num_res_blocks": 2,
                                "attn_resolutions": [],
                                "dropout": 0.0
                            },
                            "lossconfig": {
                                "target": "torch.nn.Identity"
                            }
                        }
                    },
                    
                    "cond_stage_config": {
                        "target": "ldm.modules.encoders.modules.FrozenCLIPEmbedder"
                    },
                    
                    "unet_config": {
                        "target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
                        "params": {
                            "image_size": 32,
                            "in_channels": 4,
                            "out_channels": 4,
                            "model_channels": 320,
                            "attention_resolutions": [4, 2, 1],
                            "num_res_blocks": 2,
                            "channel_mult": [1, 2, 4, 4],
                            "num_heads": 8,
                            "use_spatial_transformer": True,
                            "transformer_depth": 1,
                            "context_dim": 768,
                            "use_checkpoint": True,
                            "legacy": False
                        }
                    }
                }
            }
        })

        # 创建模型
        model = instantiate_from_config(config.model).to(device)
        
        # 加载权重，使用strict=False忽略不匹配的键
        state_dict = load_state_dict(model_path, location=device)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"警告: 缺失的键: {missing_keys[:10]}...")  # 只显示前10个
        if unexpected_keys:
            print(f"警告: 意外的键: {unexpected_keys[:10]}...")  # 只显示前10个
            
        model.eval()

    return model, device


def test_text_to_image(model, device, prompt, output_path, use_controlnet=True):
    """测试纯文本生成"""
    print(f"文本生成测试: {prompt}")

    # 创建采样器
    ddim_sampler = DDIMSampler(model)

    # 设置参数
    num_samples = 1
    ddim_steps = 20  # 减少步数以加快速度
    scale = 9.0
    seed = 42

    # 设置随机种子
    torch.manual_seed(seed)

    # 准备提示词
    prompts = [prompt] * num_samples
    negative_prompts = ["rain, water drops, blurry, low quality"] * num_samples

    # 生成图像
    with torch.no_grad():
        if use_controlnet:
            # ControlNet格式
            cond = {
                "c_concat": [None],  # 没有图像控制信号
                "c_crossattn": [model.get_learned_conditioning(prompts)]
            }
            un_cond = {
                "c_concat": [None],  # 没有图像控制信号
                "c_crossattn": [model.get_learned_conditioning(negative_prompts)]
            }
        else:
            # 原始SD格式
            cond = model.get_learned_conditioning(prompts)
            un_cond = model.get_learned_conditioning(negative_prompts)

        shape = [4, 64, 64]
        samples, _ = ddim_sampler.sample(
            ddim_steps, num_samples, shape, cond, eta=0, verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond
        )

        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples = 255. * x_samples.permute(0, 2, 3, 1).cpu().numpy()

    # 保存图像
    sample = x_samples[0]
    Image.fromarray(sample.astype(np.uint8)).save(output_path)
    print(f"保存: {output_path}")


def test_image_conditioning(model, device, image_path, prompt, output_path, use_controlnet=True):
    """测试图像条件生成"""
    print(f"图像条件测试: {image_path}")

    # 检查输入图像
    if not os.path.exists(image_path):
        print(f"输入图像不存在: {image_path}")
        return

    # 创建采样器
    ddim_sampler = DDIMSampler(model)

    # 加载和预处理图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))

    # 归一化到[-1, 1]
    image = (image.astype(np.float32) / 127.5) - 1.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)

    # 设置参数
    num_samples = 1
    ddim_steps = 20
    scale = 9.0
    seed = 42

    # 设置随机种子
    torch.manual_seed(seed)

    # 准备提示词
    prompts = [prompt] * num_samples
    negative_prompts = ["rain, water drops, blurry, low quality"] * num_samples

    # 生成图像
    with torch.no_grad():
        if use_controlnet:
            # 使用图像作为控制信号
            control = torch.cat([image] * num_samples, dim=0)
            
            # ControlNet格式
            cond = {
                "c_concat": [control],
                "c_crossattn": [model.get_learned_conditioning(prompts)]
            }
            un_cond = {
                "c_concat": [control],
                "c_crossattn": [model.get_learned_conditioning(negative_prompts)]
            }
        else:
            # 原始SD格式 - 只使用文本条件
            cond = model.get_learned_conditioning(prompts)
            un_cond = model.get_learned_conditioning(negative_prompts)

        shape = [4, 64, 64]
        samples, _ = ddim_sampler.sample(
            ddim_steps, num_samples, shape, cond, eta=0, verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond
        )

        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples = 255. * x_samples.permute(0, 2, 3, 1).cpu().numpy()

    # 保存图像
    sample = x_samples[0]
    Image.fromarray(sample.astype(np.uint8)).save(output_path)
    print(f"保存: {output_path}")


def test_reconstruction(model, device, image_path, output_path, use_controlnet=True):
    """测试重建模式"""
    print(f"重建模式测试: {image_path}")

    # 检查输入图像
    if not os.path.exists(image_path):
        print(f"输入图像不存在: {image_path}")
        return

    # 创建采样器
    ddim_sampler = DDIMSampler(model)

    # 加载和预处理图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))

    # 归一化到[-1, 1]
    image = (image.astype(np.float32) / 127.5) - 1.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)

    # 设置参数
    num_samples = 1
    ddim_steps = 20
    scale = 9.0
    seed = 42

    # 设置随机种子
    torch.manual_seed(seed)

    # 重建模式：使用空提示词
    prompts = [""] * num_samples
    negative_prompts = [""] * num_samples

    # 生成图像
    with torch.no_grad():
        if use_controlnet:
            # 使用图像作为控制信号
            control = torch.cat([image] * num_samples, dim=0)
            
            # ControlNet格式
            cond = {
                "c_concat": [control],
                "c_crossattn": [model.get_learned_conditioning(prompts)]
            }
            un_cond = {
                "c_concat": [control],
                "c_crossattn": [model.get_learned_conditioning(negative_prompts)]
            }
        else:
            # 原始SD格式 - 只使用文本条件
            cond = model.get_learned_conditioning(prompts)
            un_cond = model.get_learned_conditioning(negative_prompts)

        shape = [4, 64, 64]
        samples, _ = ddim_sampler.sample(
            ddim_steps, num_samples, shape, cond, eta=0, verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond
        )

        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples = 255. * x_samples.permute(0, 2, 3, 1).cpu().numpy()

    # 保存图像
    sample = x_samples[0]
    Image.fromarray(sample.astype(np.uint8)).save(output_path)
    print(f"保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='快速去雨效果测试')
    parser.add_argument('--sd_version', type=str, default='sd15', choices=['sd15', 'sd21'],
                       help='SD版本 (sd15 或 sd21)')
    parser.add_argument('--use_controlnet', action='store_true', default=False,
                       help='是否使用ControlNet模型 (默认使用原始SD模型)')
    parser.add_argument('--output_dir', type=str, default='./tests/quick_test_results',
                       help='输出目录')
    parser.add_argument('--test_image', type=str, default=None,
                       help='指定测试图像路径（可选）')

    args = parser.parse_args()


    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    model, device = load_model(args.sd_version, args.use_controlnet)

    model_type = "ControlNet" if args.use_controlnet else "Pure_SD"
    print(f"\n{'='*50}")
    print(f"开始 {args.sd_version.upper()} {model_type} 快速测试")
    print(f"{'='*50}")

    # 1. 文本条件测试
    text_prompts = [
        "A clear sunny day without rain, high quality image",
        "Remove rain from the scene, clear weather conditions"
    ]

    for i, prompt in enumerate(text_prompts):
        output_path = os.path.join(args.output_dir, f"{args.sd_version}_{model_type}_text_{i+1}.png")
        test_text_to_image(model, device, prompt, output_path, args.use_controlnet)

    # 2. 图像条件测试
    # 选择测试图像
    if args.test_image and os.path.exists(args.test_image):
        test_images = [args.test_image]
    else:
        # 从raining数据集中选择图像
        source_dir = "./datasets/raining/source"
        if os.path.exists(source_dir):
            source_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
            test_images = [os.path.join(source_dir, source_files[0])] if source_files else []
        else:
            test_images = []

    for i, image_path in enumerate(test_images):
        # 图像+文本条件测试
        output_path = os.path.join(args.output_dir, f"{args.sd_version}_{model_type}_image_text_{i+1}.png")
        test_image_conditioning(
            model, device, image_path,
            "Remove rain from the image, clear weather, high quality",
            output_path, args.use_controlnet
        )

        # 加载和预处理图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))

        # 保存图像
        original_image_path = os.path.join(args.output_dir, f"{args.sd_version}_{model_type}_original_{i+1}.png")
        Image.fromarray(image.astype(np.uint8)).save(original_image_path)

        # 重建模式测试
        output_path = os.path.join(args.output_dir, f"{args.sd_version}_{model_type}_reconstruction_{i+1}.png")
        test_reconstruction(model, device, image_path, output_path, args.use_controlnet)

    print(f"\n{'='*50}")
    print(f"{args.sd_version.upper()} {model_type} 快速测试完成")
    print(f"结果保存在: {args.output_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 