#!/usr/bin/env python3
"""
快速去雨效果测试脚本
简化版本，用于快速验证模型效果
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


def load_model(sd_version='sd15'):
    """加载模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if sd_version == 'sd15':
        config_path = './models/cldm_v15.yaml'
        model_path = './models/controlnet/control_sd15_init.ckpt'
    elif sd_version == 'sd21':
        config_path = './models/cldm_v21.yaml'
        model_path = './models/controlnet/control_sd21_ini.ckpt'
    else:
        raise ValueError(f"不支持的SD版本: {sd_version}")

    print(f"加载 {sd_version.upper()} 模型...")

    # 检查文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 创建模型
    model = create_model(config_path).to(device)
    model.load_state_dict(load_state_dict(model_path, location=device))
    model.eval()

    return model, device


def test_text_to_image(model, device, prompt, output_path):
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
        uc = model.get_learned_conditioning(negative_prompts)
        c = model.get_learned_conditioning(prompts)

        shape = [4, 64, 64]
        samples, _ = ddim_sampler.sample(
            ddim_steps, num_samples, shape, c, uc, eta=0, verbose=False
        )

        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples = 255. * x_samples.permute(0, 2, 3, 1).cpu().numpy()

    # 保存图像
    sample = x_samples[0]
    Image.fromarray(sample.astype(np.uint8)).save(output_path)
    print(f"保存: {output_path}")


def test_image_conditioning(model, device, image_path, prompt, output_path):
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
        uc = model.get_learned_conditioning(negative_prompts)
        c = model.get_learned_conditioning(prompts)

        # 使用图像作为控制信号
        control = torch.cat([image] * num_samples, dim=0)

        shape = [4, 64, 64]
        samples, _ = ddim_sampler.sample(
            ddim_steps, num_samples, shape, c, uc, eta=0, verbose=False, x_T=None,
            control=control
        )

        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples = 255. * x_samples.permute(0, 2, 3, 1).cpu().numpy()

    # 保存图像
    sample = x_samples[0]
    Image.fromarray(sample.astype(np.uint8)).save(output_path)
    print(f"保存: {output_path}")


def test_reconstruction(model, device, image_path, output_path):
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
        uc = model.get_learned_conditioning(negative_prompts)
        c = model.get_learned_conditioning(prompts)

        # 使用图像作为控制信号
        control = torch.cat([image] * num_samples, dim=0)

        shape = [4, 64, 64]
        samples, _ = ddim_sampler.sample(
            ddim_steps, num_samples, shape, c, uc, eta=0, verbose=False, x_T=None,
            control=control
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
    parser.add_argument('--output_dir', type=str, default='./tests/quick_test_results',
                       help='输出目录')
    parser.add_argument('--test_image', type=str, default=None,
                       help='指定测试图像路径（可选）')

    args = parser.parse_args()

    try:
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)

        # 加载模型
        model, device = load_model(args.sd_version)

        print(f"\n{'='*50}")
        print(f"开始 {args.sd_version.upper()} 快速测试")
        print(f"{'='*50}")

        # 1. 文本条件测试
        text_prompts = [
            "A clear sunny day without rain, high quality image",
            "Remove rain from the scene, clear weather conditions"
        ]

        for i, prompt in enumerate(text_prompts):
            output_path = os.path.join(args.output_dir, f"{args.sd_version}_text_{i+1}.png")
            test_text_to_image(model, device, prompt, output_path)

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
            output_path = os.path.join(args.output_dir, f"{args.sd_version}_image_text_{i+1}.png")
            test_image_conditioning(
                model, device, image_path,
                "Remove rain from the image, clear weather, high quality",
                output_path
            )

            # 重建模式测试
            output_path = os.path.join(args.output_dir, f"{args.sd_version}_reconstruction_{i+1}.png")
            test_reconstruction(model, device, image_path, output_path)

        print(f"\n{'='*50}")
        print(f"{args.sd_version.upper()} 快速测试完成")
        print(f"结果保存在: {args.output_dir}")
        print(f"{'='*50}")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 