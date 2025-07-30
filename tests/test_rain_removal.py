#!/usr/bin/env python3
"""
ControlNet去雨效果测试脚本
测试SD15和SD21模型在不同条件下的去雨效果
"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from share import *
from cores.cldm.model import create_model, load_state_dict
from cores.cldm.ddim_hacked import DDIMSampler


class RainRemovalTester:
    def __init__(self, sd_version='sd15'):
        """
        初始化测试器
        
        Args:
            sd_version: SD版本 ('sd15' 或 'sd21')
        """
        self.sd_version = sd_version
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 根据SD版本加载配置
        if sd_version == 'sd15':
            config_path = './models/cldm_v15.yaml'
            model_path = './models/controlnet/control_sd15_init.ckpt'
        elif sd_version == 'sd21':
            config_path = './models/cldm_v21.yaml'
            model_path = './models/controlnet/control_sd21_ini.ckpt'
        else:
            raise ValueError(f"不支持的SD版本: {sd_version}")

        print(f"正在加载 {sd_version.upper()} 模型...")
        print(f"配置路径: {config_path}")
        print(f"模型路径: {model_path}")

        # 检查文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 创建模型
        self.model = create_model(config_path).to(self.device)
        self.model.load_state_dict(load_state_dict(model_path, location=self.device))
        self.model.eval()

        # 创建采样器
        self.ddim_sampler = DDIMSampler(self.model)

        print(f"✓ {sd_version.upper()} 模型加载完成")

    def text_to_image(self, prompt, output_path, negative_prompt="", num_samples=1, 
                     ddim_steps=50, scale=9.0, seed=42):
        """
        1. 纯文本条件生成

        Args:
            prompt: 正向提示词
            output_path: 输出路径
            negative_prompt: 负向提示词
            num_samples: 生成样本数
            ddim_steps: DDIM步数
            scale: CFG scale
            seed: 随机种子
        """
        print(f"\n=== 文本条件生成测试 ===")
        print(f"正向提示词: {prompt}")
        print(f"负向提示词: {negative_prompt}")

        # 设置随机种子
        torch.manual_seed(seed)

        # 准备提示词
        prompts = [prompt] * num_samples
        negative_prompts = [negative_prompt] * num_samples

        # 生成图像
        with torch.no_grad():
            uc = self.model.get_learned_conditioning(negative_prompts)
            c = self.model.get_learned_conditioning(prompts)
            
            shape = [4, 64, 64]  # 默认尺寸
            samples, _ = self.ddim_sampler.sample(
                ddim_steps, num_samples, shape, c, uc, eta=0, verbose=False
            )

            x_samples = self.model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples = 255. * x_samples.permute(0, 2, 3, 1).cpu().numpy()

        # 保存图像
        for i, sample in enumerate(x_samples):
            sample_path = f"{output_path}_text_{i+1}.png"
            Image.fromarray(sample.astype(np.uint8)).save(sample_path)
            print(f"保存图像: {sample_path}")

    def image_and_text_to_image(self, image_path, prompt, output_path, negative_prompt="", 
                               num_samples=1, ddim_steps=50, scale=9.0, seed=42):
        """
        2. 图像+文本条件生成
        
        Args:
            image_path: 输入图像路径
            prompt: 正向提示词
            output_path: 输出路径
            negative_prompt: 负向提示词
            num_samples: 生成样本数
            ddim_steps: DDIM步数
            scale: CFG scale
            seed: 随机种子
        """
        print(f"\n=== 图像+文本条件生成测试 ===")
        print(f"输入图像: {image_path}")
        print(f"正向提示词: {prompt}")
        print(f"负向提示词: {negative_prompt}")

        # 检查输入图像是否存在
        if not os.path.exists(image_path):
            print(f"警告: 输入图像不存在: {image_path}")
            return

        # 加载和预处理图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))

        # 归一化到[-1, 1]
        image = (image.astype(np.float32) / 127.5) - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 设置随机种子
        torch.manual_seed(seed)

        # 准备提示词
        prompts = [prompt] * num_samples
        negative_prompts = [negative_prompt] * num_samples

        # 生成图像
        with torch.no_grad():
            uc = self.model.get_learned_conditioning(negative_prompts)
            c = self.model.get_learned_conditioning(prompts)
            
            # 使用图像作为控制信号
            control = torch.cat([image] * num_samples, dim=0)
            
            shape = [4, 64, 64]
            samples, _ = self.ddim_sampler.sample(
                ddim_steps, num_samples, shape, c, uc, eta=0, verbose=False, x_T=None,
                control=control
            )
            
            x_samples = self.model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples = 255. * x_samples.permute(0, 2, 3, 1).cpu().numpy()

        # 保存图像
        for i, sample in enumerate(x_samples):
            sample_path = f"{output_path}_image_text_{i+1}.png"
            Image.fromarray(sample.astype(np.uint8)).save(sample_path)
            print(f"保存图像: {sample_path}")
    
    def reconstruction_test(self, image_path, output_path, num_samples=1, 
                           ddim_steps=50, scale=9.0, seed=42):
        """
        3. 重建模式测试（类似训练阶段）
        
        Args:
            image_path: 输入图像路径
            output_path: 输出路径
            num_samples: 生成样本数
            ddim_steps: DDIM步数
            scale: CFG scale
            seed: 随机种子
        """
        print(f"\n=== 重建模式测试 ===")
        print(f"输入图像: {image_path}")
        
        # 检查输入图像是否存在
        if not os.path.exists(image_path):
            print(f"警告: 输入图像不存在: {image_path}")
            return
        
        # 加载和预处理图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        
        # 归一化到[-1, 1]
        image = (image.astype(np.float32) / 127.5) - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # 设置随机种子
        torch.manual_seed(seed)
        
        # 重建模式：使用空提示词
        prompts = [""] * num_samples
        negative_prompts = [""] * num_samples
        
        # 生成图像
        with torch.no_grad():
            uc = self.model.get_learned_conditioning(negative_prompts)
            c = self.model.get_learned_conditioning(prompts)
            
            # 使用图像作为控制信号
            control = torch.cat([image] * num_samples, dim=0)
            
            shape = [4, 64, 64]
            samples, _ = self.ddim_sampler.sample(
                ddim_steps, num_samples, shape, c, uc, eta=0, verbose=False, x_T=None,
                control=control
            )
            
            x_samples = self.model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples = 255. * x_samples.permute(0, 2, 3, 1).cpu().numpy()
        
        # 保存图像
        for i, sample in enumerate(x_samples):
            sample_path = f"{output_path}_reconstruction_{i+1}.png"
            Image.fromarray(sample.astype(np.uint8)).save(sample_path)
            print(f"保存图像: {sample_path}")
    
    def run_all_tests(self, output_dir="./tests/test_results"):
        """
        运行所有测试

        Args:
            output_dir: 输出目录
        """
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir = os.path.join(output_dir, f"{self.sd_version}_test_{timestamp}")
        os.makedirs(test_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"开始 {self.sd_version.upper()} 模型测试")
        print(f"输出目录: {test_dir}")
        print(f"{'='*60}")
        
        # 1. 文本条件测试
        text_prompts = [
            "A clear sunny day without rain, high quality image, detailed",
            "Remove rain from the scene, clear weather conditions, professional photography",
            "Clean image without rain, bright and clear weather, sharp details"
        ]
        
        for i, prompt in enumerate(text_prompts):
            output_path = os.path.join(test_dir, f"text_test_{i+1}")
            self.text_to_image(
                prompt=prompt,
                output_path=output_path,
                negative_prompt="rain, water drops, blurry, low quality",
                num_samples=1,
                ddim_steps=50,
                scale=9.0,
                seed=42
            )

        # 2. 图像+文本条件测试
        # 从raining数据集中选择一些测试图像
        source_dir = "./datasets/raining/source"
        if os.path.exists(source_dir):
            source_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
            test_images = source_files[:3]  # 选择前3张图像

            for i, image_file in enumerate(test_images):
                image_path = os.path.join(source_dir, image_file)
                output_path = os.path.join(test_dir, f"image_text_test_{i+1}")

                self.image_and_text_to_image(
                    image_path=image_path,
                    prompt="Remove rain from the image, clear weather, high quality, detailed",
                    output_path=output_path,
                    negative_prompt="rain, water drops, blurry, low quality",
                    num_samples=1,
                    ddim_steps=50,
                    scale=9.0,
                    seed=42
                )

        # 3. 重建模式测试
        if os.path.exists(source_dir):
            for i, image_file in enumerate(test_images):
                image_path = os.path.join(source_dir, image_file)
                output_path = os.path.join(test_dir, f"reconstruction_test_{i+1}")
                
                self.reconstruction_test(
                    image_path=image_path,
                    output_path=output_path,
                    num_samples=1,
                    ddim_steps=50,
                    scale=9.0,
                    seed=42
                )

        print(f"\n{'='*60}")
        print(f"{self.sd_version.upper()} 模型测试完成")
        print(f"结果保存在: {test_dir}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='ControlNet去雨效果测试')
    parser.add_argument('--sd_version', type=str, default='sd15', choices=['sd15', 'sd21'],
                       help='SD版本 (sd15 或 sd21)')
    parser.add_argument('--output_dir', type=str, default='./tests/test_results',
                       help='输出目录')
    parser.add_argument('--test_type', type=str, choices=['all', 'text', 'image_text', 'reconstruction'],
                       default='all', help='测试类型')

    args = parser.parse_args()

    try:
        # 创建测试器
        tester = RainRemovalTester(sd_version=args.sd_version)

        # 运行测试
        tester.run_all_tests(output_dir=args.output_dir)

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 