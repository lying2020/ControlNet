"""
Gradio公共工具模块
只包含真正公共的代码部分
"""

import sys
import os
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from share import *
import configs.config as config

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ControlNetProcessor:
    """ControlNet处理器基类 - 只包含公共的处理逻辑"""

    def __init__(self, model_yaml, model_path):
        """
        初始化处理器
        
        Args:
            model_yaml: 模型配置文件路径
            model_path: ControlNet模型路径
        """
        self.model = None
        self.ddim_sampler = None
        self._load_model(model_yaml, model_path)

    def _load_model(self, model_yaml, model_path):
        """加载模型"""
        self.model = create_model(model_yaml).cpu()
        self.model.load_state_dict(load_state_dict(model_path, location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)

    def process_image(self, input_image, prompt, a_prompt, n_prompt, num_samples, 
                     image_resolution, ddim_steps, guess_mode, strength, scale, 
                     seed, eta, detected_map):
        """
        处理图像的核心函数 - 公共的ControlNet处理逻辑
        
        Args:
            input_image: 输入图像
            prompt: 提示词
            a_prompt: 附加提示词
            n_prompt: 负面提示词
            num_samples: 生成样本数
            image_resolution: 图像分辨率
            ddim_steps: DDIM步数
            guess_mode: 猜测模式
            strength: 控制强度
            scale: 引导比例
            seed: 随机种子
            eta: DDIM eta参数
            detected_map: 检测到的控制图 (由具体gradio文件提供)
        
        Returns:
            处理结果列表
        """
        with torch.no_grad():
            # 预处理输入图像
            input_image = HWC3(input_image)
            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            # 准备控制信号
            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            # 设置随机种子
            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            # 内存优化
            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            # 准备条件
            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            # 设置控制比例
            # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

            # 采样
            samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                             shape, cond, verbose=False, eta=eta,
                                                             unconditional_guidance_scale=scale,
                                                             unconditional_conditioning=un_cond)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            # 解码结果
            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
            return [detected_map] + results

    def process_image_with_detector(self, input_image, prompt, a_prompt, n_prompt, num_samples, 
                                  image_resolution, detect_resolution, ddim_steps, guess_mode, 
                                  strength, scale, seed, eta, detector_func, **detector_kwargs):
        """
        带检测器的图像处理函数 - 用于需要检测器的场景
        
        Args:
            input_image: 输入图像
            prompt: 提示词
            a_prompt: 附加提示词
            n_prompt: 负面提示词
            num_samples: 生成样本数
            image_resolution: 图像分辨率
            detect_resolution: 检测分辨率
            ddim_steps: DDIM步数
            guess_mode: 猜测模式
            strength: 控制强度
            scale: 引导比例
            seed: 随机种子
            eta: DDIM eta参数
            detector_func: 检测器函数
            **detector_kwargs: 检测器参数
        
        Returns:
            处理结果列表
        """
        with torch.no_grad():
            # 预处理输入图像
            input_image = HWC3(input_image)
            detected_map = detector_func(resize_image(input_image, detect_resolution), **detector_kwargs)
            detected_map = HWC3(detected_map)
            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            # 调整检测图尺寸
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

            # 准备控制信号
            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            # 设置随机种子
            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            # 内存优化
            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            # 准备条件
            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            # 设置控制比例
            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

            # 采样
            samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                             shape, cond, verbose=False, eta=eta,
                                                             unconditional_guidance_scale=scale,
                                                             unconditional_conditioning=un_cond)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            # 解码结果
            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
            return [detected_map] + results


def create_common_ui(title, include_detect_resolution=True):
    """
    创建通用的Gradio界面 - 只包含公共的UI元素

    Args:
        title: 界面标题
        include_detect_resolution: 是否包含检测分辨率滑块
    
    Returns:
        Gradio界面组件
    """
    with gr.Row():
        gr.Markdown(f"## {title}")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")

            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                if include_detect_resolution:
                    detect_resolution = gr.Slider(label="Detection Resolution", minimum=128, maximum=1024, value=512, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')

        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')

    if include_detect_resolution:
        return input_image, prompt, run_button, num_samples, image_resolution, detect_resolution, strength, guess_mode, ddim_steps, scale, seed, eta, a_prompt, n_prompt, result_gallery
    else:
        return input_image, prompt, run_button, num_samples, image_resolution, strength, guess_mode, ddim_steps, scale, seed, eta, a_prompt, n_prompt, result_gallery