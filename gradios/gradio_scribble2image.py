"""
使用公共模块的简化版Scribble2Image
检测器特定逻辑保留在各自的文件中
"""
import sys
import numpy as np
import torch
import gradio as gr
from common_utils import ControlNetProcessor, create_common_ui
from cores.annotator.util import resize_image, HWC3

# 创建处理器
processor = ControlNetProcessor('./models/cldm_v15.yaml', './models/controlnet/control_sd15_scribble.pth')

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, 
           ddim_steps, guess_mode, strength, scale, seed, eta):
    """Scribble特定的处理函数"""
    with torch.no_grad():
        # Scribble检测逻辑
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) < 127] = 255

        # 使用公共处理逻辑
        return processor.process_image(
            input_image=input_image,
            prompt=prompt,
            a_prompt=a_prompt,
            n_prompt=n_prompt,
            num_samples=num_samples,
            image_resolution=image_resolution,
            ddim_steps=ddim_steps,
            guess_mode=guess_mode,
            strength=strength,
            scale=scale,
            seed=seed,
            eta=eta,
            detected_map=detected_map
        )

# 创建界面
block = gr.Blocks().queue()
with block:
    # 使用公共UI组件（不包含检测分辨率）
    input_image, prompt, run_button, num_samples, image_resolution, strength, guess_mode, ddim_steps, scale, seed, eta, a_prompt, n_prompt, result_gallery = create_common_ui("Control Stable Diffusion with Scribble Maps", include_detect_resolution=False)

    # 绑定事件
    run_button.click(
        fn=process,
        inputs=[input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, 
                ddim_steps, guess_mode, strength, scale, seed, eta],
        outputs=[result_gallery]
    )

# 启动服务
if __name__ == "__main__":
    block.launch(server_name='0.0.0.0')
