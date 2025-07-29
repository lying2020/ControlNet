"""
使用公共模块的简化版Canny2Image
检测器特定逻辑保留在各自的文件中
"""
import sys
import torch
import gradio as gr
from common_utils import ControlNetProcessor, create_common_ui
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector

# 创建处理器
processor = ControlNetProcessor('./models/cldm_v15.yaml', './models/control_sd15_canny.pth')
apply_canny = CannyDetector()

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, 
           ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    """Canny特定的处理函数"""
    with torch.no_grad():
        # Canny检测逻辑
        img = resize_image(HWC3(input_image), image_resolution)
        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

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
    # 使用公共UI组件
    input_image, prompt, run_button, num_samples, image_resolution, strength, guess_mode, ddim_steps, scale, seed, eta, a_prompt, n_prompt, result_gallery = create_common_ui("Control Stable Diffusion with Canny Edge Maps")

    # 添加Canny特定参数
    with gr.Accordion("Advanced options", open=False):
        low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
        high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)

    # 绑定事件
    run_button.click(
        fn=process,
        inputs=[input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, 
                ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold],
        outputs=[result_gallery]
    )

# 启动服务
if __name__ == "__main__":
    block.launch(server_name='0.0.0.0')