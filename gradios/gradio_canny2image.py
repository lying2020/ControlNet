"""
使用公共模块的简化版Canny2Image
检测器特定逻辑保留在各自的文件中
"""
import sys
import cv2
import torch
import gradio as gr
from common_utils import ControlNetProcessor, create_common_ui
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector

# 创建处理器
processor = ControlNetProcessor('./models/cldm_v15.yaml', './models/controlnet/control_sd15_canny.pth')
apply_canny = CannyDetector()

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, 
           detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    """Canny特定的处理函数"""
    with torch.no_grad():
        # Canny检测逻辑
        input_image = HWC3(input_image)
        detected_map = apply_canny(resize_image(input_image, detect_resolution), low_threshold, high_threshold)
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

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
    input_image, prompt, run_button, num_samples, image_resolution, detect_resolution, strength, guess_mode, ddim_steps, scale, seed, eta, a_prompt, n_prompt, result_gallery = create_common_ui("Control Stable Diffusion with Canny Edge Maps")

    # 添加Canny特定参数
    with gr.Accordion("Advanced options", open=False):
        low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
        high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)

    # 绑定事件
    run_button.click(
        fn=process,
        inputs=[input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, 
                detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold],
        outputs=[result_gallery]
    )

# 启动服务
if __name__ == "__main__":
    print("正在启动 Gradio 界面...")
    try:
        # 尝试使用本地模式
        block.launch(server_name='127.0.0.1', share=False, server_port=7860)
    except Exception as e:
        print(f"本地模式启动失败: {e}")
        print("尝试使用默认设置...")
        block.launch()