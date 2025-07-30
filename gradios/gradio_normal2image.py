"""
使用公共模块的简化版Normal2Image
检测器特定逻辑保留在各自的文件中
"""
import sys
import cv2
import torch
import gradio as gr
from common_utils import ControlNetProcessor, create_common_ui
from cores.annotator.util import resize_image, HWC3
from cores.annotator.midas import MidasDetector

# 创建处理器
processor = ControlNetProcessor('./models/cldm_v15.yaml', './models/controlnet/control_sd15_normal.pth')
apply_midas = MidasDetector()

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, 
           detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, bg_threshold):
    """Normal特定的处理函数"""
    with torch.no_grad():
        # Normal检测逻辑
        input_image = HWC3(input_image)
        _, detected_map = apply_midas(resize_image(input_image, detect_resolution), bg_th=bg_threshold)
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
            detected_map=detected_map[:, :, ::-1]  # 注意：normal需要BGR转RGB
        )

# 创建界面
block = gr.Blocks().queue()
with block:
    # 使用公共UI组件
    input_image, prompt, run_button, num_samples, image_resolution, detect_resolution, strength, guess_mode, ddim_steps, scale, seed, eta, a_prompt, n_prompt, result_gallery = create_common_ui("Control Stable Diffusion with Normal Maps")

    # 添加Normal特定参数
    with gr.Accordion("Advanced options", open=False):
        bg_threshold = gr.Slider(label="Normal background threshold", minimum=0.0, maximum=1.0, value=0.4, step=0.01)

    # 绑定事件
    run_button.click(
        fn=process,
        inputs=[input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, 
                detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, bg_threshold],
        outputs=[result_gallery]
    )

# 启动服务
if __name__ == "__main__":
    block.launch(server_name='0.0.0.0')
