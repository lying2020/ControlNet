"""
使用公共模块的简化版Pose2Image
检测器特定逻辑保留在各自的文件中
"""
import sys
import cv2
import torch
import gradio as gr
from common_utils import ControlNetProcessor, create_common_ui
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector

# 创建处理器
processor = ControlNetProcessor('./models/controlnet/control_v11p_sd15_openpose.yaml', './models/controlnet/control_v11p_sd15_openpose.pth')
apply_openpose = OpenposeDetector()

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, 
           detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    """Pose特定的处理函数"""
    with torch.no_grad():
        # Pose检测逻辑
        input_image = HWC3(input_image)
        detected_map, _ = apply_openpose(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

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
    input_image, prompt, run_button, num_samples, image_resolution, detect_resolution, strength, guess_mode, ddim_steps, scale, seed, eta, a_prompt, n_prompt, result_gallery = create_common_ui("Control Stable Diffusion with Human Pose")

    # 绑定事件
    run_button.click(
        fn=process,
        inputs=[input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, 
                detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta],
        outputs=[result_gallery]
    )

# 启动服务
if __name__ == "__main__":
    print("正在启动 Gradio 界面...")
    block.launch(server_name='127.0.0.1', share=True, server_port=7860)
