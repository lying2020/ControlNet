"""
使用公共模块的简化版Interactive Scribble2Image
检测器特定逻辑保留在各自的文件中
"""
import sys
import numpy as np
import torch
import gradio as gr
from common_utils import ControlNetProcessor, create_common_ui
from annotator.util import resize_image, HWC3

# 创建处理器
processor = ControlNetProcessor('./models/cldm_v15.yaml', './models/controlnet/control_sd15_scribble.pth')

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, 
           ddim_steps, guess_mode, strength, scale, seed, eta):
    """Interactive Scribble特定的处理函数"""
    with torch.no_grad():
        # Interactive Scribble检测逻辑
        img = resize_image(HWC3(input_image['mask'][:, :, 0]), image_resolution)
        H, W, C = img.shape

        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) > 127] = 255

        # 使用公共处理逻辑
        results = processor.process_image(
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
        
        # Interactive Scribble特定的后处理
        processed_map = 255 - detected_map
        return [processed_map] + results[1:]  # 替换第一个结果（检测图）

def create_canvas(w, h):
    """创建画布"""
    return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255

# 创建界面
block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Interactive Scribbles")
    with gr.Row():
        with gr.Column():
            canvas_width = gr.Slider(label="Canvas Width", minimum=256, maximum=1024, value=512, step=1)
            canvas_height = gr.Slider(label="Canvas Height", minimum=256, maximum=1024, value=512, step=1)
            create_button = gr.Button(label="Start", value='Open drawing canvas!')
            input_image = gr.Image(source='upload', type='numpy', tool='sketch')
            gr.Markdown(value='Do not forget to change your brush width to make it thinner. (Gradio do not allow developers to set brush width so you need to do it manually.) '
                              'Just click on the small pencil icon in the upper right corner of the above block.')
            create_button.click(fn=create_canvas, inputs=[canvas_width, canvas_height], outputs=[input_image])
            
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            
            # 使用公共UI组件（不包含检测分辨率）
            num_samples, image_resolution, strength, guess_mode, ddim_steps, scale, seed, eta, a_prompt, n_prompt, result_gallery = create_common_ui("", include_detect_resolution=False)[3:]

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
