"""
统一的ControlNet Gradio界面
包含所有类型的ControlNet处理器
"""

import gradio as gr
from common_utils import (
    CannyProcessor, DepthProcessor, PoseProcessor, HEDProcessor,
    MLSDProcessor, NormalProcessor, ScribbleProcessor, SegmentationProcessor,
    DETECTOR_PARAMS
)

# 处理器配置
PROCESSORS = {
    'canny': {
        'processor': CannyProcessor('./models/controlnet/control_sd15_canny.pth'),
        'title': 'Control Stable Diffusion with Canny Edge Maps',
        'params': DETECTOR_PARAMS['canny']
    },
    'depth': {
        'processor': DepthProcessor('./models/controlnet/control_sd15_depth.pth'),
        'title': 'Control Stable Diffusion with Depth Maps',
        'params': DETECTOR_PARAMS['depth']
    },
    'pose': {
        'processor': PoseProcessor('./models/controlnet/control_sd15_openpose.pth'),
        'title': 'Control Stable Diffusion with Human Pose',
        'params': DETECTOR_PARAMS['pose']
    },
    'hed': {
        'processor': HEDProcessor('./models/controlnet/control_sd15_hed.pth'),
        'title': 'Control Stable Diffusion with HED Edge Maps',
        'params': DETECTOR_PARAMS['hed']
    },
    'mlsd': {
        'processor': MLSDProcessor('./models/controlnet/control_sd15_mlsd.pth'),
        'title': 'Control Stable Diffusion with MLSD Line Detection',
        'params': DETECTOR_PARAMS['mlsd']
    },
    'normal': {
        'processor': NormalProcessor('./models/controlnet/control_sd15_normal.pth'),
        'title': 'Control Stable Diffusion with Normal Maps',
        'params': DETECTOR_PARAMS['normal']
    },
    'scribble': {
        'processor': ScribbleProcessor('./models/controlnet/control_sd15_scribble.pth'),
        'title': 'Control Stable Diffusion with Scribble',
        'params': {}
    },
    'segmentation': {
        'processor': SegmentationProcessor('./models/controlnet/control_sd15_seg.pth'),
        'title': 'Control Stable Diffusion with Segmentation',
        'params': DETECTOR_PARAMS['segmentation']
    }
}

def create_unified_interface():
    """创建统一的界面"""

    with gr.Blocks(title="ControlNet Unified Interface") as demo:
        gr.Markdown("# ControlNet Unified Interface")
        gr.Markdown("Select a ControlNet type and upload an image to generate controlled images.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 选择ControlNet类型
                control_type = gr.Dropdown(
                    choices=list(PROCESSORS.keys()),
                    value='canny',
                    label="ControlNet Type",
                    info="Select the type of control to apply"
                )
                
                # 输入图像
                input_image = gr.Image(
                    source='upload',
                    type="numpy",
                    label="Input Image"
                )
                
                # 提示词
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here..."
                )
                
                # 运行按钮
                run_button = gr.Button("Generate", variant="primary")
                
                # 高级选项
                with gr.Accordion("Advanced Options", open=False):
                    num_samples = gr.Slider(
                        label="Number of Images",
                        minimum=1,
                        maximum=12,
                        value=1,
                        step=1
                    )
                    
                    image_resolution = gr.Slider(
                        label="Image Resolution",
                        minimum=256,
                        maximum=768,
                        value=512,
                        step=64
                    )
                    
                    strength = gr.Slider(
                        label="Control Strength",
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.01
                    )
                    
                    guess_mode = gr.Checkbox(
                        label='Guess Mode',
                        value=False
                    )
                    
                    ddim_steps = gr.Slider(
                        label="DDIM Steps",
                        minimum=1,
                        maximum=100,
                        value=20,
                        step=1
                    )
                    
                    scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=0.1,
                        maximum=30.0,
                        value=9.0,
                        step=0.1
                    )
                    
                    seed = gr.Slider(
                        label="Seed",
                        minimum=-1,
                        maximum=2147483647,
                        step=1,
                        randomize=True
                    )
                    
                    eta = gr.Number(
                        label="eta (DDIM)",
                        value=0.0
                    )
                    
                    a_prompt = gr.Textbox(
                        label="Added Prompt",
                        value='best quality, extremely detailed'
                    )
                    
                    n_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
                    )
                    
                    # 动态参数容器
                    dynamic_params = gr.Group(label="Detector Parameters")
            
            with gr.Column(scale=2):
                # 输出画廊
                result_gallery = gr.Gallery(
                    label='Generated Images',
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    height="auto"
                )
        
        # 动态更新检测器参数
        def update_detector_params(control_type):
            """根据选择的ControlNet类型更新检测器参数"""
            params = PROCESSORS[control_type]['params']
            param_components = []
            
            for param_name, param_config in params.items():
                if param_config['type'] == 'slider':
                    component = gr.Slider(**param_config)
                elif param_config['type'] == 'checkbox':
                    component = gr.Checkbox(**param_config)
                elif param_config['type'] == 'textbox':
                    component = gr.Textbox(**param_config)
                param_components.append(component)
            
            return param_components

        # 处理函数
        def process_image(control_type, input_image, prompt, a_prompt, n_prompt, 
                         num_samples, image_resolution, ddim_steps, guess_mode, 
                         strength, scale, seed, eta, *detector_params):
            """处理图像的主函数"""
            if input_image is None:
                return []
            
            processor_info = PROCESSORS[control_type]
            processor = processor_info['processor']
            
            # 构建检测器参数字典
            detector_param_names = list(processor_info['params'].keys())
            detector_param_dict = {}
            for i, param_name in enumerate(detector_param_names):
                if i < len(detector_params):
                    detector_param_dict[param_name] = detector_params[i]
            
            # 调用处理器
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
                **detector_param_dict
            )
        
        # 绑定事件
        control_type.change(
            fn=update_detector_params,
            inputs=[control_type],
            outputs=[dynamic_params]
        )
        
        # 获取初始参数
        initial_params = update_detector_params('canny')
        
        # 绑定运行按钮
        run_button.click(
            fn=process_image,
            inputs=[
                control_type, input_image, prompt, a_prompt, n_prompt,
                num_samples, image_resolution, ddim_steps, guess_mode,
                strength, scale, seed, eta
            ] + initial_params,
            outputs=[result_gallery]
        )
    
    return demo

# 创建并启动界面
if __name__ == "__main__":
    demo = create_unified_interface()
    demo.launch(server_name='0.0.0.0', share=True)