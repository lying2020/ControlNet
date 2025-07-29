# ControlNet 图像去模糊训练系统

这个项目基于ControlNet构建了一个专门用于图像去模糊任务的训练系统。该系统能够学习从模糊图像生成语义一致的清晰图像。

## 项目结构

```
├── deblur_dataset.py              # 数据集类定义
├── deblur_controlnet.py           # 去模糊ControlNet模型
├── train_deblur_controlnet.py     # 训练脚本
├── create_deblur_dataset.py       # 数据集生成脚本
└── README_deblur_training.md      # 本文件
```

## 1. 数据集放置和处理

### 1.1 数据集结构

数据集应按照以下结构组织：

```
training/
├── deblur_dataset/
│   ├── clear_images/              # 清晰图片目录
│   │   ├── clear_000000.png
│   │   ├── clear_000001.png
│   │   └── ...
│   ├── blur_images/               # 模糊图片目录
│   │   ├── blur_000000.png
│   │   ├── blur_000001.png
│   │   └── ...
│   └── blur_types.json            # 数据集信息JSON文件
```

### 1.2 JSON文件格式

`blur_types.json` 文件每行包含一个JSON对象：

```json
{"clear_image": "clear_000000.png", "blur_image": "blur_000000.png", "blur_type": "motion_blur"}
{"clear_image": "clear_000001.png", "blur_image": "blur_000001.png", "blur_type": "gaussian_blur"}
```

### 1.3 自动生成数据集

使用提供的脚本自动生成训练数据集：

```bash
python create_deblur_dataset.py \
    --source_dir /path/to/source/images \
    --output_dir ./training/deblur_dataset \
    --num_samples 10000 \
    --image_size 512
```

支持的模糊类型：
- `motion_blur`: 运动模糊
- `gaussian_blur`: 高斯模糊
- `defocus_blur`: 散焦模糊
- `lens_blur`: 镜头模糊

## 2. 训练过程中的损失函数和参数更新

### 2.1 损失函数组成

训练过程中涉及以下损失函数：

1. **扩散模型损失** (继承自ControlNet)
   - 简单损失 (Simple Loss): L1/L2损失
   - VLB损失 (Variational Lower Bound Loss): 变分下界损失

2. **去模糊特定损失** (新增)
   - L1损失: 像素级重建损失
   - MSE损失: 均方误差损失
   - 感知损失: 基于VGG特征的感知损失
   - 梯度损失: 保持边缘清晰度的梯度损失

### 2.2 参数更新策略

```python
# 优化的参数包括：
params = list(self.control_model.parameters())  # ControlNet参数
if not self.sd_locked:
    params += list(self.model.diffusion_model.output_blocks.parameters())  # UNet输出块
    params += list(self.model.diffusion_model.out.parameters())           # UNet输出层
```

### 2.3 损失权重配置

```python
total_loss = (
    diffusion_loss +                                    # 扩散损失
    deblur_loss_weight * (
        l1_loss +                                       # L1损失
        mse_loss +                                      # MSE损失
        perceptual_loss_weight * perceptual_loss +      # 感知损失
        0.1 * gradient_loss                             # 梯度损失
    )
)
```

## 3. 训练命令

### 3.1 基础训练

```bash
python train_deblur_controlnet.py \
    --data_root ./training/deblur_dataset \
    --json_file blur_types.json \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --deblur_loss_weight 1.0 \
    --perceptual_loss_weight 0.1 \
    --max_epochs 100 \
    --gpus 1
```

### 3.2 使用注意力机制

```bash
python train_deblur_controlnet.py \
    --data_root ./training/deblur_dataset \
    --json_file blur_types.json \
    --use_attention \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --deblur_loss_weight 1.0 \
    --perceptual_loss_weight 0.1 \
    --max_epochs 100 \
    --gpus 1
```

### 3.3 使用数据增强

```bash
python train_deblur_controlnet.py \
    --data_root ./training/deblur_dataset \
    --json_file blur_types.json \
    --use_augmentation \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --deblur_loss_weight 1.0 \
    --perceptual_loss_weight 0.1 \
    --max_epochs 100 \
    --gpus 1
```

### 3.4 多GPU训练

```bash
python train_deblur_controlnet.py \
    --data_root ./training/deblur_dataset \
    --json_file blur_types.json \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --deblur_loss_weight 1.0 \
    --perceptual_loss_weight 0.1 \
    --max_epochs 100 \
    --gpus 2 \
    --precision 16
```

## 4. 模型架构

### 4.1 DeblurControlNet

继承自ControlLDM，添加了去模糊特定的损失函数：

```python
class DeblurControlNet(ControlLDM):
    def __init__(self, deblur_loss_weight=1.0, perceptual_loss_weight=0.1):
        # 初始化去模糊特定的损失函数
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = self._setup_perceptual_loss()
```

### 4.2 DeblurControlNetWithAttention

在基础模型上添加注意力机制：

```python
class DeblurControlNetWithAttention(DeblurControlNet):
    def __init__(self):
        self.attention_module = self._build_attention_module()
```

## 5. 训练监控

训练过程中会记录以下指标：

- `train/loss`: 总损失
- `train/loss_simple`: 简单损失
- `train/loss_vlb`: VLB损失
- `train/deblur_l1_loss`: 去模糊L1损失
- `train/deblur_mse_loss`: 去模糊MSE损失
- `train/deblur_perceptual_loss`: 感知损失
- `train/deblur_gradient_loss`: 梯度损失
- `train/total_deblur_loss`: 总去模糊损失

## 6. 推理使用

训练完成后，可以使用训练好的模型进行推理：

```python
from deblur_controlnet import DeblurControlNet

# 加载模型
model = DeblurControlNet.load_from_checkpoint("path/to/checkpoint.ckpt")

# 推理
with torch.no_grad():
    result = model.sample(
        cond={"c_concat": [blur_image], "c_crossattn": [text_prompt]},
        batch_size=1,
        ddim_steps=50
    )
```

## 7. 性能优化建议

1. **数据质量**: 确保清晰图片和模糊图片的质量
2. **损失权重**: 根据具体任务调整损失权重
3. **学习率**: 从较小的学习率开始，逐步调整
4. **批次大小**: 根据GPU内存调整批次大小
5. **数据增强**: 使用数据增强提高模型泛化能力

## 8. 故障排除

### 常见问题

1. **内存不足**: 减小批次大小或使用梯度累积
2. **训练不收敛**: 调整学习率或损失权重
3. **生成质量差**: 检查数据集质量或增加训练轮数

### 调试技巧

1. 使用较小的数据集进行快速测试
2. 监控损失曲线，确保损失在下降
3. 定期保存和检查生成的样本图像 