# ControlNet去雨效果测试脚本

本目录包含两个测试脚本，用于测试SD15和SD21模型在不同条件下的去雨效果。

## 脚本说明

### 1. `test_rain_removal.py` - 完整测试脚本
功能全面的测试脚本，包含详细的测试报告和多种测试模式。

### 2. `quick_test_rain_removal.py` - 快速测试脚本
简化版本，用于快速验证模型效果，测试速度更快。

## 测试内容

### 1. 文本条件测试
- **功能**：仅使用文本提示词生成去雨图像
- **输入**：去雨相关的文本提示词
- **输出**：生成的清晰图像
- **示例提示词**：
  - "A clear sunny day without rain, high quality image, detailed"
  - "Remove rain from the scene, clear weather conditions, professional photography"

### 2. 图像+文本条件测试
- **功能**：使用有雨图像作为条件，结合文本提示词生成去雨图像
- **输入**：有雨图像 + 去雨文本提示词
- **输出**：去除雨滴的清晰图像
- **测试图像**：从 `datasets/raining/source/` 目录中选择

### 3. 重建模式测试
- **功能**：类似训练阶段，直接输入有雨图像，输出重建的清晰图像
- **输入**：有雨图像（无文本提示词）
- **输出**：重建的清晰图像

## 使用方法

### 快速测试（推荐）

```bash
# 测试SD15模型
python quick_test_rain_removal.py --sd_version sd15

# 测试SD21模型
python quick_test_rain_removal.py --sd_version sd21

# 指定输出目录
python quick_test_rain_removal.py --sd_version sd15 --output_dir ./my_test_results

# 使用指定图像测试
python quick_test_rain_removal.py --sd_version sd15 --test_image ./path/to/rain_image.png
```

### 完整测试

```bash
# 运行所有测试
python test_rain_removal.py --sd_version sd15

# 指定输出目录
python test_rain_removal.py --sd_version sd21 --output_dir ./full_test_results
```

## 参数说明

### 快速测试脚本参数
- `--sd_version`: SD版本，可选 'sd15' 或 'sd21'（默认：sd15）
- `--output_dir`: 输出目录（默认：./quick_test_results）
- `--test_image`: 指定测试图像路径（可选）

### 完整测试脚本参数
- `--sd_version`: SD版本，可选 'sd15' 或 'sd21'（默认：sd15）
- `--output_dir`: 输出目录（默认：./test_results）
- `--test_type`: 测试类型，可选 'all', 'text', 'image_text', 'reconstruction'（默认：all）

## 输出文件说明

### 快速测试输出
```
quick_test_results/
├── sd15_text_1.png          # 文本生成测试1
├── sd15_text_2.png          # 文本生成测试2
├── sd15_image_text_1.png    # 图像+文本条件测试
└── sd15_reconstruction_1.png # 重建模式测试
```

### 完整测试输出
```
test_results/
└── sd15_test_20241201_143022/
    ├── text_test_1_text_1.png
    ├── text_test_2_text_1.png
    ├── text_test_3_text_1.png
    ├── image_text_test_1_image_text_1.png
    ├── image_text_test_2_image_text_1.png
    ├── image_text_test_3_image_text_1.png
    ├── reconstruction_test_1_reconstruction_1.png
    ├── reconstruction_test_2_reconstruction_1.png
    └── reconstruction_test_3_reconstruction_1.png
```

## 模型文件要求

确保以下模型文件存在：

### SD15模型
- `./models/cldm_v15.yaml`
- `./models/controlnet/control_sd15_init.ckpt`

### SD21模型
- `./models/cldm_v21.yaml`
- `./models/controlnet/control_sd21_ini.ckpt`

## 测试图像要求

- 测试图像会自动从 `datasets/raining/source/` 目录中选择
- 支持PNG格式图像
- 图像会被自动调整到512x512像素

## 性能优化

### 快速测试优化
- 减少DDIM步数（20步而不是50步）
- 只生成1个样本
- 简化测试流程

### 完整测试特点
- 使用标准DDIM步数（50步）
- 多种提示词组合
- 详细的测试报告

## 故障排除

### 常见问题

1. **模型文件不存在**
   ```
   FileNotFoundError: 配置文件不存在: ./models/cldm_v15.yaml
   ```
   **解决方案**：确保模型文件已正确下载并放置在正确位置

2. **CUDA内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   **解决方案**：减少DDIM步数或使用CPU模式

3. **测试图像不存在**
   ```
   警告: 输入图像不存在: ./datasets/raining/source/xxx.png
   ```
   **解决方案**：确保raining数据集已正确设置

### 调试模式

启用详细错误信息：
```bash
python quick_test_rain_removal.py --sd_version sd15 2>&1 | tee test.log
```

## 示例输出

```
==================================================
开始 SD15 快速测试
==================================================
加载 SD15 模型...
✓ SD15 模型加载完成

文本生成测试: A clear sunny day without rain, high quality image
保存: ./quick_test_results/sd15_text_1.png

文本生成测试: Remove rain from the scene, clear weather conditions
保存: ./quick_test_results/sd15_text_2.png

图像条件测试: ./datasets/raining/source/0_rain.png
保存: ./quick_test_results/sd15_image_text_1.png

重建模式测试: ./datasets/raining/source/0_rain.png
保存: ./quick_test_results/sd15_reconstruction_1.png

==================================================
SD15 快速测试完成
结果保存在: ./quick_test_results
==================================================
```

## 注意事项

1. **GPU内存**：确保有足够的GPU内存运行模型
2. **模型文件**：确保所有必需的模型文件都已下载
3. **数据集**：确保raining数据集已正确设置
4. **输出目录**：脚本会自动创建输出目录
5. **随机种子**：使用固定随机种子确保结果可重现

## 扩展功能

可以根据需要修改脚本：
- 调整DDIM步