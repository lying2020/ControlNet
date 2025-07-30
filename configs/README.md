# 配置系统说明

## 概述

本配置系统将训练配置分离为主配置和数据集配置，提高了配置的模块化和可维护性。

## 文件结构

```
configs/
├── train/
│   └── sd15_config.yaml          # 主训练配置文件
├── dataset_config_default.yaml   # 默认数据集配置文件
├── dataset_config_test.yaml      # 测试数据集配置文件
├── dataset_config_high_res.yaml  # 高分辨率数据集配置文件
├── config_loader.py              # 配置加载器
├── train_with_config.py          # 训练脚本示例
└── README.md                     # 本文件
```

## 配置文件说明

### 1. 主配置文件 (`sd15_config.yaml`)

包含模型、训练和输出相关的配置：

```yaml
model:
  config_path: './models/cldm_v15.yaml'
  resume_path: './models/controlnet/control_sd15_init.ckpt'
  sd_model_path: './models/stable-diffusion/v1-5-pruned.ckpt'
  learning_rate: 8e-6
  sd_locked: true
  only_mid_control: false

# 数据集配置引用
dataset_config: './configs/dataset_config.yaml'

training:
  max_epochs: 10
  precision: 32
  gpus: 1
  # ... 其他训练参数

output:
  log_dir: './lightning_logs'
  image_log_dir: './image_log'
```

### 2. 数据集配置文件

#### 默认配置 (`dataset_config_default.yaml`)

```yaml
dataset:
  # 数据路径配置
  data_root: './dataset/fill50k'
  json_file: 'prompt.json'
  
  # 图像处理参数
  image_size: 512
  
  # 数据加载参数
  batch_size: 8
  num_workers: 4
  shuffle: true
  pin_memory: true
  
  # 数据归一化参数
  source_normalization: [0, 1]
  target_normalization: [-1, 1]
```

#### 测试配置 (`dataset_config_test.yaml`)

```yaml
dataset:
  data_root: './dataset/test_data'
  json_file: 'test_prompt.json'
  image_size: 256  # 较小的图像尺寸，用于快速测试
  batch_size: 2    # 较小的批次大小，用于快速测试
  num_workers: 2
  # ... 其他参数
  test_mode: true
  max_samples: 100  # 限制最大样本数
```

#### 高分辨率配置 (`dataset_config_high_res.yaml`)

```yaml
dataset:
  data_root: './dataset/high_res_data'
  json_file: 'high_res_prompt.json'
  image_size: 768  # 高分辨率图像
  batch_size: 4    # 较小的批次大小，适应高分辨率
  # ... 其他参数
  high_res_mode: true
  memory_optimization: true
```

## 使用方法

### 1. 基本使用

```python
from configs.config_loader import load_training_config

# 加载默认配置
config = load_training_config("./configs/train/sd15_config.yaml")

# 加载指定数据集配置
config_test = load_training_config(
    "./configs/train/sd15_config.yaml", 
    "./configs/dataset_config_test.yaml"
)

# 访问配置参数
learning_rate = config['model']['learning_rate']
batch_size = config['dataset']['batch_size']
max_epochs = config['training']['max_epochs']
```

### 2. 配置验证

```python
from configs.config_loader import print_config

# 打印完整配置
print_config(config, "训练配置")
```

### 3. 配置修改和保存

```python
from configs.config_loader import save_config

# 修改配置
config['dataset']['batch_size'] = 16
config['training']['max_epochs'] = 20

# 保存配置
save_config(config, "./configs/train/sd15_config_modified.yaml")
```

## 优势

### 1. 模块化
- 数据集配置独立管理
- 便于不同数据集之间的切换
- 减少配置文件的复杂度

### 2. 可维护性
- 数据集参数集中管理
- 便于批量修改数据集相关参数
- 减少配置错误

### 3. 可扩展性
- 易于添加新的数据集配置
- 支持配置继承和覆盖
- 便于实验管理

## 示例

### 1. 运行配置加载器测试

```bash
cd configs
python config_loader.py
```

这将演示：
1. 列出可用的数据集配置文件
2. 验证所有配置文件
3. 加载不同配置并对比参数

### 2. 使用训练脚本

```bash
# 列出可用的数据集配置
python train_with_config.py --list_configs

# 验证所有配置文件
python train_with_config.py --validate

# 使用默认配置进行训练
python train_with_config.py --dry_run

# 使用测试配置进行训练
python train_with_config.py --dataset_config test --dry_run

# 使用高分辨率配置进行训练
python train_with_config.py --dataset_config high_res --dry_run

# 使用自定义配置文件路径
python train_with_config.py --dataset_config ./configs/dataset_config_custom.yaml --dry_run
```

## 注意事项

1. **路径引用**：数据集配置路径是相对于主配置文件的相对路径
2. **配置合并**：数据集配置会完全替换主配置中的 `dataset` 部分
3. **向后兼容**：如果主配置中没有 `dataset_config` 引用，系统会使用主配置中的 `dataset` 部分
4. **错误处理**：如果数据集配置文件不存在，系统会发出警告但不会中断程序

## 扩展

### 添加新的数据集配置

1. 创建新的数据集配置文件：
```yaml
# configs/dataset_config_custom.yaml
dataset:
  data_root: './dataset/custom'
  json_file: 'custom_prompt.json'
  image_size: 768
  batch_size: 4
  # ... 其他参数
  custom_mode: true
  custom_parameter: "value"
```

2. 使用新配置：
```bash
# 使用配置名称（如果文件名为 dataset_config_custom.yaml）
python train_with_config.py --dataset_config custom --dry_run

# 使用完整路径
python train_with_config.py --dataset_config ./configs/dataset_config_custom.yaml --dry_run
```

### 配置命名规范

- 默认配置：`dataset_config_default.yaml`
- 测试配置：`dataset_config_test.yaml`
- 高分辨率配置：`dataset_config_high_res.yaml`
- 自定义配置：`dataset_config_<name>.yaml`

### 添加配置验证

可以在 `config_loader.py` 中添加配置验证逻辑，确保配置参数的合法性。 