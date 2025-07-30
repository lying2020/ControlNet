# ControlNet 简化配置系统

## 📁 项目结构

```
ControlNet1.1/
├── config/
│   ├── train/
│   │   ├── sd15_config.yaml          # SD 1.5 配置
│   │   └── sd21_config.yaml          # SD 2.1 配置
│   └── config_loader.py               # 配置加载器
├── tool_add_control_unified.py        # 统一的模型转换工具
├── tutorial_train.py                  # SD 1.5 训练脚本
├── tutorial_train_sd21.py             # SD 2.1 训练脚本
└── ...
```

## 🚀 快速使用

### 1. 创建ControlNet模型

#### SD 1.5
```bash
python tool_add_control_unified.py --sd_version sd15
```

#### SD 2.1
```bash
python tool_add_control_unified.py --sd_version sd21
```

### 2. 训练模型

#### SD 1.5 训练
```bash
python tutorial_train.py
```

#### SD 2.1 训练
```bash
python tutorial_train_sd21.py
```

## ⚙️ 配置文件说明

### SD 1.5 配置 (config/train/sd15_config.yaml)
```yaml
model:
  config_path: './models/cldm_v15.yaml'
  resume_path: './models/controlnet/control_sd15_ini.ckpt'
  sd_model_path: './models/stable-diffusion/v1-5-pruned.ckpt'
  learning_rate: 8e-6
  sd_locked: true
  only_mid_control: false

dataset:
  data_root: './training/fill50k'
  json_file: 'prompt.json'
  image_size: 512
  batch_size: 8
  num_workers: 4
  shuffle: true
  pin_memory: true

training:
  max_epochs: 10
  precision: 32
  gpus: 1
  logger_freq: 300
  log_every_n_steps: 10
  val_check_interval: 1000
  accumulate_grad_batches: 1
  gradient_clip_val: 1.0

output:
  log_dir: './lightning_logs'
  image_log_dir: './image_log'
```

### SD 2.1 配置 (config/train/sd21_config.yaml)
```yaml
model:
  config_path: './models/cldm_v21.yaml'
  resume_path: './models/control_sd21_ini.ckpt'
  sd_model_path: './models/stable-diffusion/v2-1_512-ema-pruned.ckpt'
  learning_rate: 8e-6
  sd_locked: true
  only_mid_control: false

dataset:
  data_root: './training/fill50k'
  json_file: 'prompt.json'
  image_size: 512
  batch_size: 8
  num_workers: 4
  shuffle: true
  pin_memory: true

training:
  max_epochs: 10
  precision: 16  # SD 2.1推荐使用混合精度
  gpus: 1
  logger_freq: 300
  log_every_n_steps: 10
  val_check_interval: 1000
  accumulate_grad_batches: 1
  gradient_clip_val: 1.0

output:
  log_dir: './lightning_logs'
  image_log_dir: './image_log'
```

## 🔧 主要差异

### SD 1.5 vs SD 2.1
| 配置项 | SD 1.5 | SD 2.1 | 说明 |
|--------|--------|--------|------|
| **模型配置** | cldm_v15.yaml | cldm_v21.yaml | 不同的模型架构 |
| **预训练权重** | control_sd15_ini.ckpt | control_sd21_ini.ckpt | 不同的预训练模型 |
| **训练精度** | 32 | 16 | SD 2.1推荐混合精度 |

## 📝 修改配置

### 调整Batch Size
编辑配置文件中的 `dataset.batch_size`：
```yaml
dataset:
  batch_size: 12  # 改为12
```

### 调整学习率
编辑配置文件中的 `model.learning_rate`：
```yaml
model:
  learning_rate: 1e-5  # 改为1e-5
```

### 调整训练轮数
编辑配置文件中的 `training.max_epochs`：
```yaml
training:
  max_epochs: 20  # 改为20
```

## 🎯 使用建议

### RTX 4090 (48GB) 推荐配置
```yaml
dataset:
  batch_size: 12  # 可以更大

training:
  precision: 16   # 混合精度节省显存
```

### RTX 3090 (24GB) 推荐配置
```yaml
dataset:
  batch_size: 6   # 适中的batch size

training:
  precision: 16   # 混合精度
```

### RTX 3080 (10GB) 推荐配置
```yaml
dataset:
  batch_size: 4   # 较小的batch size

training:
  precision: 16   # 混合精度
```

## 🔄 迁移指南

### 从旧脚本迁移

#### 原来的 tutorial_train_sd21.py
```python
# 旧方式 - 硬编码参数
resume_path = './models/control_sd21_ini.ckpt'
batch_size = 8
learning_rate = 8e-6
# ...

# 新方式 - 配置文件
config = load_sd_config('sd21')
model_config = config['model']
dataset_config = config['dataset']
# ...
```

#### 原来的 tool_add_control_sd21.py
```bash
# 旧方式
python tool_add_control_sd21.py

# 新方式
python tool_add_control_unified.py --sd_version sd21
```

#### 原来的 tool_add_control.py
```bash
# 旧方式
python tool_add_control.py

# 新方式
python tool_add_control_unified.py --sd_version sd15
```

## 📊 优势

1. **配置与代码分离**: 参数在配置文件中，代码更清晰
2. **易于维护**: 修改参数只需编辑配置文件
3. **统一接口**: 两个版本使用相同的脚本结构
4. **灵活扩展**: 可以轻松添加新的配置项
5. **版本管理**: 不同版本的配置分开管理

这样的设计既保持了简单性，又提供了足够的灵活性！ 