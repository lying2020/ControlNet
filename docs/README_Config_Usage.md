# ControlNet 配置系统使用说明

## 📁 配置文件结构

所有配置都集中在 `config/train/` 目录下：

```
config/train/
├── sd15_config.yaml    # SD 1.5 配置
└── sd21_config.yaml    # SD 2.1 配置
```

## 🎯 主要改进

### 1. 数据集参数配置化
- **数据路径**: `data_root`, `json_file`
- **归一化参数**: `source_normalization`, `target_normalization`
- **加载参数**: `batch_size`, `num_workers`, `shuffle`, `pin_memory`

### 2. 模型路径配置化
- **SD模型路径**: `sd_model_path`
- **ControlNet配置**: `config_path`
- **输出路径**: `resume_path`

## 🚀 使用方法

### 1. 测试数据集
```bash
# 测试SD 1.5数据集
python tutorial_dataset_test.py --sd_version sd15 --batch_size 4

# 测试SD 2.1数据集
python tutorial_dataset_test.py --sd_version sd21 --batch_size 4
```

### 2. 训练模型
```bash
# 训练SD 1.5
python tutorial_train.py

# 训练SD 2.1
python tutorial_train_sd21.py

# 使用统一训练脚本
python train.py --sd_version sd15
python train.py --sd_version sd21
```

## ⚙️ 配置参数说明

### 数据集配置
```yaml
dataset:
  data_root: './training/fill50k'        # 数据根目录
  json_file: 'prompt.json'               # 数据索引文件
  source_normalization: [0, 1]           # 源图像归一化范围
  target_normalization: [-1, 1]          # 目标图像归一化范围
  batch_size: 8                          # 批次大小
  num_workers: 4                         # 数据加载线程数
  shuffle: true                          # 是否打乱数据
  pin_memory: true                       # 是否使用内存固定
```

### 模型配置
```yaml
model:
  config_path: './models/cldm_v15.yaml'  # ControlNet配置文件
  resume_path: './models/control_sd15_ini.ckpt'  # 输出路径
  sd_model_path: './models/stable-diffusion/v1-5-pruned.ckpt'  # SD模型路径
  learning_rate: 8e-6                    # 学习率
  sd_locked: true                        # 是否锁定SD参数
  only_mid_control: false                # 是否只训练中间层
```

### 训练配置
```yaml
training:
  max_epochs: 10                         # 最大训练轮数
  precision: 32                          # 精度 (16/32)
  gpus: 1                                # GPU数量
  logger_freq: 300                       # 日志频率
  log_every_n_steps: 10                  # 每N步记录一次
  val_check_interval: 1000               # 验证间隔
  accumulate_grad_batches: 1             # 梯度累积
  gradient_clip_val: 1.0                 # 梯度裁剪
```

## 🔧 自定义配置

### 修改数据集路径
编辑配置文件中的 `dataset.data_root` 和 `dataset.json_file`

### 修改归一化方式
```yaml
# 标准归一化 (推荐)
source_normalization: [0, 1]     # 源图像: [0, 1]
target_normalization: [-1, 1]    # 目标图像: [-1, 1]

# 其他归一化方式
source_normalization: [-1, 1]    # 源图像: [-1, 1]
target_normalization: [0, 1]     # 目标图像: [0, 1]
```

### 调整训练参数
```yaml
# 针对RTX 4090优化
batch_size: 12                    # 增大批次
precision: 16                     # 混合精度
num_workers: 8                    # 更多线程
```

## 📊 配置差异对比

| 参数 | SD 1.5 | SD 2.1 |
|------|--------|--------|
| context_dim | 768 | 1024 |
| precision | 32 | 16 |
| config_path | cldm_v15.yaml | cldm_v21.yaml |
| sd_model_path | v1-5-pruned.ckpt | v2-1_512-ema-pruned.ckpt |

## 🎉 优势

1. **配置集中**: 所有参数都在YAML文件中
2. **易于修改**: 无需修改代码即可调整参数
3. **版本管理**: 不同SD版本使用不同配置
4. **向后兼容**: 保持原有使用习惯
5. **灵活扩展**: 易于添加新的配置项 