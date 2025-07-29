# Stable Diffusion 1.5 vs 2.1 在ControlNet训练中的详细对比

## 📊 核心参数对比

| 参数 | SD 1.5 (cldm_v15.yaml) | SD 2.1 (cldm_v21.yaml) | 差异 |
|------|------------------------|------------------------|------|
| **模型文件大小** | 5.71 GB (control_sd15_ini.ckpt) | 6.67 GB (control_sd21_ini.ckpt) | +16.8% |
| **Context Dimension** | 768 | 1024 | +33.3% |
| **Text Encoder** | FrozenCLIPEmbedder | FrozenOpenCLIPEmbedder | 不同架构 |
| **Attention Heads** | num_heads: 8 | num_head_channels: 64 | 不同配置 |
| **Transformer** | 标准配置 | use_linear_in_transformer: True | 优化版本 |

## 🏗️ 架构差异详解

### 1. **文本编码器 (Text Encoder)**

#### SD 1.5 - FrozenCLIPEmbedder
```yaml
cond_stage_config:
  target: ldm.modules.encoders.modules.FrozenCLIPEmbedder
```
- **模型**: OpenAI CLIP ViT-L/14
- **特征维度**: 768
- **参数量**: ~123M
- **特点**: 标准CLIP模型，广泛使用

#### SD 2.1 - FrozenOpenCLIPEmbedder
```yaml
cond_stage_config:
  target: ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder
  params:
    freeze: True
    layer: "penultimate"
```
- **模型**: OpenCLIP ViT-H/14 (laion2b_s32b_b79k)
- **特征维度**: 1024
- **参数量**: ~354M
- **特点**: 更大更强的文本理解能力

### 2. **UNet架构差异**

#### SD 1.5
```yaml
unet_config:
  params:
    num_heads: 8
    context_dim: 768
    use_spatial_transformer: True
    transformer_depth: 1
```

#### SD 2.1
```yaml
unet_config:
  params:
    num_head_channels: 64
    context_dim: 1024
    use_spatial_transformer: True
    use_linear_in_transformer: True
    transformer_depth: 1
```

## ⚡ 训练性能对比

### **计算复杂度分析**

#### 参数量对比
- **SD 1.5 ControlNet**: ~1.4B 参数
- **SD 2.1 ControlNet**: ~1.7B 参数 (+21.4%)

#### 内存使用
- **SD 1.5**: 较低内存占用
- **SD 2.1**: 更高内存占用 (约+25%)

#### 训练速度 (基于你的配置)
| 指标 | SD 1.5 | SD 2.1 | 差异 |
|------|--------|--------|------|
| **每步耗时** | ~0.9s | ~1.07s | +19% |
| **每轮耗时** | ~3.1h | ~3.7h | +19% |
| **总训练时间** | ~62h (20轮) | ~74h (20轮) | +19% |

## 🎯 训练效果对比

### **优势对比**

#### SD 1.5 优势
- ✅ **训练速度更快** (约19%快)
- ✅ **内存占用更少**
- ✅ **更稳定** (经过充分验证)
- ✅ **兼容性更好** (更多预训练模型)
- ✅ **推理速度更快**

#### SD 2.1 优势
- ✅ **文本理解更强** (更大更强的CLIP)
- ✅ **生成质量更高** (理论上)
- ✅ **更好的细节表现**
- ✅ **更先进的架构**

### **适用场景**

#### 选择 SD 1.5 的情况
- 计算资源有限
- 需要快速迭代
- 对推理速度要求高
- 需要广泛的模型兼容性

#### 选择 SD 2.1 的情况
- 追求最高生成质量
- 有充足的计算资源
- 对文本理解要求高
- 长期训练项目

## 📈 训练策略建议

### **基于你的50k数据集**

#### SD 1.5 推荐配置
```python
# tutorial_train.py
batch_size = 4
learning_rate = 1e-5
max_epochs = 15-20
# 预计训练时间: ~55-74小时
```

#### SD 2.1 推荐配置
```python
# tutorial_train_sd21.py (当前配置)
batch_size = 4
learning_rate = 1e-5
max_epochs = 20
# 预计训练时间: ~74小时
```

### **优化建议**

#### 对于SD 2.1
1. **增加Batch Size** (如果GPU内存允许)
   ```python
   batch_size = 6  # 或8
   ```

2. **调整学习率**
   ```python
   learning_rate = 8e-6  # 稍微降低
   ```

3. **使用混合精度训练**
   ```python
   trainer = pl.Trainer(gpus=1, precision=16, callbacks=[logger], max_epochs=20)
   ```

## 🔍 实际训练监控

### **关键指标对比**

| 指标 | SD 1.5 预期 | SD 2.1 当前 | 说明 |
|------|-------------|-------------|------|
| **Loss收敛速度** | 较快 | 中等 | SD 2.1需要更多时间 |
| **Loss稳定性** | 高 | 中等 | SD 2.1可能波动更大 |
| **生成质量** | 良好 | 优秀 | SD 2.1理论上更好 |
| **训练稳定性** | 高 | 中等 | 需要更多监控 |

### **监控建议**

1. **Loss曲线监控**
   - SD 2.1的loss可能波动更大
   - 关注loss是否持续下降

2. **生成质量评估**
   - 每300步检查生成图像
   - 对比清晰度和细节表现

3. **内存使用监控**
   - SD 2.1需要更多GPU内存
   - 监控OOM风险

## 🎯 最终建议

### **当前情况分析**
你正在使用SD 2.1进行训练，这是一个不错的选择，因为：
- 有充足的计算资源
- 追求更好的生成质量
- 训练时间可以接受

### **优化建议**
1. **继续当前训练** - SD 2.1值得投入时间
2. **监控训练过程** - 关注loss和生成质量
3. **考虑混合精度** - 可以节省内存和加速训练
4. **准备对比实验** - 可以同时训练SD 1.5版本进行对比

### **预期结果**
- **训练时间**: 约74小时 (20轮)
- **生成质量**: 预期比SD 1.5更好
- **文本理解**: 更强的语义一致性
- **细节表现**: 更丰富的细节和纹理

## 📝 总结

SD 2.1相比SD 1.5在ControlNet训练中：
- **参数量增加**: +21.4%
- **训练时间增加**: +19%
- **内存占用增加**: +25%
- **预期质量提升**: 显著

对于你的50k数据集和去模糊任务，SD 2.1是一个合理的选择，建议继续训练并密切监控训练过程。 