from tutorial_dataset import MyDataset
from torch.utils.data import DataLoader
import torch
import numpy as np

print("=" * 60)
print("ControlNet Dataloader 详细结构分析")
print("=" * 60)

dataset = MyDataset()
print(f"数据集大小: {len(dataset)}")

batch_size = 4
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

print(f"\n批次大小: {batch_size}")
print("=" * 60)

# 获取第一个批次
for batch in dataloader:
    print("DataLoader 输出的批次结构:")
    print("-" * 40)
    
    for key, value in batch.items():
        print(f"\n键名: '{key}'")
        print(f"类型: {type(value)}")
        
        if isinstance(value, torch.Tensor):
            print(f"张量形状: {value.shape}")
            print(f"张量类型: {value.dtype}")
            print(f"数值范围: [{value.min().item():.3f}, {value.max().item():.3f}]")
            print(f"设备: {value.device}")
            
            # 如果是图像数据，显示更多信息
            if len(value.shape) == 4:  # [batch, height, width, channels]
                print(f"批次维度: {value.shape[0]} 个样本")
                print(f"图像尺寸: {value.shape[1]}x{value.shape[2]}")
                print(f"通道数: {value.shape[3]}")
                
        elif isinstance(value, list):
            print(f"列表长度: {len(value)}")
            print("列表内容:")
            for i, item in enumerate(value):
                print(f"  [{i}]: {item}")
                
        else:
            print(f"值: {value}")
    
    break

print("\n" + "=" * 60)
print("单个样本详细分析")
print("=" * 60)

# 分析单个样本
item = dataset[1234]
print(f"样本索引: 1234")

print("\n单个样本结构:")
print("-" * 40)

for key, value in item.items():
    print(f"\n键名: '{key}'")
    print(f"类型: {type(value)}")
    
    if isinstance(value, np.ndarray):
        print(f"数组形状: {value.shape}")
        print(f"数组类型: {value.dtype}")
        print(f"数值范围: [{value.min():.3f}, {value.max():.3f}]")
        
        if len(value.shape) == 3:  # [height, width, channels]
            print(f"图像尺寸: {value.shape[0]}x{value.shape[1]}")
            print(f"通道数: {value.shape[2]}")
            
    elif isinstance(value, str):
        print(f"字符串长度: {len(value)}")
        print(f"内容: '{value}'")
        
    else:
        print(f"值: {value}")

print("\n" + "=" * 60)
print("数据用途说明")
print("=" * 60)

print("""
ControlNet 数据说明:

1. 'jpg' (目标图像):
   - 形状: [batch_size, height, width, 3] 或 [height, width, 3]
   - 范围: [-1, 1] (归一化到扩散模型所需范围)
   - 用途: 作为扩散模型的目标图像，模型学习生成这样的图像

2. 'txt' (文本提示):
   - 类型: str 或 list[str]
   - 内容: 描述目标图像的文本
   - 用途: 通过CLIP编码器转换为文本特征，作为条件控制

3. 'hint' (控制信号):
   - 形状: [batch_size, height, width, 3] 或 [height, width, 3]
   - 范围: [0, 1] (归一化到控制信号范围)
   - 用途: 作为ControlNet的输入，提供额外的控制信息

在训练过程中:
- 'jpg' 被编码到潜在空间作为扩散目标
- 'txt' 被CLIP编码为文本条件
- 'hint' 被ControlNet处理为控制特征
- 所有条件在扩散过程中融合，指导图像生成
""")

print("=" * 60)
