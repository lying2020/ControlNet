import json
import cv2
import numpy as np
import os

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, config=None):
        """
        初始化数据集

        Args:
            config: 配置字典，包含数据集相关参数
        """
        self.config = config or {}

        # 从配置中获取路径参数
        self.data_root = self.config.get('data_root', './training/fill50k')
        self.json_file = self.config.get('json_file', 'prompt.json')

        # 从配置中获取图像处理参数
        self.image_size = self.config.get('image_size', 512)

        # 从配置中获取归一化参数
        self.source_norm = self.config.get('source_normalization', [0, 1])
        self.target_norm = self.config.get('target_normalization', [-1, 1])

        # 加载数据
        self.data = []
        json_path = os.path.join(self.data_root, self.json_file)
        with open(json_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # 构建完整路径
        source_path = os.path.join(self.data_root, source_filename)
        target_path = os.path.join(self.data_root, target_filename)

        # 读取图像并检查是否成功
        source = cv2.imread(source_path)
        target = cv2.imread(target_path)
        
        if source is None:
            raise ValueError(f"无法读取source图像: {source_path}")
        if target is None:
            raise ValueError(f"无法读取target图像: {target_path}")

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # 调整图像尺寸到指定大小
        source = cv2.resize(source, (self.image_size, self.image_size))
        target = cv2.resize(target, (self.image_size, self.image_size))

        # 确保图像尺寸正确
        assert source.shape == (self.image_size, self.image_size, 3), f"Source图像尺寸错误: {source.shape}, 期望: ({self.image_size}, {self.image_size}, 3)"
        assert target.shape == (self.image_size, self.image_size, 3), f"Target图像尺寸错误: {target.shape}, 期望: ({self.image_size}, {self.image_size}, 3)"

        # 根据配置进行归一化
        source_min, source_max = self.source_norm
        target_min, target_max = self.target_norm

        # Normalize source images
        if source_max == 1.0:
            source = source.astype(np.float32) / 255.0
        else:
            source = (source.astype(np.float32) / 127.5) - 1.0

        # Normalize target images
        if target_max == 1.0:
            target = target.astype(np.float32) / 255.0
        else:
            target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

