import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import torch


class DeblurDataset(Dataset):
    """
    图像去模糊数据集
    数据格式：
    - clear_images/: 清晰图片
    - blur_images/: 模糊图片  
    - blur_types.json: 包含图片对和模糊类型信息的JSON文件
    """
    
    def __init__(self, data_root, json_file, image_size=512):
        self.data_root = data_root
        self.image_size = image_size
        self.data = []
        
        # 读取JSON文件
        with open(os.path.join(data_root, json_file), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取文件名和模糊类型
        clear_filename = item['clear_image']
        blur_filename = item['blur_image']
        blur_type = item['blur_type']  # 例如: "motion_blur", "gaussian_blur", "defocus_blur"
        
        # 读取图片
        clear_path = os.path.join(self.data_root, 'clear_images', clear_filename)
        blur_path = os.path.join(self.data_root, 'blur_images', blur_filename)
        
        clear_img = cv2.imread(clear_path)
        blur_img = cv2.imread(blur_path)
        
        # BGR转RGB
        clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
        
        # 调整图片大小
        clear_img = cv2.resize(clear_img, (self.image_size, self.image_size))
        blur_img = cv2.resize(blur_img, (self.image_size, self.image_size))
        
        # 归一化
        # 清晰图片作为目标，归一化到[-1, 1]
        clear_img = (clear_img.astype(np.float32) / 127.5) - 1.0
        # 模糊图片作为控制信号，归一化到[0, 1]
        blur_img = blur_img.astype(np.float32) / 255.0
        
        # 构建提示文本
        prompt = f"a clear, sharp image, {blur_type} removed, high quality, detailed"
        
        return dict(
            jpg=clear_img,      # 目标清晰图片 (target)
            txt=prompt,         # 文本提示
            hint=blur_img,      # 控制信号：模糊图片
            blur_type=blur_type # 模糊类型信息
        )


class DeblurDatasetWithAugmentation(Dataset):
    """
    带数据增强的图像去模糊数据集
    """
    
    def __init__(self, data_root, json_file, image_size=512, augment=True):
        self.data_root = data_root
        self.image_size = image_size
        self.augment = augment
        self.data = []
        
        with open(os.path.join(data_root, json_file), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def augment_image(self, img):
        """简单的数据增强"""
        if not self.augment:
            return img
            
        # 随机水平翻转
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)
        
        # 随机亮度调整
        if np.random.random() > 0.5:
            alpha = 0.8 + np.random.random() * 0.4  # 0.8-1.2
            img = np.clip(img * alpha, 0, 255).astype(np.uint8)
        
        return img
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        clear_filename = item['clear_image']
        blur_filename = item['blur_image']
        blur_type = item['blur_type']
        
        clear_path = os.path.join(self.data_root, 'clear_images', clear_filename)
        blur_path = os.path.join(self.data_root, 'blur_images', blur_filename)
        
        clear_img = cv2.imread(clear_path)
        blur_img = cv2.imread(blur_path)
        
        clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
        
        # 数据增强
        if self.augment:
            clear_img = self.augment_image(clear_img)
            blur_img = self.augment_image(blur_img)
        
        # 调整大小
        clear_img = cv2.resize(clear_img, (self.image_size, self.image_size))
        blur_img = cv2.resize(blur_img, (self.image_size, self.image_size))
        
        # 归一化
        clear_img = (clear_img.astype(np.float32) / 127.5) - 1.0
        blur_img = blur_img.astype(np.float32) / 255.0
        
        # 构建更详细的提示文本
        blur_type_descriptions = {
            "motion_blur": "motion blur removed, sharp and clear",
            "gaussian_blur": "gaussian blur removed, crisp and detailed", 
            "defocus_blur": "defocus blur removed, in focus and sharp",
            "lens_blur": "lens blur removed, clear and focused"
        }
        
        description = blur_type_descriptions.get(blur_type, "blur removed, clear and sharp")
        prompt = f"a {description} image, high quality, detailed, professional photography"
        
        return dict(
            jpg=clear_img,
            txt=prompt,
            hint=blur_img,
            blur_type=blur_type
        ) 