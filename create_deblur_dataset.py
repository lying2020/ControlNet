import os
import json
import cv2
import numpy as np
from PIL import Image, ImageFilter
import argparse


def create_motion_blur(image, angle=45, distance=10):
    """创建运动模糊"""
    # 创建运动模糊核
    kernel = np.zeros((distance, distance))
    center = distance // 2
    
    # 计算运动方向
    angle_rad = np.radians(angle)
    dx = int(np.cos(angle_rad) * distance / 2)
    dy = int(np.sin(angle_rad) * distance / 2)
    
    # 绘制运动轨迹
    for i in range(distance):
        x = center + int(i * dx / distance)
        y = center + int(i * dy / distance)
        if 0 <= x < distance and 0 <= y < distance:
            kernel[y, x] = 1.0 / distance
    
    # 应用模糊
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def create_gaussian_blur(image, sigma=3):
    """创建高斯模糊"""
    return cv2.GaussianBlur(image, (0, 0), sigma)


def create_defocus_blur(image, radius=5):
    """创建散焦模糊"""
    # 使用PIL的散焦模糊
    pil_image = Image.fromarray(image)
    blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred)


def create_lens_blur(image, strength=0.5):
    """创建镜头模糊"""
    # 简化的镜头模糊实现
    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    blurred = cv2.filter2D(image, -1, kernel)
    
    # 混合原图和模糊图像
    result = cv2.addWeighted(image, 1 - strength, blurred, strength, 0)
    return result


def create_deblur_dataset(source_dir, output_dir, num_samples=1000, image_size=512):
    """创建去模糊数据集"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'clear_images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'blur_images'), exist_ok=True)
    
    # 获取源图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    source_images = []
    
    for ext in image_extensions:
        source_images.extend([f for f in os.listdir(source_dir) if f.lower().endswith(ext)])
    
    if not source_images:
        raise ValueError(f"No images found in {source_dir}")
    
    # 模糊类型和对应的函数
    blur_types = {
        'motion_blur': create_motion_blur,
        'gaussian_blur': create_gaussian_blur,
        'defocus_blur': create_defocus_blur,
        'lens_blur': create_lens_blur
    }
    
    dataset_info = []
    
    for i in range(num_samples):
        # 随机选择源图像
        source_image_name = np.random.choice(source_images)
        source_image_path = os.path.join(source_dir, source_image_name)
        
        # 读取图像
        image = cv2.imread(source_image_path)
        if image is None:
            continue
            
        # 转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整图像大小
        image = cv2.resize(image, (image_size, image_size))
        
        # 随机选择模糊类型
        blur_type = np.random.choice(list(blur_types.keys()))
        blur_func = blur_types[blur_type]
        
        # 创建模糊图像
        if blur_type == 'motion_blur':
            angle = np.random.uniform(0, 360)
            distance = np.random.randint(5, 20)
            blurred_image = blur_func(image, angle, distance)
        elif blur_type == 'gaussian_blur':
            sigma = np.random.uniform(1, 5)
            blurred_image = blur_func(image, sigma)
        elif blur_type == 'defocus_blur':
            radius = np.random.uniform(1, 8)
            blurred_image = blur_func(image, radius)
        elif blur_type == 'lens_blur':
            strength = np.random.uniform(0.3, 0.8)
            blurred_image = blur_func(image, strength)
        
        # 保存图像
        clear_filename = f"clear_{i:06d}.png"
        blur_filename = f"blur_{i:06d}.png"
        
        clear_path = os.path.join(output_dir, 'clear_images', clear_filename)
        blur_path = os.path.join(output_dir, 'blur_images', blur_filename)
        
        # 保存清晰图像
        cv2.imwrite(clear_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # 保存模糊图像
        cv2.imwrite(blur_path, cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR))
        
        # 添加到数据集信息
        dataset_info.append({
            'clear_image': clear_filename,
            'blur_image': blur_filename,
            'blur_type': blur_type,
            'source_image': source_image_name
        })
        
        if (i + 1) % 100 == 0:
            print(f"Created {i + 1} samples...")
    
    # 保存JSON文件
    json_path = os.path.join(output_dir, 'blur_types.json')
    with open(json_path, 'w') as f:
        for item in dataset_info:
            f.write(json.dumps(item) + '\n')
    
    print(f"Dataset created successfully!")
    print(f"Total samples: {len(dataset_info)}")
    print(f"Output directory: {output_dir}")
    print(f"JSON file: {json_path}")
    
    # 统计模糊类型分布
    blur_type_counts = {}
    for item in dataset_info:
        blur_type = item['blur_type']
        blur_type_counts[blur_type] = blur_type_counts.get(blur_type, 0) + 1
    
    print("\nBlur type distribution:")
    for blur_type, count in blur_type_counts.items():
        print(f"  {blur_type}: {count} samples")


def main():
    parser = argparse.ArgumentParser(description='Create deblur dataset')
    parser.add_argument('--source_dir', type=str, required=True,
                       help='Directory containing source images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for dataset')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to create')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size for training')
    
    args = parser.parse_args()
    
    create_deblur_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        image_size=args.image_size
    )


if __name__ == "__main__":
    main() 