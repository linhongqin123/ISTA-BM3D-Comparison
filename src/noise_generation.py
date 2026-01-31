"""
噪声生成模块
支持高斯噪声和椒盐噪声
"""

import numpy as np
import cv2

def add_gaussian_noise(image, sigma=25):
    """
    添加高斯噪声
    
    参数:
        image: 输入图像 (0-255范围)
        sigma: 噪声标准差
        
    返回:
        noisy_image: 含噪图像
    """
    if image.max() <= 1.0:  # 归一化图像
        image = image * 255.0
    
    image = image.astype(np.float32)
    noise = np.random.randn(*image.shape) * sigma
    noisy = image + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return noisy

def add_salt_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    """
    添加椒盐噪声
    
    参数:
        image: 输入图像
        salt_prob: 盐噪声概率
        pepper_prob: 椒噪声概率
        
    返回:
        noisy_image: 含噪图像
    """
    if image.max() <= 1.0:  # 归一化图像
        image = image * 255.0
    
    image = image.astype(np.uint8)
    noisy = image.copy()
    
    # 添加盐噪声（白色）
    salt_mask = np.random.random(image.shape) < salt_prob
    noisy[salt_mask] = 255
    
    # 添加椒噪声（黑色）
    pepper_mask = np.random.random(image.shape) < pepper_prob
    noisy[pepper_mask] = 0
    
    return noisy

def add_mixed_noise(image, gaussian_sigma=0, salt_prob=0, pepper_prob=0):
    """
    添加混合噪声
    
    参数:
        image: 输入图像
        gaussian_sigma: 高斯噪声标准差
        salt_prob: 盐噪声概率
        pepper_prob: 椒噪声概率
        
    返回:
        noisy_image: 含噪图像
    """
    noisy = image.copy()
    
    if gaussian_sigma > 0:
        noisy = add_gaussian_noise(noisy, gaussian_sigma)
    
    if salt_prob > 0 or pepper_prob > 0:
        noisy = add_salt_pepper_noise(noisy, salt_prob, pepper_prob)
    
    return noisy

def calculate_noise_level(original, noisy):
    """
    计算噪声水平
    
    参数:
        original: 原始图像
        noisy: 含噪图像
        
    返回:
        noise_level: 噪声水平（标准差）
    """
    if original.dtype != np.float32:
        original = original.astype(np.float32)
    if noisy.dtype != np.float32:
        noisy = noisy.astype(np.float32)
    
    noise = noisy - original
    noise_level = np.std(noise)
    
    return noise_level

def generate_noise_experiments(original_image):
    """
    生成实验所需的噪声图像组
    
    参数:
        original_image: 原始图像
        
    返回:
        noise_experiments: 噪声实验配置字典
    """
    experiments = []
    
    # 高斯噪声实验组
    gaussian_params = [10, 25, 50]  # 噪声标准差
    
    for sigma in gaussian_params:
        noisy = add_gaussian_noise(original_image.copy(), sigma)
        experiments.append({
            'type': 'gaussian',
            'sigma': sigma,
            'noisy_image': noisy,
            'name': f'gaussian_sigma_{sigma}'
        })
    
    # 椒盐噪声实验组
    salt_pepper_params = [
        (0.02, 0.02),  # 低噪声
        (0.05, 0.05),  # 中等噪声
        (0.1, 0.1)     # 高噪声
    ]
    
    for i, (salt_prob, pepper_prob) in enumerate(salt_pepper_params):
        noisy = add_salt_pepper_noise(original_image.copy(), salt_prob, pepper_prob)
        total_prob = salt_prob + pepper_prob
        experiments.append({
            'type': 'salt_pepper',
            'salt_prob': salt_prob,
            'pepper_prob': pepper_prob,
            'noisy_image': noisy,
            'name': f'salt_pepper_{i+1}_prob_{total_prob:.2f}'
        })
    
    return experiments

if __name__ == "__main__":
    # 测试代码
    test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    # 测试高斯噪声
    gaussian_noisy = add_gaussian_noise(test_image, 25)
    print(f"高斯噪声图像形状: {gaussian_noisy.shape}")
    
    # 测试椒盐噪声
    sp_noisy = add_salt_pepper_noise(test_image, 0.05, 0.05)
    print(f"椒盐噪声图像形状: {sp_noisy.shape}")
    
    # 计算噪声水平
    noise_level = calculate_noise_level(test_image, gaussian_noisy)
    print(f"高斯噪声水平: {noise_level:.2f}")