"""
诊断BM3D问题
"""

import numpy as np
import cv2
import bm3d
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.noise_generation import add_gaussian_noise
from src.evaluation import calculate_psnr

print("诊断BM3D问题")
print("="*60)

# 1. 测试BM3D基本功能
print("\n1. 测试BM3D基本功能...")

# 创建简单的测试图像
np.random.seed(42)
original = np.ones((100, 100)) * 128
original = original.astype(np.uint8)

# 添加噪声
noisy = add_gaussian_noise(original.copy(), sigma=25)
print(f"原始图像: {original.shape}, 范围: [{original.min()}, {original.max()}]")
print(f"含噪图像: {noisy.shape}, 范围: [{noisy.min()}, {noisy.max()}]")

# 方法1：直接使用BM3D库
print("\n2. 直接使用BM3D库...")
try:
    # 转换为float32并归一化到[0,1]
    noisy_float = noisy.astype(np.float32) / 255.0
    
    # 设置噪声水平（相对于[0,1]范围）
    sigma = 25 / 255.0  # 注意：BM3D期望sigma在[0,1]范围内
    
    print(f"BM3D参数: sigma={sigma:.4f}")
    print(f"输入图像范围: [{noisy_float.min():.3f}, {noisy_float.max():.3f}]")
    
    # 运行BM3D
    denoised_bm3d = bm3d.bm3d(noisy_float, sigma)
    
    print(f"BM3D输出范围: [{denoised_bm3d.min():.3f}, {denoised_bm3d.max():.3f}]")
    
    # 转换回[0,255]
    denoised_bm3d = np.clip(denoised_bm3d, 0, 1)
    denoised_bm3d = (denoised_bm3d * 255).astype(np.uint8)
    
    # 计算PSNR
    psnr = calculate_psnr(original, denoised_bm3d)
    print(f"BM3D PSNR: {psnr:.2f} dB")
    
except Exception as e:
    print(f"BM3D直接调用失败: {e}")
    import traceback
    traceback.print_exc()

# 2. 测试我们的BM3D包装器
print("\n3. 测试我们的BM3D包装器...")
try:
    from src.bm3d_wrapper import BM3DDenoiser
    
    denoiser = BM3DDenoiser(sigma_psd=25, verbose=True)
    denoised = denoiser.denoise(noisy.copy())
    
    psnr = calculate_psnr(original, denoised)
    print(f"包装器BM3D PSNR: {psnr:.2f} dB")
    print(f"处理时间: {denoiser.get_processing_time():.2f} 秒")
    
except Exception as e:
    print(f"BM3D包装器失败: {e}")
    import traceback
    traceback.print_exc()

# 3. 检查噪声水平和图像范围
print("\n4. 检查噪声水平和图像范围...")

# 创建一个更明显的测试
test_image = cv2.imread('data/Set14/baboon.png', cv2.IMREAD_GRAYSCALE)
if test_image is not None:
    print(f"测试图像: {test_image.shape}, 范围: [{test_image.min()}, {test_image.max()}]")
    
    # 添加噪声
    noisy_test = add_gaussian_noise(test_image.copy(), sigma=25)
    
    # 检查像素值范围
    print(f"含噪图像范围: [{noisy_test.min()}, {noisy_test.max()}]")
    
    # 检查噪声水平
    noise = noisy_test.astype(np.float32) - test_image.astype(np.float32)
    noise_std = np.std(noise)
    print(f"实际噪声标准差: {noise_std:.2f}")
    
    # 归一化后的噪声水平
    normalized_noise_std = noise_std / 255.0
    print(f"归一化噪声标准差: {normalized_noise_std:.4f}")
    
    # 测试不同sigma值
    print("\n5. 测试不同sigma值对BM3D的影响...")
    
    sigma_values = [5, 10, 15, 20, 25, 30, 50]
    
    for sigma in sigma_values:
        print(f"\n  sigma={sigma}:")
        
        # 直接调用BM3D
        noisy_float = noisy_test.astype(np.float32) / 255.0
        sigma_normalized = sigma / 255.0
        
        try:
            denoised_float = bm3d.bm3d(noisy_float, sigma_normalized)
            denoised = (np.clip(denoised_float, 0, 1) * 255).astype(np.uint8)
            
            psnr = calculate_psnr(test_image, denoised)
            print(f"    BM3D PSNR: {psnr:.2f} dB")
        except Exception as e:
            print(f"    失败: {e}")

print("\n" + "="*60)
print("诊断完成")
print("="*60)