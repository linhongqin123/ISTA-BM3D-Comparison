"""
快速测试修复的BM3D
"""

import numpy as np
import cv2
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("="*60)
print("测试修复的BM3D")
print("="*60)

# 1. 创建测试图像
print("\n1. 创建测试图像...")
np.random.seed(42)
original = np.random.rand(256, 256) * 255
original = original.astype(np.uint8)

# 添加高斯噪声
from src.noise_generation import add_gaussian_noise
noisy = add_gaussian_noise(original.copy(), sigma=25)

print(f"原始图像: {original.shape}, 范围: [{original.min()}, {original.max()}]")
print(f"含噪图像: {noisy.shape}, 范围: [{noisy.min()}, {noisy.max()}]")

# 2. 测试直接使用BM3D库
print("\n2. 测试直接使用BM3D库...")

import bm3d
from src.evaluation import calculate_psnr, calculate_ssim

# 转换为float32并归一化到[0,1]
noisy_float = noisy.astype(np.float32) / 255.0
sigma_normalized = 25 / 255.0  # 注意：sigma需要归一化

print(f"BM3D输入范围: [{noisy_float.min():.3f}, {noisy_float.max():.3f}]")
print(f"BM3D sigma: {sigma_normalized:.4f}")

# 运行BM3D
denoised_float = bm3d.bm3d(noisy_float, sigma_normalized)

print(f"BM3D输出范围: [{denoised_float.min():.3f}, {denoised_float.max():.3f}]")

# 转换回[0,255]
denoised = (np.clip(denoised_float, 0, 1) * 255).astype(np.uint8)

# 计算指标
psnr = calculate_psnr(original, denoised)
ssim = calculate_ssim(original, denoised)

print(f"直接BM3D: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")

# 3. 测试修复的BM3D包装器
print("\n3. 测试修复的BM3D包装器...")

try:
    # 动态定义修复的BM3D类
    class SimpleBM3D:
        def __init__(self, sigma_psd=25, verbose=True):
            self.sigma_normalized = sigma_psd / 255.0
            self.verbose = verbose
        
        def denoise(self, noisy_image):
            # 转换为float32并归一化
            noisy_float = noisy_image.astype(np.float32) / 255.0
            
            if self.verbose:
                print(f"修复BM3D: sigma={self.sigma_normalized:.4f}")
            
            # 调用BM3D
            denoised_float = bm3d.bm3d(noisy_float, self.sigma_normalized)
            
            # 转换回[0,255]
            denoised = (np.clip(denoised_float, 0, 1) * 255).astype(np.uint8)
            return denoised
    
    # 测试
    denoiser = SimpleBM3D(sigma_psd=25, verbose=True)
    denoised_fixed = denoiser.denoise(noisy)
    
    psnr_fixed = calculate_psnr(original, denoised_fixed)
    ssim_fixed = calculate_ssim(original, denoised_fixed)
    
    print(f"修复BM3D: PSNR={psnr_fixed:.2f} dB, SSIM={ssim_fixed:.4f}")
    
except Exception as e:
    print(f"修复BM3D失败: {e}")

# 4. 测试不同sigma值
print("\n4. 测试不同sigma值...")

sigma_values = [10, 25, 50]
results = []

for sigma in sigma_values:
    print(f"\n  sigma={sigma}:")
    
    # 添加噪声
    noisy_test = add_gaussian_noise(original.copy(), sigma=sigma)
    
    # 直接BM3D
    noisy_float = noisy_test.astype(np.float32) / 255.0
    sigma_norm = sigma / 255.0
    
    denoised_float = bm3d.bm3d(noisy_float, sigma_norm)
    denoised_test = (np.clip(denoised_float, 0, 1) * 255).astype(np.uint8)
    
    psnr_test = calculate_psnr(original, denoised_test)
    ssim_test = calculate_ssim(original, denoised_test)
    
    print(f"    BM3D: PSNR={psnr_test:.2f} dB, SSIM={ssim_test:.4f}")
    
    results.append({
        'sigma': sigma,
        'psnr': psnr_test,
        'ssim': ssim_test
    })

print("\n" + "="*60)
print("测试完成!")
print("="*60)

# 总结
print("\n总结:")
for r in results:
    print(f"sigma={r['sigma']}: PSNR={r['psnr']:.2f} dB, SSIM={r['ssim']:.4f}")