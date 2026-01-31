"""
简单测试脚本 - 验证基本功能
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

print("导入模块...")
try:
    from src.ista_denoiser import ISTADenoiser
    print("✓ 成功导入ISTADenoiser")
except Exception as e:
    print(f"✗ 导入ISTADenoiser失败: {e}")

try:
    from src.noise_generation import add_gaussian_noise
    print("✓ 成功导入noise_generation")
except Exception as e:
    print(f"✗ 导入noise_generation失败: {e}")

try:
    from src.evaluation import calculate_psnr
    print("✓ 成功导入evaluation")
except Exception as e:
    print(f"✗ 导入evaluation失败: {e}")

# 测试基本功能
print("\n测试基本功能...")

# 创建测试图像
test_image = np.random.rand(64, 64) * 255
test_image = test_image.astype(np.uint8)
print(f"测试图像形状: {test_image.shape}")

# 添加噪声
noisy = test_image + np.random.randn(64, 64) * 25
noisy = np.clip(noisy, 0, 255).astype(np.uint8)
print(f"含噪图像形状: {noisy.shape}")

# 创建ISTA去噪器
try:
    denoiser = ISTADenoiser(max_iter=10, lambda_=0.1, step_size=1.0, verbose=True)
    print("✓ 成功创建ISTADenoiser")
    
    # 去噪
    denoised = denoiser.denoise(noisy, method='image')
    print(f"✓ 去噪完成，形状: {denoised.shape}")
    
    # 计算PSNR
    psnr = calculate_psnr(test_image, denoised)
    print(f"✓ PSNR计算完成: {psnr:.2f} dB")
    
    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(noisy, cmap='gray')
    axes[1].set_title('Noisy')
    axes[1].axis('off')
    
    axes[2].imshow(denoised, cmap='gray')
    axes[2].set_title('Denoised')
    axes[2].axis('off')
    
    plt.suptitle(f'ISTA Test - PSNR: {psnr:.2f} dB')
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成!")