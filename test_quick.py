"""
快速测试脚本 - 不依赖visualization模块
"""

import numpy as np
import cv2
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("="*60)
print("ISTA快速测试（不依赖可视化）")
print("="*60)

# 1. 测试噪声生成
print("\n1. 测试噪声生成模块...")
try:
    from src.noise_generation import add_gaussian_noise, add_salt_pepper_noise
    
    # 创建测试图像
    test_img = np.random.rand(64, 64) * 255
    test_img = test_img.astype(np.uint8)
    
    # 生成高斯噪声
    noisy_gaussian = add_gaussian_noise(test_img, sigma=25)
    print(f"✓ 高斯噪声生成成功，形状: {noisy_gaussian.shape}")
    
    # 生成椒盐噪声
    noisy_sp = add_salt_pepper_noise(test_img, salt_prob=0.05, pepper_prob=0.05)
    print(f"✓ 椒盐噪声生成成功，形状: {noisy_sp.shape}")
    
except Exception as e:
    print(f"✗ 噪声生成模块失败: {e}")

# 2. 测试评估模块
print("\n2. 测试评估模块...")
try:
    from src.evaluation import calculate_psnr, calculate_ssim
    
    # 创建测试数据
    img1 = np.random.rand(64, 64)
    img2 = img1 + np.random.randn(64, 64) * 0.1
    
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    
    print(f"✓ PSNR计算成功: {psnr:.2f} dB")
    print(f"✓ SSIM计算成功: {ssim:.4f}")
    
except Exception as e:
    print(f"✗ 评估模块失败: {e}")

# 3. 测试ISTA核心算法（不依赖可视化）
print("\n3. 测试ISTA核心算法...")
try:
    # 直接导入ISTA类，避免visualization依赖
    sys.path.insert(0, os.path.join(project_root, 'src'))
    
    # 复制ISTADenoiser的核心代码到这里进行测试
    print("正在测试ISTA核心功能...")
    
    # 创建简单的ISTA实现用于测试
    class SimpleISTA:
        @staticmethod
        def soft_threshold(x, threshold):
            """软阈值算子"""
            return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
        
        def denoise_simple(self, noisy_img, lambda_=0.1, n_iter=10, step_size=1.0):
            """简化的ISTA去噪"""
            x = noisy_img.astype(np.float32).copy()
            y = noisy_img.astype(np.float32).copy()
            
            # 归一化
            if y.max() > 1.0:
                y = y / 255.0
                x = x / 255.0
            
            for k in range(n_iter):
                # 梯度下降
                gradient = x - y
                x = x - step_size * gradient
                
                # 软阈值
                x = self.soft_threshold(x, lambda_ * step_size)
                
                # 计算残差
                if k % 5 == 0:
                    residual = np.mean(np.abs(gradient))
                    print(f"  迭代 {k+1}/{n_iter}, 残差: {residual:.6f}")
            
            # 恢复范围
            x = np.clip(x * 255, 0, 255).astype(np.uint8)
            return x
    
    # 测试简单ISTA
    simple_ista = SimpleISTA()
    
    # 创建测试图像
    test_image = np.random.rand(64, 64) * 255
    test_image = test_image.astype(np.uint8)
    noisy = test_image + np.random.randn(64, 64) * 25
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # 去噪
    denoised = simple_ista.denoise_simple(noisy, lambda_=0.1, n_iter=10)
    
    print(f"✓ ISTA去噪成功，输入形状: {noisy.shape}, 输出形状: {denoised.shape}")
    
    # 计算PSNR
    psnr = calculate_psnr(test_image, denoised)
    print(f"✓ 去噪PSNR: {psnr:.2f} dB")
    
except Exception as e:
    print(f"✗ ISTA测试失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 测试完整的ISTADenoiser（跳过可视化）
print("\n4. 测试完整的ISTADenoiser（跳过plot_convergence）...")
try:
    # 修改导入方式，避免visualization依赖
    # 直接导入ISTA类
    from src.ista_denoiser import ISTADenoiser as FullISTA
    
    # 创建去噪器（关闭verbose，避免可能的导入问题）
    denoiser = FullISTA(
        max_iter=10,
        lambda_=0.1,
        step_size=1.0,
        verbose=False  # 关闭详细输出，避免可能的导入问题
    )
    
    # 创建测试数据
    test_image = np.random.rand(64, 64) * 255
    test_image = test_image.astype(np.uint8)
    noisy = test_image + np.random.randn(64, 64) * 25
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # 使用图像域方法（最简单）
    denoised = denoiser.denoise(noisy, method='image')
    
    print(f"✓ 完整ISTADenoiser测试成功")
    print(f"  输入形状: {noisy.shape}")
    print(f"  输出形状: {denoised.shape}")
    
    # 计算指标
    psnr = calculate_psnr(test_image, denoised)
    ssim = calculate_ssim(test_image, denoised)
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")
    
    # 测试收敛历史获取
    history = denoiser.get_convergence_history()
    if history['iterations']:
        print(f"  收敛历史: {len(history['iterations'])} 次迭代")
    
except Exception as e:
    print(f"✗ 完整ISTA测试失败: {e}")

print("\n" + "="*60)
print("快速测试完成!")
print("="*60)