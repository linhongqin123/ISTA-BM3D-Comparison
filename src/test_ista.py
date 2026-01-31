"""
测试ISTA算法的脚本
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os

from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 现在可以导入src模块
from src.ista_denoiser import ISTADenoiser, ista_denoise
from src.noise_generation import add_gaussian_noise, add_salt_pepper_noise
from src.evaluation import calculate_metrics
from src.visualization import plot_image_comparison, save_figure

def test_basic_functionality():
    """测试基本功能"""
    print("测试ISTA基本功能...")
    
    # 创建测试图像
    original = np.random.rand(128, 128) * 255
    original = original.astype(np.uint8)
    
    # 添加高斯噪声
    noisy_gaussian = add_gaussian_noise(original, sigma=25)
    
    # 测试图像域ISTA
    print("\n1. 测试图像域ISTA...")
    denoiser = ISTADenoiser(
        max_iter=30,
        lambda_=0.1,
        step_size=1.0,
        verbose=True
    )
    
    denoised_image = denoiser.denoise(noisy_gaussian, method='image')
    
    # 计算指标
    metrics = calculate_metrics(original, denoised_image, "ISTA (Image Domain)")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    
    # 测试小波域ISTA
    print("\n2. 测试小波域ISTA...")
    denoiser = ISTADenoiser(
        wavelet_type='db4',
        max_iter=30,
        lambda_=0.1,
        step_size=1.0,
        verbose=True
    )
    
    denoised_wavelet = denoiser.denoise(noisy_gaussian, method='wavelet')
    
    metrics = calculate_metrics(original, denoised_wavelet, "ISTA (Wavelet Domain)")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    
    # 绘制收敛曲线
    denoiser.plot_convergence()
    
    # 图像对比
    fig = plot_image_comparison(
        original,
        noisy_gaussian,
        [denoised_image, denoised_wavelet],
        ['Original', 'Noisy (σ=25)', 'ISTA Image', 'ISTA Wavelet']
    )
    
    plt.show()
    
    return original, noisy_gaussian, denoised_image, denoised_wavelet

def test_noise_types():
    """测试不同噪声类型"""
    print("\n测试不同噪声类型的ISTA性能...")
    
    # 加载或创建测试图像
    test_image_path = 'data/Set14/baboon.png'
    test_image = None
    
    # 尝试加载图像
    if os.path.exists(test_image_path):
        test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    
    if test_image is None:
        # 如果找不到图像，创建一个测试图像
        print(f"未找到 {test_image_path}，创建测试图像")
        test_image = np.random.rand(256, 256) * 255
        test_image = test_image.astype(np.uint8)
    else:
        print(f"成功加载图像: {test_image_path}")
        print(f"图像形状: {test_image.shape}")
    
    # 生成不同噪声
    noises = []
    
    # 高斯噪声
    gaussian_params = [10, 25, 50]
    for sigma in gaussian_params:
        noisy = add_gaussian_noise(test_image.copy(), sigma)
        noises.append((noisy, f'Gaussian σ={sigma}'))
    
    # 椒盐噪声
    sp_params = [(0.02, 0.02), (0.05, 0.05), (0.1, 0.1)]
    for salt_prob, pepper_prob in sp_params:
        noisy = add_salt_pepper_noise(test_image.copy(), salt_prob, pepper_prob)
        noises.append((noisy, f'Salt&Pepper {salt_prob+pepper_prob:.1%}'))
    
    # 测试每种噪声
    results = []
    
    for noisy, noise_name in noises:
        print(f"\n处理噪声: {noise_name}")
        
        # 使用ISTA去噪
        denoiser = ISTADenoiser(
            max_iter=50,
            lambda_=0.1,
            step_size=1.0,
            verbose=False
        )
        
        denoised = denoiser.denoise(noisy, method='wavelet')
        
        # 计算指标
        metrics = calculate_metrics(test_image, denoised, f"ISTA ({noise_name})")
        results.append((noise_name, metrics['psnr'], metrics['ssim']))
        
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
    
    # 打印汇总
    print("\n" + "="*60)
    print("噪声类型测试汇总:")
    print("="*60)
    for name, psnr, ssim in results:
        print(f"{name:20s} | PSNR: {psnr:6.2f} dB | SSIM: {ssim:.4f}")

def test_convergence_with_step_size():
    """测试步长对收敛的影响"""
    print("\n测试步长对收敛的影响...")
    
    # 创建测试图像
    original = np.random.rand(128, 128) * 255
    original = original.astype(np.uint8)
    noisy = add_gaussian_noise(original, sigma=25)
    
    # 不同步长
    step_sizes = [0.1, 0.5, 1.0, 1.5, 2.0]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, step_size in enumerate(step_sizes):
        if idx >= len(axes):
            break
        
        print(f"测试步长: {step_size}")
        
        denoiser = ISTADenoiser(
            max_iter=100,
            lambda_=0.1,
            step_size=step_size,
            verbose=False
        )
        
        denoised = denoiser.denoise(noisy, method='wavelet')
        history = denoiser.get_convergence_history()
        
        if history['residual']:
            iterations = history['iterations']
            residuals = history['residual']
            
            axes[idx].plot(iterations, residuals, 'b-', linewidth=2)
            axes[idx].set_xlabel('迭代次数')
            axes[idx].set_ylabel('残差')
            axes[idx].set_title(f'步长 = {step_size}')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_yscale('log')
        
        # 计算最终PSNR
        psnr = calculate_metrics(original, denoised, "")['psnr']
        axes[idx].text(0.05, 0.95, f'PSNR: {psnr:.2f} dB',
                      transform=axes[idx].transAxes,
                      verticalalignment='top')
    
    plt.suptitle('不同步长下的ISTA收敛行为', fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    """主测试函数"""
    print("="*60)
    print("ISTA算法测试套件")
    print("="*60)
    
    # 创建输出目录
    output_dir = Path('results/ista_tests')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试1: 基本功能
    print("\n测试1: 基本功能测试")
    test_basic_functionality()
    
    # 测试2: 噪声类型
    print("\n\n测试2: 不同噪声类型测试")
    test_noise_types()
    
    # 测试3: 收敛性分析
    print("\n\n测试3: 收敛性分析")
    test_convergence_with_step_size()
    
    print("\n" + "="*60)
    print("所有测试完成!")
    print("="*60)

if __name__ == "__main__":
    main()