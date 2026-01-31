"""
ISTA参数优化实验
用于确定最佳参数配置
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

from src.ista_denoiser import ISTADenoiser, parameter_sweep
from src.noise_generation import add_gaussian_noise, add_salt_pepper_noise
from src.evaluation import calculate_psnr, calculate_ssim

def optimize_ista_parameters(test_image: np.ndarray, 
                           noise_type: str = 'gaussian',
                           noise_level: float = 25):
    """
    优化ISTA参数
    
    参数:
        test_image: 测试图像
        noise_type: 噪声类型 ('gaussian' 或 'salt_pepper')
        noise_level: 噪声水平（高斯噪声的sigma或椒盐噪声的概率）
        
    返回:
        最佳参数配置
    """
    print(f"开始优化ISTA参数，噪声类型: {noise_type}, 水平: {noise_level}")
    
    # 生成含噪图像
    if noise_type == 'gaussian':
        noisy_image = add_gaussian_noise(test_image.copy(), sigma=noise_level)
    else:
        noisy_image = add_salt_pepper_noise(test_image.copy(), 
                                          salt_prob=noise_level/2, 
                                          pepper_prob=noise_level/2)
    
    # 定义参数搜索范围
    lambda_range = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    step_size_range = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    wavelet_types = ['haar', 'db4', 'sym8']
    methods = ['wavelet', 'image', 'dct']
    
    best_results = {
        'psnr': 0,
        'ssim': 0,
        'params': {},
        'method': '',
        'wavelet': ''
    }
    
    all_results = []
    
    # 遍历所有参数组合
    for wavelet in wavelet_types:
        for method in methods:
            for lambda_ in lambda_range:
                for step_size in step_size_range:
                    print(f"测试: wavelet={wavelet}, method={method}, "
                          f"lambda={lambda_:.3f}, step_size={step_size:.2f}")
                    
                    try:
                        # 创建去噪器
                        denoiser = ISTADenoiser(
                            wavelet_type=wavelet,
                            max_iter=50,
                            lambda_=lambda_,
                            step_size=step_size,
                            verbose=False
                        )
                        
                        # 去噪
                        denoised = denoiser.denoise(noisy_image, method=method)
                        
                        # 计算指标
                        psnr = calculate_psnr(test_image, denoised)
                        ssim = calculate_ssim(test_image, denoised)
                        
                        # 记录结果
                        result = {
                            'wavelet': wavelet,
                            'method': method,
                            'lambda': lambda_,
                            'step_size': step_size,
                            'psnr': psnr,
                            'ssim': ssim,
                            'iterations': len(denoiser.convergence_history['iterations'])
                        }
                        all_results.append(result)
                        
                        print(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
                        
                        # 更新最佳结果
                        if psnr > best_results['psnr']:
                            best_results['psnr'] = psnr
                            best_results['ssim'] = ssim
                            best_results['params'] = {
                                'lambda': lambda_,
                                'step_size': step_size
                            }
                            best_results['method'] = method
                            best_results['wavelet'] = wavelet
                            
                    except Exception as e:
                        print(f"  错误: {e}")
                        continue
    
    return best_results, all_results

def convergence_analysis(test_image: np.ndarray,
                        noisy_image: np.ndarray,
                        lambda_: float = 0.1,
                        step_size_range: list = None):
    """
    收敛性分析
    
    参数:
        test_image: 原始图像
        noisy_image: 含噪图像
        lambda_: 正则化参数
        step_size_range: 步长范围
    """
    if step_size_range is None:
        step_size_range = [0.1, 0.5, 1.0, 1.5, 2.0]
    
    # 设置图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, step_size in enumerate(step_size_range):
        if idx >= len(axes):
            break
            
        # 创建去噪器
        denoiser = ISTADenoiser(
            max_iter=100,
            lambda_=lambda_,
            step_size=step_size,
            verbose=False
        )
        
        # 去噪并记录收敛历史
        denoised = denoiser.denoise(noisy_image, method='wavelet')
        history = denoiser.get_convergence_history()
        
        # 计算每次迭代的PSNR
        psnr_history = []
        for i in range(len(history['iterations'])):
            # 这里简化处理，实际应该保存每次迭代的中间结果
            pass
        
        # 绘制收敛曲线
        if history['residual']:
            iterations = history['iterations']
            residuals = history['residual']
            
            axes[idx].plot(iterations, residuals, 'b-', linewidth=2)
            axes[idx].set_xlabel('迭代次数')
            axes[idx].set_ylabel('残差')
            axes[idx].set_title(f'步长 = {step_size}')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_yscale('log')
    
    plt.suptitle(f'ISTA收敛性分析 (λ={lambda_})', fontsize=16)
    plt.tight_layout()
    plt.show()

def find_optimal_iterations(test_image: np.ndarray,
                          noisy_image: np.ndarray,
                          lambda_: float = 0.1,
                          step_size: float = 1.0,
                          max_iter: int = 200):
    """
    寻找最佳迭代次数
    
    参数:
        test_image: 原始图像
        noisy_image: 含噪图像
        lambda_: 正则化参数
        step_size: 步长
        max_iter: 最大迭代次数
    """
    print(f"寻找最佳迭代次数...")
    print(f"参数: lambda={lambda_}, step_size={step_size}")
    
    # 存储不同迭代次数的结果
    results = []
    
    iteration_points = [10, 20, 30, 40, 50, 75, 100, 150, 200]
    
    for n_iter in iteration_points:
        print(f"测试迭代次数: {n_iter}")
        
        # 创建去噪器
        denoiser = ISTADenoiser(
            max_iter=n_iter,
            lambda_=lambda_,
            step_size=step_size,
            verbose=False
        )
        
        # 去噪
        denoised = denoiser.denoise(noisy_image, method='wavelet')
        
        # 计算指标
        psnr = calculate_psnr(test_image, denoised)
        ssim = calculate_ssim(test_image, denoised)
        
        results.append({
            'iterations': n_iter,
            'psnr': psnr,
            'ssim': ssim,
            'actual_iterations': len(denoiser.convergence_history['iterations'])
        })
        
        print(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    
    # 绘制结果
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    iterations = [r['iterations'] for r in results]
    psnrs = [r['psnr'] for r in results]
    ssims = [r['ssim'] for r in results]
    
    # PSNR曲线
    ax1.plot(iterations, psnrs, 'b-o', linewidth=2, markersize=8, label='PSNR')
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('PSNR (dB)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    # SSIM曲线（次坐标轴）
    ax2 = ax1.twinx()
    ax2.plot(iterations, ssims, 'r-s', linewidth=2, markersize=8, label='SSIM')
    ax2.set_ylabel('SSIM', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('ISTA性能 vs 迭代次数', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 找到最佳迭代次数
    best_result = max(results, key=lambda x: x['psnr'])
    print(f"\n最佳迭代次数: {best_result['iterations']}")
    print(f"最佳PSNR: {best_result['psnr']:.2f} dB")
    print(f"最佳SSIM: {best_result['ssim']:.4f}")
    
    return results, best_result

def main():
    """主函数"""
    print("ISTA参数优化实验")
    
    # 加载测试图像
    # 这里使用示例图像，实际应使用Set14中的图像
    test_image = cv2.imread('data/Set14/baboon.png', cv2.IMREAD_GRAYSCALE)
    if test_image is None:
        # 创建测试图像
        test_image = np.random.rand(256, 256) * 255
        test_image = test_image.astype(np.uint8)
    
    print(f"测试图像形状: {test_image.shape}")
    
    # 实验1: 高斯噪声参数优化
    print("\n" + "="*60)
    print("实验1: 高斯噪声参数优化")
    print("="*60)
    
    noisy_gaussian = add_gaussian_noise(test_image.copy(), sigma=25)
    
    best_params, all_results = optimize_ista_parameters(
        test_image, noise_type='gaussian', noise_level=25
    )
    
    print(f"\n高斯噪声最佳参数:")
    print(f"  方法: {best_params['method']}")
    print(f"  小波: {best_params['wavelet']}")
    print(f"  lambda: {best_params['params']['lambda']:.3f}")
    print(f"  步长: {best_params['params']['step_size']:.2f}")
    print(f"  最佳PSNR: {best_params['psnr']:.2f} dB")
    print(f"  最佳SSIM: {best_params['ssim']:.4f}")
    
    # 实验2: 椒盐噪声参数优化
    print("\n" + "="*60)
    print("实验2: 椒盐噪声参数优化")
    print("="*60)
    
    noisy_sp = add_salt_pepper_noise(test_image.copy(), salt_prob=0.05, pepper_prob=0.05)
    
    best_params_sp, _ = optimize_ista_parameters(
        test_image, noise_type='salt_pepper', noise_level=0.1
    )
    
    print(f"\n椒盐噪声最佳参数:")
    print(f"  方法: {best_params_sp['method']}")
    print(f"  小波: {best_params_sp['wavelet']}")
    print(f"  lambda: {best_params_sp['params']['lambda']:.3f}")
    print(f"  步长: {best_params_sp['params']['step_size']:.2f}")
    print(f"  最佳PSNR: {best_params_sp['psnr']:.2f} dB")
    print(f"  最佳SSIM: {best_params_sp['ssim']:.4f}")
    
    # 实验3: 收敛性分析
    print("\n" + "="*60)
    print("实验3: 收敛性分析")
    print("="*60)
    
    convergence_analysis(test_image, noisy_gaussian, lambda_=0.1)
    
    # 实验4: 最佳迭代次数分析
    print("\n" + "="*60)
    print("实验4: 最佳迭代次数分析")
    print("="*60)
    
    iteration_results, best_iter = find_optimal_iterations(
        test_image, noisy_gaussian, lambda_=0.1, step_size=1.0
    )
    
    # 保存结果
    output_dir = Path('results/parameter_optimization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_summary = {
        'gaussian_noise_best': best_params,
        'salt_pepper_noise_best': best_params_sp,
        'best_iteration': best_iter,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_dir / 'optimization_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n参数优化完成！")
    print(f"结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()