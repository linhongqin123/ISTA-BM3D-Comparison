"""
ISTA收敛性分析
分析步长、正则化参数和迭代次数对收敛的影响
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.noise_generation import add_gaussian_noise
from src.ista_denoiser import ISTADenoiser
from src.evaluation import calculate_psnr, calculate_ssim

def analyze_convergence():
    """分析ISTA收敛行为"""
    print("="*60)
    print("ISTA收敛性分析")
    print("="*60)
    
    # 创建测试图像
    np.random.seed(42)
    print("\n1. 创建测试图像...")
    original = np.random.rand(256, 256) * 255
    original = original.astype(np.uint8)
    
    # 添加噪声
    noisy = add_gaussian_noise(original, sigma=25)
    print(f"  原始图像形状: {original.shape}")
    print(f"  含噪图像形状: {noisy.shape}")
    
    # 创建输出目录
    output_dir = Path('results/convergence_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 实验1: 不同步长的收敛行为
    print("\n2. 不同步长的收敛行为 (λ=0.1, 最大迭代=100)")
    
    step_sizes = [0.1, 0.5, 1.0, 1.5, 2.0]
    colors = ['b', 'g', 'r', 'c', 'm']
    
    plt.figure(figsize=(12, 8))
    
    for step_size, color in zip(step_sizes, colors):
        print(f"  测试步长: {step_size}")
        
        # 创建ISTA去噪器
        denoiser = ISTADenoiser(
            max_iter=100,
            lambda_=0.1,
            step_size=step_size,
            verbose=False
        )
        
        # 去噪（使用图像域方法，更稳定）
        denoised = denoiser.denoise(noisy, method='image')
        history = denoiser.get_convergence_history()
        
        if history['residual']:
            iterations = history['iterations']
            residuals = history['residual']
            
            plt.plot(iterations, residuals, 
                    color=color, marker='o', markersize=4, 
                    linewidth=1.5, label=f'Step Size={step_size}')
            
            print(f"    最终迭代: {len(iterations)}, 最终残差: {residuals[-1]:.6f}")
    
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.title('ISTA Convergence with Different Step Sizes (λ=0.1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    
    plot_file = output_dir / 'convergence_step_sizes.png'
    plt.savefig(plot_file, dpi=150)
    print(f"  图表已保存到: {plot_file}")
    plt.close()
    
    # 实验2: 不同lambda的收敛行为
    print("\n3. 不同λ的收敛行为 (步长=1.0, 最大迭代=100)")
    
    lambdas = [0.01, 0.05, 0.1, 0.2, 0.5]
    markers = ['o', 's', '^', 'D', 'v']
    
    plt.figure(figsize=(12, 8))
    
    for lambda_, marker in zip(lambdas, markers):
        print(f"  测试λ: {lambda_}")
        
        denoiser = ISTADenoiser(
            max_iter=100,
            lambda_=lambda_,
            step_size=1.0,
            verbose=False
        )
        
        denoised = denoiser.denoise(noisy, method='image')
        history = denoiser.get_convergence_history()
        
        if history['residual']:
            iterations = history['iterations']
            residuals = history['residual']
            
            plt.plot(iterations, residuals, 
                    marker=marker, markersize=5, 
                    linewidth=1.5, label=f'λ={lambda_}')
            
            # 计算最终PSNR
            final_psnr = calculate_psnr(original, denoised)
            print(f"    最终PSNR: {final_psnr:.2f} dB, 最终残差: {residuals[-1]:.6f}")
    
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.title('ISTA Convergence with Different λ (Step Size=1.0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    
    plot_file = output_dir / 'convergence_lambdas.png'
    plt.savefig(plot_file, dpi=150)
    print(f"  图表已保存到: {plot_file}")
    plt.close()
    
    # 实验3: 迭代次数与PSNR的关系
    print("\n4. 迭代次数与最终PSNR的关系 (λ=0.1, 步长=1.0)")
    
    iteration_points = [10, 20, 30, 40, 50, 75, 100]
    psnr_values = []
    ssim_values = []
    
    for n_iter in iteration_points:
        print(f"  测试迭代次数: {n_iter}")
        
        denoiser = ISTADenoiser(
            max_iter=n_iter,
            lambda_=0.1,
            step_size=1.0,
            verbose=False
        )
        
        denoised = denoiser.denoise(noisy, method='image')
        
        # 计算指标
        psnr = calculate_psnr(original, denoised)
        ssim = calculate_ssim(original, denoised)
        
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        
        print(f"    PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    
    # 绘制迭代次数与PSNR的关系
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(iteration_points, psnr_values, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Final PSNR vs Iteration Count')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(iteration_points, ssim_values, 'rs-', linewidth=2, markersize=8)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('SSIM')
    ax2.set_title('Final SSIM vs Iteration Count')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = output_dir / 'iteration_vs_quality.png'
    plt.savefig(plot_file, dpi=150)
    print(f"  图表已保存到: {plot_file}")
    plt.close()
    
    # 实验4: 步长与最终PSNR的关系
    print("\n5. 步长与最终PSNR的关系 (λ=0.1, 迭代=50)")
    
    step_sizes_detailed = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.8, 2.0]
    psnr_by_step = []
    residuals_by_step = []
    
    for step_size in step_sizes_detailed:
        print(f"  测试步长: {step_size}")
        
        denoiser = ISTADenoiser(
            max_iter=50,
            lambda_=0.1,
            step_size=step_size,
            verbose=False
        )
        
        denoised = denoiser.denoise(noisy, method='image')
        history = denoiser.get_convergence_history()
        
        psnr = calculate_psnr(original, denoised)
        final_residual = history['residual'][-1] if history['residual'] else 0
        
        psnr_by_step.append(psnr)
        residuals_by_step.append(final_residual)
        
        print(f"    PSNR: {psnr:.2f} dB, 最终残差: {final_residual:.6f}")
    
    # 绘制步长与PSNR的关系
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(step_sizes_detailed, psnr_by_step, 'go-', linewidth=2, markersize=8)
    ax1.set_xlabel('Step Size')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Final PSNR vs Step Size (λ=0.1, Iter=50)')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(step_sizes_detailed, residuals_by_step, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Step Size')
    ax2.set_ylabel('Final Residual')
    ax2.set_title('Final Residual vs Step Size (λ=0.1, Iter=50)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = output_dir / 'stepsize_vs_performance.png'
    plt.savefig(plot_file, dpi=150)
    print(f"  图表已保存到: {plot_file}")
    plt.close()
    
    # 实验5: 完整收敛过程展示（最佳参数）
    print("\n6. 完整收敛过程展示 (λ=0.1, 步长=1.0, 迭代=100)")
    
    denoiser = ISTADenoiser(
        max_iter=100,
        lambda_=0.1,
        step_size=1.0,
        verbose=False
    )
    
    denoised = denoiser.denoise(noisy, method='image')
    history = denoiser.get_convergence_history()
    
    # 计算每次迭代的PSNR（模拟）
    print("  计算收敛过程中的PSNR变化...")
    
    # 为了计算每次迭代的PSNR，我们需要保存每次迭代的结果
    # 这里我们模拟这个计算过程
    simulated_psnr = []
    y = noisy.astype(np.float32) / 255.0
    x = y.copy()
    
    for k in range(100):
        x_old = x.copy()
        
        # 梯度下降
        gradient = x - y
        x = x - 1.0 * gradient
        
        # 软阈值
        x = np.sign(x) * np.maximum(np.abs(x) - 0.1 * 1.0, 0)
        
        # 计算当前PSNR
        current_denoised = np.clip(x * 255, 0, 255).astype(np.uint8)
        psnr = calculate_psnr(original, current_denoised)
        simulated_psnr.append(psnr)
        
        # 检查收敛
        residual = np.linalg.norm(x - x_old) / (np.linalg.norm(x_old) + 1e-10)
        if residual < 1e-6:
            break
    
    # 绘制完整收敛过程
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 残差收敛
    axes[0, 0].plot(history['iterations'], history['residual'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].set_title('Residual Convergence')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 2. PSNR收敛
    axes[0, 1].plot(range(1, len(simulated_psnr)+1), simulated_psnr, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].set_title('PSNR Convergence')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 目标函数收敛（如果有）
    if history['objective']:
        axes[1, 0].plot(history['iterations'], history['objective'], 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Objective Value')
        axes[1, 0].set_title('Objective Function Convergence')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 收敛速度（残差对数下降率）
    if len(history['residual']) > 1:
        log_residuals = np.log(history['residual'])
        convergence_rate = np.diff(log_residuals)
        
        axes[1, 1].plot(range(2, len(history['residual'])+1), convergence_rate, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Log Residual Difference')
        axes[1, 1].set_title('Convergence Rate')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.suptitle('Complete ISTA Convergence Analysis (λ=0.1, Step=1.0)', fontsize=14)
    plt.tight_layout()
    
    plot_file = output_dir / 'complete_convergence_analysis.png'
    plt.savefig(plot_file, dpi=150)
    print(f"  图表已保存到: {plot_file}")
    plt.close()
    
    # 保存分析结果
    # 为了找到最佳步长，我们需要重新计算一些值
    print("\n7. 寻找最佳参数...")
    
    # 测试不同参数组合
    param_results = []
    
    for lambda_test in [0.01, 0.05, 0.1, 0.2]:
        for step_test in [0.5, 1.0, 1.5]:
            print(f"  测试 λ={lambda_test}, 步长={step_test}")
            
            denoiser = ISTADenoiser(
                max_iter=50,
                lambda_=lambda_test,
                step_size=step_test,
                verbose=False
            )
            
            denoised_test = denoiser.denoise(noisy, method='image')
            history_test = denoiser.get_convergence_history()
            
            psnr_test = calculate_psnr(original, denoised_test)
            ssim_test = calculate_ssim(original, denoised_test)
            
            param_results.append({
                'lambda': lambda_test,
                'step_size': step_test,
                'psnr': psnr_test,
                'ssim': ssim_test,
                'iterations': len(history_test['iterations']),
                'final_residual': history_test['residual'][-1] if history_test['residual'] else 0
            })
            
            print(f"    PSNR: {psnr_test:.2f} dB, 迭代: {len(history_test['iterations'])}")
    
    # 找到最佳参数
    best_result = max(param_results, key=lambda x: x['psnr'])
    
    analysis_results = {
        'step_size_analysis': {
            'step_sizes': step_sizes,
            'optimal_step_size': best_result['step_size'],
            'max_psnr': best_result['psnr']
        },
        'lambda_analysis': {
            'lambdas': lambdas,
            'optimal_lambda': best_result['lambda'],
            'max_psnr': best_result['psnr']
        },
        'iteration_analysis': {
            'iterations': iteration_points,
            'optimal_iterations': 50,  # 我们测试的固定值
            'psnr_values': psnr_values,
            'ssim_values': ssim_values
        },
        'convergence_summary': {
            'final_psnr': calculate_psnr(original, denoised),
            'final_ssim': calculate_ssim(original, denoised),
            'total_iterations': len(history['iterations']),
            'final_residual': history['residual'][-1] if history['residual'] else 0
        },
        'best_parameters': best_result
    }
    
    import json
    results_file = output_dir / 'convergence_analysis.json'
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n分析结果已保存到: {results_file}")
    
    # 打印总结
    print("\n" + "="*60)
    print("收敛性分析总结")
    print("="*60)
    
    print(f"\n最佳参数配置:")
    print(f"  步长: {best_result['step_size']}")
    print(f"  λ: {best_result['lambda']}")
    print(f"  迭代次数: {best_result['iterations']}")
    
    print(f"\n最佳参数性能:")
    print(f"  PSNR: {best_result['psnr']:.2f} dB")
    print(f"  SSIM: {best_result['ssim']:.4f}")
    print(f"  实际迭代次数: {best_result['iterations']}")
    print(f"  最终残差: {best_result['final_residual']:.6f}")
    
    print(f"\n所有结果已保存到: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    analyze_convergence()