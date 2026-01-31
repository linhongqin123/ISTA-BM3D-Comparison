"""
修正的实验脚本 - 使用修复的BM3D
"""

import os
import sys
import numpy as np
import cv2
import json
import time
from pathlib import Path
import pandas as pd

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("="*70)
print("ISTA vs BM3D 修正对比实验")
print("="*70)

def load_set14_images(data_dir='data/Set14', max_images=3):
    """加载Set14数据集中的图像"""
    print(f"\n加载Set14图像...")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"错误: 数据目录不存在: {data_path}")
        return [], []
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(data_path.glob(f'*{ext}'))
        image_files.extend(data_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"警告: 在 {data_path} 中未找到图像文件")
        return [], []
    
    # 限制图像数量
    if max_images:
        image_files = image_files[:max_images]
    
    images = []
    names = []
    
    for img_path in image_files:
        try:
            # 读取图像并转为灰度
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                names.append(img_path.stem)
                print(f"  已加载: {img_path.stem} ({img.shape})")
        except Exception as e:
            print(f"  加载 {img_path} 失败: {e}")
    
    print(f"成功加载 {len(images)} 张图像")
    return images, names

def create_noisy_datasets(images, noise_configs):
    """创建含噪图像数据集"""
    from src.noise_generation import add_gaussian_noise, add_salt_pepper_noise
    
    noisy_datasets = []
    
    for img_idx, original in enumerate(images):
        for config in noise_configs:
            noise_type = config['type']
            
            if noise_type == 'gaussian':
                noisy = add_gaussian_noise(original.copy(), config['sigma'])
                noise_info = f"gaussian_sigma_{config['sigma']}"
            elif noise_type == 'salt_pepper':
                noisy = add_salt_pepper_noise(
                    original.copy(), 
                    config['salt_prob'], 
                    config['pepper_prob']
                )
                total_prob = config['salt_prob'] + config['pepper_prob']
                noise_info = f"salt_pepper_{total_prob:.2f}"
            else:
                continue
            
            noisy_datasets.append({
                'original': original,
                'noisy': noisy,
                'noise_type': noise_type,
                'noise_info': noise_info,
                'noise_params': config,
                'image_idx': img_idx
            })
    
    print(f"创建了 {len(noisy_datasets)} 组含噪图像")
    return noisy_datasets

def run_ista_experiment(noisy_datasets, lambda_=0.1, max_iter=50):
    """运行ISTA实验"""
    from src.ista_denoiser import ISTADenoiser
    from src.evaluation import calculate_psnr, calculate_ssim
    
    print(f"\n运行ISTA实验 (λ={lambda_}, 迭代={max_iter})...")
    
    results = []
    
    for i, dataset in enumerate(noisy_datasets):
        print(f"  处理 {i+1}/{len(noisy_datasets)}: {dataset['noise_info']}")
        
        start_time = time.time()
        
        # 创建ISTA去噪器
        denoiser = ISTADenoiser(
            max_iter=max_iter,
            lambda_=lambda_,
            step_size=1.0,
            verbose=False
        )
        
        # 使用图像域ISTA（稳定）
        denoised = denoiser.denoise(dataset['noisy'].copy(), method='image')
        
        # 计算时间
        elapsed = time.time() - start_time
        
        # 计算指标
        psnr = calculate_psnr(dataset['original'], denoised)
        ssim = calculate_ssim(dataset['original'], denoised)
        
        # 获取收敛信息
        history = denoiser.get_convergence_history()
        iterations = len(history['iterations']) if history['iterations'] else max_iter
        
        result = {
            'algorithm': 'ISTA',
            'image_idx': dataset['image_idx'],
            'noise_type': dataset['noise_type'],
            'noise_info': dataset['noise_info'],
            'noise_params': dataset['noise_params'],
            'psnr': psnr,
            'ssim': ssim,
            'time': elapsed,
            'iterations': iterations,
            'lambda': lambda_
        }
        
        results.append(result)
        
        print(f"    PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, 时间: {elapsed:.2f}秒")
    
    return results

def run_bm3d_experiment_fixed(noisy_datasets):
    """运行修复的BM3D实验"""
    try:
        # 使用修复的BM3D
        from bm3d_wrapper_fixed import BM3DDenoiserFixed
        from src.evaluation import calculate_psnr, calculate_ssim
    except ImportError:
        print("创建修复的BM3D包装器...")
        # 动态创建修复的BM3D包装器
        import bm3d
        import numpy as np
        
        class BM3DDenoiserFixed:
            def __init__(self, sigma_psd=25, verbose=False):
                self.sigma_input = sigma_psd
                self.sigma_normalized = sigma_psd / 255.0
                self.verbose = verbose
            
            def denoise(self, noisy_image):
                # 转换为float32并归一化
                noisy_float = noisy_image.astype(np.float32) / 255.0
                
                if self.verbose:
                    print(f"BM3D: sigma={self.sigma_normalized:.4f}")
                
                # 调用BM3D
                denoised_float = bm3d.bm3d(noisy_float, self.sigma_normalized)
                
                # 转换回[0,255]
                denoised = (np.clip(denoised_float, 0, 1) * 255).astype(np.uint8)
                return denoised
        
        from src.evaluation import calculate_psnr, calculate_ssim
    
    print("\n运行修复的BM3D实验...")
    
    results = []
    
    for i, dataset in enumerate(noisy_datasets):
        print(f"  处理 {i+1}/{len(noisy_datasets)}: {dataset['noise_info']}")
        
        # 设置BM3D噪声参数
        if dataset['noise_type'] == 'gaussian':
            sigma_psd = dataset['noise_params']['sigma']
        else:
            # 对于椒盐噪声，使用经验值
            sigma_psd = 25
        
        start_time = time.time()
        
        # 创建修复的BM3D去噪器
        denoiser = BM3DDenoiserFixed(sigma_psd=sigma_psd, verbose=False)
        
        # 去噪
        denoised = denoiser.denoise(dataset['noisy'].copy())
        
        # 计算时间
        elapsed = time.time() - start_time
        
        # 计算指标
        psnr = calculate_psnr(dataset['original'], denoised)
        ssim = calculate_ssim(dataset['original'], denoised)
        
        result = {
            'algorithm': 'BM3D_fixed',
            'image_idx': dataset['image_idx'],
            'noise_type': dataset['noise_type'],
            'noise_info': dataset['noise_info'],
            'noise_params': dataset['noise_params'],
            'psnr': psnr,
            'ssim': ssim,
            'time': elapsed,
            'sigma_psd': sigma_psd
        }
        
        results.append(result)
        
        print(f"    PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, 时间: {elapsed:.2f}秒")
    
    return results

def analyze_and_save_results(ista_results, bm3d_results, output_dir):
    """分析并保存结果"""
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 合并结果
    all_results = ista_results + bm3d_results
    
    # 转换为DataFrame
    df = pd.DataFrame(all_results)
    
    # 保存原始数据
    results_file = output_dir / 'results_fixed.csv'
    df.to_csv(results_file, index=False)
    print(f"\n原始结果已保存到: {results_file}")
    
    # 按算法和噪声类型分组统计
    print("\n" + "="*70)
    print("实验结果统计")
    print("="*70)
    
    for algorithm in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algorithm]
        print(f"\n{algorithm}:")
        print(f"  平均PSNR: {algo_df['psnr'].mean():.2f} ± {algo_df['psnr'].std():.2f} dB")
        print(f"  平均SSIM: {algo_df['ssim'].mean():.4f} ± {algo_df['ssim'].std():.4f}")
        print(f"  平均时间: {algo_df['time'].mean():.2f} ± {algo_df['time'].std():.2f} 秒")
        
        # 按噪声类型细分
        for noise_type in ['gaussian', 'salt_pepper']:
            type_df = algo_df[algo_df['noise_type'] == noise_type]
            if len(type_df) > 0:
                print(f"  {noise_type}: {len(type_df)}组，平均PSNR: {type_df['psnr'].mean():.2f} dB")
    
    # 保存分析结果
    analysis = {}
    for algorithm in df['algorithm'].unique():
        analysis[algorithm] = {}
        algo_df = df[df['algorithm'] == algorithm]
        
        for noise_type in ['gaussian', 'salt_pepper']:
            type_df = algo_df[algo_df['noise_type'] == noise_type]
            if len(type_df) > 0:
                analysis[algorithm][noise_type] = {
                    'count': len(type_df),
                    'avg_psnr': float(type_df['psnr'].mean()),
                    'avg_ssim': float(type_df['ssim'].mean()),
                    'avg_time': float(type_df['time'].mean())
                }
    
    analysis_file = output_dir / 'analysis_fixed.json'
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n分析结果已保存到: {analysis_file}")
    
    return df, analysis

def create_comparison_charts(df, output_dir):
    """创建对比图表"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        output_dir = Path(output_dir)
        
        # 1. PSNR对比（高斯噪声）
        gaussian_df = df[df['noise_type'] == 'gaussian']
        
        if len(gaussian_df) > 0:
            plt.figure(figsize=(10, 6))
            
            # 按算法和sigma分组
            algorithms = gaussian_df['algorithm'].unique()
            
            for algo in algorithms:
                algo_df = gaussian_df[gaussian_df['algorithm'] == algo]
                
                # 提取sigma和平均PSNR
                sigmas = []
                avg_psnrs = []
                
                # 按sigma分组
                for sigma in [10, 25, 50]:
                    sigma_df = algo_df[algo_df['noise_info'].str.contains(f'sigma_{sigma}')]
                    if len(sigma_df) > 0:
                        sigmas.append(sigma)
                        avg_psnrs.append(sigma_df['psnr'].mean())
                
                if sigmas:
                    plt.plot(sigmas, avg_psnrs, 'o-', linewidth=2, markersize=8, label=algo)
            
            plt.xlabel('Noise Sigma')
            plt.ylabel('Average PSNR (dB)')
            plt.title('PSNR Comparison - Gaussian Noise')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            chart_file = output_dir / 'psnr_comparison_gaussian.png'
            plt.savefig(chart_file, dpi=150)
            print(f"高斯噪声PSNR对比图已保存到: {chart_file}")
            plt.close()
        
        # 2. 时间对比
        plt.figure(figsize=(10, 6))
        
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo]
            plt.bar(algo, algo_df['time'].mean(), label=algo)
        
        plt.xlabel('Algorithm')
        plt.ylabel('Average Time (seconds)')
        plt.title('Computation Time Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        chart_file = output_dir / 'time_comparison.png'
        plt.savefig(chart_file, dpi=150)
        print(f"时间对比图已保存到: {chart_file}")
        plt.close()
        
    except Exception as e:
        print(f"创建图表时出错: {e}")

def main():
    """主函数"""
    # 实验配置
    config = {
        'data_dir': 'data/Set14',
        'max_images': 3,
        'output_dir': 'results/experiment_fixed',
        'noise_configs': [
            # 高斯噪声
            {'type': 'gaussian', 'sigma': 10},
            {'type': 'gaussian', 'sigma': 25},
            {'type': 'gaussian', 'sigma': 50},
            # 椒盐噪声
            {'type': 'salt_pepper', 'salt_prob': 0.02, 'pepper_prob': 0.02},
            {'type': 'salt_pepper', 'salt_prob': 0.05, 'pepper_prob': 0.05},
            {'type': 'salt_pepper', 'salt_prob': 0.10, 'pepper_prob': 0.10}
        ],
        'ista_params': {
            'lambda_': 0.1,
            'max_iter': 50
        }
    }
    
    print(f"实验配置:")
    print(f"  数据目录: {config['data_dir']}")
    print(f"  最大图像数: {config['max_images']}")
    print(f"  输出目录: {config['output_dir']}")
    print(f"  噪声配置: {len(config['noise_configs'])} 种")
    
    # 1. 加载图像
    images, names = load_set14_images(config['data_dir'], config['max_images'])
    
    if not images:
        print("错误: 没有加载到图像")
        return
    
    # 2. 创建含噪图像数据集
    noisy_datasets = create_noisy_datasets(images, config['noise_configs'])
    
    if not noisy_datasets:
        print("错误: 没有创建含噪图像")
        return
    
    # 3. 运行ISTA实验
    ista_results = run_ista_experiment(
        noisy_datasets, 
        lambda_=config['ista_params']['lambda_'],
        max_iter=config['ista_params']['max_iter']
    )
    
    # 4. 运行修复的BM3D实验
    bm3d_results = run_bm3d_experiment_fixed(noisy_datasets)
    
    # 5. 分析并保存结果
    df, analysis = analyze_and_save_results(
        ista_results, bm3d_results, config['output_dir']
    )
    
    # 6. 创建对比图表
    create_comparison_charts(df, config['output_dir'])
    
    print("\n" + "="*70)
    print("修正实验完成!")
    print(f"所有结果保存在: {config['output_dir']}")
    print("="*70)

if __name__ == "__main__":
    main()