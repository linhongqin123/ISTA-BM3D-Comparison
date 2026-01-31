"""
完整实验脚本 - 运行ISTA和BM3D对比实验
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
print("ISTA vs BM3D 完整对比实验")
print("="*70)

def load_set14_images(data_dir='data/Set14', max_images=5):
    """加载Set14数据集中的图像"""
    print(f"\n加载Set14图像...")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"错误: 数据目录不存在: {data_path}")
        return [], []
    
    # 支持的文件格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(data_path.glob(f'*{ext}'))
        image_files.extend(data_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"警告: 在 {data_path} 中未找到图像文件")
        # 创建一些测试图像
        print("创建测试图像...")
        images = [np.random.rand(256, 256) * 255 for _ in range(3)]
        images = [img.astype(np.uint8) for img in images]
        names = [f'test_{i}' for i in range(len(images))]
        return images, names
    
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
        
        # 去噪
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

def run_bm3d_experiment(noisy_datasets):
    """运行BM3D实验"""
    try:
        from src.bm3d_wrapper import BM3DDenoiser
        from src.evaluation import calculate_psnr, calculate_ssim
    except ImportError as e:
        print(f"无法导入BM3D: {e}")
        print("请运行: pip install bm3d")
        return []
    
    print("\n运行BM3D实验...")
    
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
        
        # 创建BM3D去噪器
        denoiser = BM3DDenoiser(sigma_psd=sigma_psd, verbose=False)
        
        # 去噪
        denoised = denoiser.denoise(dataset['noisy'].copy())
        
        # 计算时间
        elapsed = time.time() - start_time
        
        # 计算指标
        psnr = calculate_psnr(dataset['original'], denoised)
        ssim = calculate_ssim(dataset['original'], denoised)
        
        result = {
            'algorithm': 'BM3D',
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

def analyze_results(all_results):
    """分析实验结果"""
    print("\n" + "="*70)
    print("实验结果分析")
    print("="*70)
    
    # 转换为DataFrame
    import pandas as pd
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("没有实验结果")
        return {}
    
    # 按算法和噪声类型分组统计
    analysis = {}
    
    for algorithm in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algorithm]
        analysis[algorithm] = {}
        
        for noise_type in algo_df['noise_type'].unique():
            type_df = algo_df[algo_df['noise_type'] == noise_type]
            
            if noise_type == 'gaussian':
                # 按sigma分组
                sigmas = sorted(type_df['noise_params'].apply(lambda x: x['sigma']).unique())
                psnr_by_sigma = []
                time_by_sigma = []
                
                for sigma in sigmas:
                    sigma_df = type_df[type_df['noise_params'].apply(lambda x: x['sigma'] == sigma)]
                    psnr_by_sigma.append(sigma_df['psnr'].mean())
                    time_by_sigma.append(sigma_df['time'].mean())
                
                analysis[algorithm][noise_type] = {
                    'sigmas': sigmas,
                    'avg_psnr': psnr_by_sigma,
                    'avg_time': time_by_sigma
                }
            else:
                # 椒盐噪声
                probs = []
                for params in type_df['noise_params']:
                    probs.append(params['salt_prob'] + params['pepper_prob'])
                
                # 分组统计
                unique_probs = sorted(set(probs))
                psnr_by_prob = []
                time_by_prob = []
                
                for prob in unique_probs:
                    prob_df = type_df[[abs(p - prob) < 0.001 for p in probs]]
                    psnr_by_prob.append(prob_df['psnr'].mean())
                    time_by_prob.append(prob_df['time'].mean())
                
                analysis[algorithm][noise_type] = {
                    'probs': unique_probs,
                    'avg_psnr': psnr_by_prob,
                    'avg_time': time_by_prob
                }
    
    return analysis, df

def save_results(results_df, analysis, output_dir='results/full_experiment'):
    """保存实验结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存原始数据
    results_file = output_dir / 'raw_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"原始结果已保存到: {results_file}")
    
    # 保存分析结果
    analysis_file = output_dir / 'analysis.json'
    with open(analysis_file, 'w') as f:
        # 转换numpy数组为list以便JSON序列化
        import json
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        json.dump(analysis, f, indent=2, default=convert)
    print(f"分析结果已保存到: {analysis_file}")
    
    # 生成汇总表格
    summary = results_df.groupby(['algorithm', 'noise_type', 'noise_info']).agg({
        'psnr': ['mean', 'std', 'min', 'max'],
        'ssim': ['mean', 'std'],
        'time': ['mean', 'std']
    }).round(3)
    
    summary_file = output_dir / 'summary_statistics.csv'
    summary.to_csv(summary_file)
    print(f"汇总统计已保存到: {summary_file}")
    
    return output_dir

def print_summary(analysis, results_df):
    """打印实验总结"""
    print("\n" + "="*70)
    print("实验总结")
    print("="*70)
    
    # 总体统计
    print(f"\n总体统计:")
    print(f"  总实验数: {len(results_df)}")
    print(f"  算法: {results_df['algorithm'].unique().tolist()}")
    print(f"  噪声类型: {results_df['noise_type'].unique().tolist()}")
    
    # 各算法平均性能
    print(f"\n各算法平均性能:")
    for algorithm in results_df['algorithm'].unique():
        algo_df = results_df[results_df['algorithm'] == algorithm]
        print(f"  {algorithm}:")
        print(f"    平均PSNR: {algo_df['psnr'].mean():.2f} ± {algo_df['psnr'].std():.2f} dB")
        print(f"    平均SSIM: {algo_df['ssim'].mean():.4f} ± {algo_df['ssim'].std():.4f}")
        print(f"    平均时间: {algo_df['time'].mean():.2f} ± {algo_df['time'].std():.2f} 秒")
    
    # 最佳算法
    print(f"\n最佳PSNR算法:")
    best_psnr_idx = results_df['psnr'].idxmax()
    best_result = results_df.loc[best_psnr_idx]
    print(f"  {best_result['algorithm']} 在 {best_result['noise_info']}")
    print(f"  PSNR: {best_result['psnr']:.2f} dB, SSIM: {best_result['ssim']:.4f}")
    
    print(f"\n最快算法:")
    fastest_idx = results_df['time'].idxmin()
    fastest_result = results_df.loc[fastest_idx]
    print(f"  {fastest_result['algorithm']} 在 {fastest_result['noise_info']}")
    print(f"  时间: {fastest_result['time']:.2f} 秒")

def create_simple_visualizations(analysis, output_dir):
    """创建简单的可视化图表"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        
        # 使用非交互式后端
        matplotlib.use('Agg')
        
        output_dir = Path(output_dir)
        
        # 1. PSNR对比图（高斯噪声）
        if any('gaussian' in analysis.get(algo, {}) for algo in analysis):
            plt.figure(figsize=(10, 6))
            
            for algorithm in analysis:
                if 'gaussian' in analysis[algorithm]:
                    data = analysis[algorithm]['gaussian']
                    plt.plot(data['sigmas'], data['avg_psnr'], 'o-', 
                            linewidth=2, markersize=8, label=algorithm)
            
            plt.xlabel('Noise Sigma')
            plt.ylabel('PSNR (dB)')
            plt.title('PSNR Comparison - Gaussian Noise')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_file = output_dir / 'psnr_gaussian.png'
            plt.savefig(plot_file, dpi=150)
            print(f"高斯噪声PSNR对比图已保存到: {plot_file}")
            plt.close()
        
        # 2. 时间对比图
        plt.figure(figsize=(10, 6))
        
        for algorithm in analysis:
            if 'gaussian' in analysis[algorithm]:
                data = analysis[algorithm]['gaussian']
                plt.plot(data['sigmas'], data['avg_time'], 's--', 
                        linewidth=2, markersize=8, label=f"{algorithm} Time")
        
        plt.xlabel('Noise Sigma')
        plt.ylabel('Time (seconds)')
        plt.title('Computation Time Comparison - Gaussian Noise')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_file = output_dir / 'time_gaussian.png'
        plt.savefig(plot_file, dpi=150)
        print(f"时间对比图已保存到: {plot_file}")
        plt.close()
        
    except Exception as e:
        print(f"创建可视化图表时出错: {e}")
        print("跳过可视化部分")

def main():
    """主函数"""
    # 实验配置
    config = {
        'data_dir': 'data/Set14',
        'max_images': 3,  # 每张图像测试
        'output_dir': 'results/full_experiment',
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
    print(f"  ISTA参数: λ={config['ista_params']['lambda_']}, 迭代={config['ista_params']['max_iter']}")
    
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
    
    all_results = []
    
    # 3. 运行ISTA实验
    ista_results = run_ista_experiment(
        noisy_datasets, 
        lambda_=config['ista_params']['lambda_'],
        max_iter=config['ista_params']['max_iter']
    )
    all_results.extend(ista_results)
    
    # 4. 运行BM3D实验
    bm3d_results = run_bm3d_experiment(noisy_datasets)
    all_results.extend(bm3d_results)
    
    if not all_results:
        print("错误: 没有实验结果")
        return
    
    # 5. 分析结果
    analysis, results_df = analyze_results(all_results)
    
    # 6. 保存结果
    output_dir = save_results(results_df, analysis, config['output_dir'])
    
    # 7. 打印总结
    print_summary(analysis, results_df)
    
    # 8. 创建可视化图表
    create_simple_visualizations(analysis, output_dir)
    
    print("\n" + "="*70)
    print("实验完成!")
    print(f"所有结果保存在: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
    