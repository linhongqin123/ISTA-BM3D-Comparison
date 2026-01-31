"""
主实验脚本
协调所有模块完成实验
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import time

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from src.noise_generation import (
    add_gaussian_noise, 
    add_salt_pepper_noise,
    generate_noise_experiments
)
from src.evaluation import (
    calculate_metrics,
    benchmark_performance,
    save_metrics_to_csv
)
from src.visualization import (
    plot_image_comparison,
    plot_metrics_comparison,
    save_figure
)

class DenoisingExperiment:
    """
    去噪实验主类
    """
    
    def __init__(self, data_dir='data/Set14', output_dir='results'):
        """
        初始化实验
        
        参数:
            data_dir: 数据目录
            output_dir: 输出目录
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'tables').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        
        # 实验结果存储
        self.results = {
            'gaussian': {},
            'salt_pepper': {}
        }
        
        print(f"实验设置:")
        print(f"  数据目录: {self.data_dir}")
        print(f"  输出目录: {self.output_dir}")
    
    def load_dataset(self, max_images=None):
        """
        加载数据集
        
        参数:
            max_images: 最大加载图像数
            
        返回:
            images: 图像列表
            names: 图像名称列表
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 支持的文件格式
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        # 查找所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.data_dir.glob(f'*{ext}'))
            image_files.extend(self.data_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            raise FileNotFoundError(f"在 {self.data_dir} 中未找到图像文件")
        
        # 限制图像数量
        if max_images:
            image_files = image_files[:max_images]
        
        # 加载图像
        images = []
        names = []
        
        print(f"加载 {len(image_files)} 张图像...")
        for img_path in tqdm(image_files, desc="加载图像"):
            try:
                # 读取图像（强制转为灰度）
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"警告: 无法读取 {img_path}")
                    continue
                
                images.append(img)
                names.append(img_path.stem)
            except Exception as e:
                print(f"错误: 加载 {img_path} 时出错: {e}")
        
        if not images:
            raise RuntimeError("未成功加载任何图像")
        
        print(f"成功加载 {len(images)} 张图像")
        print(f"图像尺寸: {images[0].shape}")
        
        return images, names
    
    def prepare_noisy_images(self, original_images):
        """
        准备含噪图像
        
        参数:
            original_images: 原始图像列表
            
        返回:
            noisy_experiments: 噪声实验配置列表
        """
        print("\n准备含噪图像...")
        
        noisy_experiments = []
        
        # 噪声参数配置
        noise_configs = [
            # (类型, 参数, 名称)
            ('gaussian', {'sigma': 10}, '高斯噪声 σ=10'),
            ('gaussian', {'sigma': 25}, '高斯噪声 σ=25'),
            ('gaussian', {'sigma': 50}, '高斯噪声 σ=50'),
            ('salt_pepper', {'salt_prob': 0.02, 'pepper_prob': 0.02}, '椒盐噪声 4%'),
            ('salt_pepper', {'salt_prob': 0.05, 'pepper_prob': 0.05}, '椒盐噪声 10%'),
            ('salt_pepper', {'salt_prob': 0.1, 'pepper_prob': 0.1}, '椒盐噪声 20%'),
        ]
        
        for img_idx, original in enumerate(tqdm(original_images, desc="生成噪声")):
            for noise_type, params, name in noise_configs:
                if noise_type == 'gaussian':
                    noisy = add_gaussian_noise(original.copy(), **params)
                elif noise_type == 'salt_pepper':
                    noisy = add_salt_pepper_noise(original.copy(), **params)
                else:
                    continue
                
                noisy_experiments.append({
                    'original': original,
                    'noisy': noisy,
                    'noise_type': noise_type,
                    'noise_params': params,
                    'noise_name': name,
                    'image_idx': img_idx
                })
        
        print(f"生成 {len(noisy_experiments)} 组噪声图像")
        return noisy_experiments
    
    def run_algorithm(self, algorithm_func, algorithm_name, noisy_experiments):
        """
        运行算法
        
        参数:
            algorithm_func: 算法函数
            algorithm_name: 算法名称
            noisy_experiments: 噪声实验配置列表
            
        返回:
            results: 结果列表
        """
        print(f"\n运行 {algorithm_name} 算法...")
        
        results = []
        
        for exp in tqdm(noisy_experiments, desc=f"处理 {algorithm_name}"):
            start_time = time.time()
            
            # 运行去噪算法
            denoised = algorithm_func(exp['noisy'].copy())
            
            elapsed = time.time() - start_time
            
            # 计算指标
            metrics = calculate_metrics(
                exp['original'], 
                denoised,
                algorithm_name=algorithm_name,
                verbose=False
            )
            
            # 添加额外信息
            result = {
                **metrics,
                'noise_type': exp['noise_type'],
                'noise_params': exp['noise_params'],
                'noise_name': exp['noise_name'],
                'image_idx': exp['image_idx'],
                'time': elapsed
            }
            
            # 如果图像是ppt3.png，保存可视化结果
            if exp.get('is_ppt3', False):
                self.save_ppt3_results(exp['original'], exp['noisy'], denoised, 
                                      algorithm_name, exp['noise_name'])
            
            results.append(result)
        
        return results
    
    def save_ppt3_results(self, original, noisy, denoised, algorithm, noise_name):
        """
        保存ppt3.png的特殊结果
        
        参数:
            original: 原始图像
            noisy: 含噪图像
            denoised: 去噪图像
            algorithm: 算法名称
            noise_name: 噪声名称
        """
        # 创建保存目录
        ppt3_dir = self.output_dir / 'images' / 'ppt3'
        ppt3_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存图像
        import matplotlib.pyplot as plt
        
        fig = plot_image_comparison(
            original, 
            noisy, 
            [denoised],
            titles_list=['Original', f'Noisy ({noise_name})', f'{algorithm} Denoised']
        )
        
        # 生成安全的文件名
        safe_algorithm = algorithm.replace(' ', '_').replace('/', '_')
        safe_noise = noise_name.replace(' ', '_').replace('/', '_')
        filename = ppt3_dir / f'ppt3_{safe_algorithm}_{safe_noise}.png'
        
        save_figure(fig, str(filename))
        
        # 保存图像数据
        cv2.imwrite(str(ppt3_dir / f'ppt3_original.png'), original)
        cv2.imwrite(str(ppt3_dir / f'ppt3_{safe_algorithm}_denoised.png'), denoised)
    
    def analyze_results(self, all_results):
        """
        分析结果
        
        参数:
            all_results: 所有算法结果
            
        返回:
            analysis: 分析结果字典
        """
        print("\n分析实验结果...")
        
        analysis = {
            'summary': {},
            'by_noise_type': {},
            'by_algorithm': {}
        }
        
        # 按算法和噪声类型组织结果
        for algo_name, results in all_results.items():
            analysis['by_algorithm'][algo_name] = {
                'avg_psnr': np.mean([r['psnr'] for r in results]),
                'avg_ssim': np.mean([r['ssim'] for r in results]),
                'avg_time': np.mean([r['time'] for r in results]),
                'total_results': len(results)
            }
        
        # 按噪声类型分析
        noise_types = set()
        for results in all_results.values():
            for r in results:
                noise_types.add(r['noise_type'])
        
        for noise_type in noise_types:
            analysis['by_noise_type'][noise_type] = {}
            for algo_name, results in all_results.items():
                type_results = [r for r in results if r['noise_type'] == noise_type]
                if type_results:
                    analysis['by_noise_type'][noise_type][algo_name] = {
                        'avg_psnr': np.mean([r['psnr'] for r in type_results]),
                        'avg_ssim': np.mean([r['ssim'] for r in type_results]),
                        'avg_time': np.mean([r['time'] for r in type_results]),
                        'count': len(type_results)
                    }
        
        # 总结
        analysis['summary'] = {
            'total_algorithms': len(all_results),
            'total_experiments': sum(len(r) for r in all_results.values()),
            'noise_types': list(noise_types),
            'best_psnr_algorithm': max(
                analysis['by_algorithm'].items(),
                key=lambda x: x[1]['avg_psnr']
            )[0] if analysis['by_algorithm'] else None,
            'fastest_algorithm': min(
                analysis['by_algorithm'].items(),
                key=lambda x: x[1]['avg_time']
            )[0] if analysis['by_algorithm'] else None
        }
        
        return analysis
    
    def generate_report(self, analysis, all_results):
        """
        生成报告
        
        参数:
            analysis: 分析结果
            all_results: 所有算法结果
        """
        print("\n生成实验报告...")
        
        # 保存分析结果到JSON
        analysis_file = self.output_dir / 'analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"分析结果已保存到: {analysis_file}")
        
        # 生成汇总表格
        self.generate_summary_tables(analysis)
        
        # 生成对比图表
        self.generate_comparison_charts(all_results)
        
        # 打印汇总信息
        self.print_summary(analysis)
    
    def generate_summary_tables(self, analysis):
        """生成汇总表格"""
        import pandas as pd
        
        # 算法性能汇总表
        algo_data = []
        for algo_name, metrics in analysis['by_algorithm'].items():
            algo_data.append({
                'Algorithm': algo_name,
                'Avg PSNR (dB)': f"{metrics['avg_psnr']:.2f}",
                'Avg SSIM': f"{metrics['avg_ssim']:.4f}",
                'Avg Time (s)': f"{metrics['avg_time']:.2f}",
                'Num Experiments': metrics['total_results']
            })
        
        algo_df = pd.DataFrame(algo_data)
        algo_table_file = self.output_dir / 'tables' / 'algorithm_summary.csv'
        algo_df.to_csv(algo_table_file, index=False)
        print(f"算法汇总表已保存到: {algo_table_file}")
        
        # 噪声类型性能表
        for noise_type, algo_metrics in analysis['by_noise_type'].items():
            noise_data = []
            for algo_name, metrics in algo_metrics.items():
                noise_data.append({
                    'Algorithm': algo_name,
                    'Avg PSNR (dB)': f"{metrics['avg_psnr']:.2f}",
                    'Avg SSIM': f"{metrics['avg_ssim']:.4f}",
                    'Avg Time (s)': f"{metrics['avg_time']:.2f}"
                })
            
            noise_df = pd.DataFrame(noise_data)
            noise_table_file = self.output_dir / 'tables' / f'{noise_type}_performance.csv'
            noise_df.to_csv(noise_table_file, index=False)
            print(f"{noise_type}性能表已保存到: {noise_table_file}")
    
    def generate_comparison_charts(self, all_results):
        """生成对比图表"""
        import matplotlib.pyplot as plt
        
        # 按噪声类型分组
        noise_types = ['gaussian', 'salt_pepper']
        
        for noise_type in noise_types:
            # 收集数据
            metrics_data = {}
            
            for algo_name, results in all_results.items():
                # 过滤该噪声类型的结果
                type_results = [r for r in results if r['noise_type'] == noise_type]
                if not type_results:
                    continue
                
                # 按噪声参数分组（对于高斯噪声是sigma，对于椒盐噪声是总概率）
                if noise_type == 'gaussian':
                    # 按sigma值分组
                    sigmas = sorted(list(set(r['noise_params']['sigma'] for r in type_results)))
                    psnr_by_sigma = []
                    
                    for sigma in sigmas:
                        sigma_results = [r for r in type_results if r['noise_params']['sigma'] == sigma]
                        avg_psnr = np.mean([r['psnr'] for r in sigma_results])
                        psnr_by_sigma.append(avg_psnr)
                    
                    metrics_data[algo_name] = psnr_by_sigma
                    noise_levels = sigmas
                    xlabel = 'Sigma'
                else:
                    # 按总概率分组
                    probs = sorted(list(set(
                        r['noise_params']['salt_prob'] + r['noise_params']['pepper_prob'] 
                        for r in type_results
                    )))
                    psnr_by_prob = []
                    
                    for prob in probs:
                        prob_results = [r for r in type_results 
                                      if abs((r['noise_params']['salt_prob'] + 
                                              r['noise_params']['pepper_prob']) - prob) < 1e-5]
                        avg_psnr = np.mean([r['psnr'] for r in prob_results])
                        psnr_by_prob.append(avg_psnr)
                    
                    metrics_data[algo_name] = psnr_by_prob
                    noise_levels = [p * 100 for p in probs]  # 转换为百分比
                    xlabel = 'Noise Percentage (%)'
            
            # 绘制图表
            if metrics_data:
                fig = plot_metrics_comparison(
                    metrics_data, 
                    noise_levels,
                    metric_type='PSNR',
                    figsize=(10, 6)
                )
                
                chart_file = self.output_dir / 'plots' / f'{noise_type}_psnr_comparison.png'
                save_figure(fig, str(chart_file))
    
    def print_summary(self, analysis):
        """打印汇总信息"""
        print("\n" + "="*60)
        print("实验总结")
        print("="*60)
        
        print(f"\n总算法数: {analysis['summary']['total_algorithms']}")
        print(f"总实验数: {analysis['summary']['total_experiments']}")
        print(f"噪声类型: {', '.join(analysis['summary']['noise_types'])}")
        
        if analysis['summary']['best_psnr_algorithm']:
            print(f"\n最佳PSNR算法: {analysis['summary']['best_psnr_algorithm']}")
            best_algo = analysis['by_algorithm'][analysis['summary']['best_psnr_algorithm']]
            print(f"  平均PSNR: {best_algo['avg_psnr']:.2f} dB")
            print(f"  平均SSIM: {best_algo['avg_ssim']:.4f}")
            print(f"  平均时间: {best_algo['avg_time']:.2f} 秒")
        
        if analysis['summary']['fastest_algorithm']:
            fastest_algo = analysis['by_algorithm'][analysis['summary']['fastest_algorithm']]
            print(f"\n最快算法: {analysis['summary']['fastest_algorithm']}")
            print(f"  平均时间: {fastest_algo['avg_time']:.2f} 秒")
        
        print("\n按噪声类型分析:")
        for noise_type, algo_metrics in analysis['by_noise_type'].items():
            print(f"\n  {noise_type.upper()}:")
            for algo_name, metrics in algo_metrics.items():
                print(f"    {algo_name}: PSNR={metrics['avg_psnr']:.2f}dB, "
                      f"SSIM={metrics['avg_ssim']:.4f}, "
                      f"Time={metrics['avg_time']:.2f}s")
        
        print("\n" + "="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='图像去噪实验')
    parser.add_argument('--data_dir', type=str, default='data/Set14',
                       help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='输出目录路径')
    parser.add_argument('--max_images', type=int, default=5,
                       help='最大处理图像数')
    parser.add_argument('--run_bm3d', action='store_true',
                       help='运行BM3D算法')
    parser.add_argument('--run_ista', action='store_true',
                       help='运行ISTA算法')
    
    args = parser.parse_args()
    
    # 创建实验实例
    experiment = DenoisingExperiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    try:
        # 1. 加载数据集
        print("\n" + "="*60)
        print("步骤 1: 加载数据集")
        print("="*60)
        images, names = experiment.load_dataset(max_images=args.max_images)
        
        # 2. 准备含噪图像
        print("\n" + "="*60)
        print("步骤 2: 准备含噪图像")
        print("="*60)
        noisy_experiments = experiment.prepare_noisy_images(images)
        
        # 标记ppt3.png（如果存在）
        for i, name in enumerate(names):
            if 'ppt3' in name.lower():
                for exp in noisy_experiments:
                    if exp['image_idx'] == i:
                        exp['is_ppt3'] = True
        
        # 3. 运行算法
        print("\n" + "="*60)
        print("步骤 3: 运行去噪算法")
        print("="*60)
        
        all_results = {}
        
        # 运行ISTA算法（这里使用简单的均值滤波作为示例）
        if args.run_ista:
            from src.ista_denoiser import ista_denoise
            
            def ista_wrapper(noisy_image):
                """ISTA包装器"""
                # 这里使用示例参数，实际实现时会调整
                return ista_denoise(noisy_image, lambda_=0.1, n_iter=50, step_size=0.1)
            
            ista_results = experiment.run_algorithm(
                ista_wrapper, 
                "ISTA",
                noisy_experiments
            )
            all_results['ISTA'] = ista_results
        
        # 运行BM3D算法
        if args.run_bm3d:
            from src.bm3d_wrapper import bm3d_denoise
            
            def bm3d_wrapper(noisy_image):
                """BM3D包装器"""
                return bm3d_denoise(noisy_image, sigma_psd=25)
            
            bm3d_results = experiment.run_algorithm(
                bm3d_wrapper,
                "BM3D",
                noisy_experiments
            )
            all_results['BM3D'] = bm3d_results
        
        # 如果没有指定算法，运行示例算法
        if not (args.run_ista or args.run_bm3d):
            print("警告: 未指定算法，运行示例算法")
            
            def example_denoise(noisy_image):
                """示例去噪算法（均值滤波）"""
                return cv2.blur(noisy_image, (5, 5))
            
            example_results = experiment.run_algorithm(
                example_denoise,
                "Example_Filter",
                noisy_experiments
            )
            all_results['Example'] = example_results
        
        # 4. 分析结果
        print("\n" + "="*60)
        print("步骤 4: 分析结果")
        print("="*60)
        analysis = experiment.analyze_results(all_results)
        
        # 5. 生成报告
        print("\n" + "="*60)
        print("步骤 5: 生成报告")
        print("="*60)
        experiment.generate_report(analysis, all_results)
        
        print("\n实验完成!")
        print(f"结果保存在: {experiment.output_dir}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
