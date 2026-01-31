# run_experiments.py
#!/usr/bin/env python3
"""
一键运行所有实验的脚本
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Run denoising experiments')
    
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--images', type=str, nargs='+',
                       help='Specific images to process')
    parser.add_argument('--noise', type=str, choices=['gaussian', 'salt_pepper', 'all'],
                       default='all', help='Type of noise to test')
    parser.add_argument('--method', type=str, choices=['ista', 'bm3d', 'all'],
                       default='all', help='Denoising method to test')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--convergence', action='store_true',
                       help='Run convergence analysis')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ISTA vs BM3D Denoising Experiment")
    print("=" * 60)
    
    # 检查配置文件
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found.")
        print("Using default configuration.")
        config_path = None
    
    # 导入实验类
    from src.experiments.main_experiment import DenoisingExperiment
    
    try:
        # 创建实验实例
        experiment = DenoisingExperiment(config_path)
        
        # 如果指定了特定图像，更新配置
        if args.images:
            experiment.config['data']['target_images'] = args.images
        
        # 运行实验
        print("\nStarting experiments...")
        results = experiment.run_all_experiments()
        
        # 运行收敛性分析
        if args.convergence:
            print("\nRunning convergence analysis...")
            experiment.analyze_convergence('ppt3.png', sigma=25)
        
        print("\n" + "=" * 60)
        print("EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # 显示结果位置
        results_dir = experiment.config['output']['results_dir']
        print(f"\nResults are available in: {results_dir}/")
        print("\nDirectory structure:")
        print(f"  {results_dir}/quantitative/  - Quantitative results in JSON format")
        print(f"  {results_dir}/tables/        - CSV and LaTeX tables")
        print(f"  {results_dir}/images/        - Visualization images")
        print(f"  {results_dir}/convergence/   - Convergence analysis plots")
        
        print("\nTo view the results, you can:")
        print("  1. Check the summary report: cat results/summary_report.txt")
        print("  2. View the tables: less results/tables/gaussian_results.csv")
        print("  3. Open the visualization images in your image viewer")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()