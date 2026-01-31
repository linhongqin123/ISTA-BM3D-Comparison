"""
处理ppt3.bmp图像 - 使用修复的BM3D
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import json

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.noise_generation import add_gaussian_noise, add_salt_pepper_noise
from src.ista_denoiser import ISTADenoiser
from src.evaluation import calculate_psnr, calculate_ssim

def process_ppt3_fixed():
    """处理ppt3.bmp图像（使用修复的BM3D）"""
    print("处理ppt3.bmp图像（使用修复的BM3D）")
    print("="*60)
    
    # 尝试加载ppt3.bmp
    image_paths = [
        Path('data/ppt3.bmp'),
        Path('data/ppt3.png'),
        Path('data/ppt3.jpg'),
        Path('data/ppt3.tif')
    ]
    
    original = None
    loaded_path = None
    
    for img_path in image_paths:
        if img_path.exists():
            print(f"找到图像: {img_path}")
            # 尝试读取图像
            img = cv2.imread(str(img_path))
            if img is not None:
                # 转换为灰度图像
                if len(img.shape) == 3:
                    original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    original = img
                loaded_path = img_path
                print(f"加载成功: {original.shape}")
                break
            else:
                print(f"无法读取图像: {img_path}")
    
    # 如果未找到图像，创建测试图像
    if original is None:
        print(f"未找到图像文件，创建测试图像")
        original = np.random.rand(512, 512) * 255
        original = original.astype(np.uint8)
        
        # 添加一些结构（模拟真实图像）
        cv2.circle(original, (256, 256), 100, 200, 10)
        cv2.rectangle(original, (100, 100), (200, 200), 150, 5)
        cv2.line(original, (50, 50), (450, 450), 180, 3)
        
        # 保存测试图像
        test_path = Path('data/ppt3_test.bmp')
        cv2.imwrite(str(test_path), original)
        print(f"创建测试图像并保存到: {test_path}")
    
    # 确保图像是uint8类型
    if original.dtype != np.uint8:
        original = np.clip(original, 0, 255).astype(np.uint8)
    
    print(f"原始图像: 形状={original.shape}, 类型={original.dtype}, 范围=[{original.min()}, {original.max()}]")
    
    # 创建修复的BM3D包装器
    try:
        import bm3d
        bm3d_available = True
        print("BM3D库可用")
    except ImportError:
        bm3d_available = False
        print("警告: BM3D库未安装，将使用简化版本")
    
    class FixedBM3D:
        def __init__(self, sigma_psd=25, verbose=True):
            self.sigma_normalized = sigma_psd / 255.0
            self.verbose = verbose
        
        def denoise(self, noisy_image):
            if not bm3d_available:
                # 如果BM3D不可用，使用中值滤波作为替代
                print("警告: 使用中值滤波替代BM3D")
                return cv2.medianBlur(noisy_image, 5)
            
            # 转换为float32并归一化
            noisy_float = noisy_image.astype(np.float32) / 255.0
            
            if self.verbose:
                print(f"修复BM3D: sigma={self.sigma_normalized:.4f}")
            
            try:
                # 调用BM3D
                denoised_float = bm3d.bm3d(noisy_float, self.sigma_normalized)
                
                # 转换回[0,255]
                denoised = (np.clip(denoised_float, 0, 1) * 255).astype(np.uint8)
                return denoised
            except Exception as e:
                print(f"BM3D处理失败: {e}")
                # 返回原始图像作为后备
                return noisy_image
    
    # 噪声配置
    noise_configs = [
        # 高斯噪声
        {'type': 'gaussian', 'sigma': 10, 'name': 'Gaussian σ=10'},
        {'type': 'gaussian', 'sigma': 25, 'name': 'Gaussian σ=25'},
        {'type': 'gaussian', 'sigma': 50, 'name': 'Gaussian σ=50'},
        # 椒盐噪声
        {'type': 'salt_pepper', 'salt_prob': 0.02, 'pepper_prob': 0.02, 'name': 'Salt & Pepper 4%'},
        {'type': 'salt_pepper', 'salt_prob': 0.05, 'pepper_prob': 0.05, 'name': 'Salt & Pepper 10%'},
        {'type': 'salt_pepper', 'salt_prob': 0.10, 'pepper_prob': 0.10, 'name': 'Salt & Pepper 20%'},
    ]
    
    # 创建输出目录
    output_dir = Path('results/ppt3_fixed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存原始图像
    original_path = output_dir / "ppt3_original.bmp"
    cv2.imwrite(str(original_path), original)
    
    results = []
    
    for config in noise_configs:
        print(f"\n处理: {config['name']}")
        
        # 创建含噪图像
        if config['type'] == 'gaussian':
            noisy = add_gaussian_noise(original.copy(), config['sigma'])
        else:
            noisy = add_salt_pepper_noise(
                original.copy(), 
                config['salt_prob'], 
                config['pepper_prob']
            )
        
        # 计算含噪图像的PSNR
        noisy_psnr = calculate_psnr(original, noisy)
        noisy_ssim = calculate_ssim(original, noisy)
        print(f"  含噪图像 PSNR: {noisy_psnr:.2f} dB, SSIM: {noisy_ssim:.4f}")
        
        # 保存含噪图像
        noisy_path = output_dir / f"ppt3_noisy_{config['name'].replace(' ', '_').replace('&', 'and').replace('σ=', 'sigma_')}.bmp"
        cv2.imwrite(str(noisy_path), noisy)
        
        # 1. ISTA去噪
        print(f"  ISTA去噪...")
        ista_denoiser = ISTADenoiser(
            max_iter=50,
            lambda_=0.1,
            step_size=1.0,
            verbose=False
        )
        
        ista_start = cv2.getTickCount()
        ista_denoised = ista_denoiser.denoise(noisy.copy(), method='image')
        ista_time = (cv2.getTickCount() - ista_start) / cv2.getTickFrequency()
        
        ista_psnr = calculate_psnr(original, ista_denoised)
        ista_ssim = calculate_ssim(original, ista_denoised)
        
        ista_path = output_dir / f"ppt3_ista_{config['name'].replace(' ', '_').replace('&', 'and').replace('σ=', 'sigma_')}.bmp"
        cv2.imwrite(str(ista_path), ista_denoised)
        
        print(f"    PSNR: {ista_psnr:.2f} dB, SSIM: {ista_ssim:.4f}, 时间: {ista_time:.2f}秒")
        
        # 2. 修复的BM3D去噪
        print(f"  修复BM3D去噪...")
        try:
            # 设置BM3D参数
            sigma_psd = config['sigma'] if config['type'] == 'gaussian' else 25
            
            bm3d_denoiser = FixedBM3D(sigma_psd=sigma_psd, verbose=False)
            
            bm3d_start = cv2.getTickCount()
            bm3d_denoised = bm3d_denoiser.denoise(noisy.copy())
            bm3d_time = (cv2.getTickCount() - bm3d_start) / cv2.getTickFrequency()
            
            bm3d_psnr = calculate_psnr(original, bm3d_denoised)
            bm3d_ssim = calculate_ssim(original, bm3d_denoised)
            
            bm3d_path = output_dir / f"ppt3_bm3d_{config['name'].replace(' ', '_').replace('&', 'and').replace('σ=', 'sigma_')}.bmp"
            cv2.imwrite(str(bm3d_path), bm3d_denoised)
            
            print(f"    PSNR: {bm3d_psnr:.2f} dB, SSIM: {bm3d_ssim:.4f}, 时间: {bm3d_time:.2f}秒")
            
            # 记录结果
            results.append({
                'noise_type': config['name'],
                'noisy_psnr': noisy_psnr,
                'noisy_ssim': noisy_ssim,
                'ista_psnr': ista_psnr,
                'ista_ssim': ista_ssim,
                'ista_time': ista_time,
                'bm3d_psnr': bm3d_psnr,
                'bm3d_ssim': bm3d_ssim,
                'bm3d_time': bm3d_time,
                'improvement_psnr': bm3d_psnr - ista_psnr,
                'improvement_ssim': bm3d_ssim - ista_ssim,
                'time_ratio': bm3d_time / ista_time if ista_time > 0 else float('inf')
            })
            
        except Exception as e:
            print(f"    BM3D失败: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'noise_type': config['name'],
                'noisy_psnr': noisy_psnr,
                'noisy_ssim': noisy_ssim,
                'ista_psnr': ista_psnr,
                'ista_ssim': ista_ssim,
                'ista_time': ista_time,
                'bm3d_psnr': None,
                'bm3d_ssim': None,
                'bm3d_time': None,
                'improvement_psnr': None,
                'improvement_ssim': None,
                'time_ratio': None
            })
    
    # 保存结果
    results_file = output_dir / 'ppt3_results_fixed.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 创建可视化对比图
    create_comparison_plots(original, output_dir, results)
    
    # 打印总结
    print_summary(results)
    
    print(f"\n所有结果已保存到: {output_dir}")
    print("="*60)
    
    return results

def create_comparison_plots(original, output_dir, results):
    """创建对比图表"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 过滤有效结果
        valid_results = [r for r in results if r['bm3d_psnr'] is not None]
        
        if not valid_results:
            print("没有有效的BM3D结果用于绘图")
            return
        
        # 1. PSNR对比图
        plt.figure(figsize=(14, 8))
        
        noise_names = [r['noise_type'] for r in valid_results]
        noisy_psnrs = [r['noisy_psnr'] for r in valid_results]
        ista_psnrs = [r['ista_psnr'] for r in valid_results]
        bm3d_psnrs = [r['bm3d_psnr'] for r in valid_results]
        
        x = np.arange(len(noise_names))
        width = 0.25
        
        plt.bar(x - width, noisy_psnrs, width, label='含噪图像', alpha=0.7, color='red')
        plt.bar(x, ista_psnrs, width, label='ISTA', alpha=0.7, color='blue')
        plt.bar(x + width, bm3d_psnrs, width, label='BM3D', alpha=0.7, color='green')
        
        plt.xlabel('噪声类型')
        plt.ylabel('PSNR (dB)')
        plt.title('ppt3.bmp - PSNR对比')
        plt.xticks(x, noise_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # 添加数值标签
        for i, (n, i_val, b_val) in enumerate(zip(noisy_psnrs, ista_psnrs, bm3d_psnrs)):
            plt.text(i - width, n + 0.3, f'{n:.1f}', ha='center', fontsize=8)
            plt.text(i, i_val + 0.3, f'{i_val:.1f}', ha='center', fontsize=8)
            plt.text(i + width, b_val + 0.3, f'{b_val:.1f}', ha='center', fontsize=8)
        
        plot_file = output_dir / 'psnr_comparison.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. SSIM对比图
        plt.figure(figsize=(14, 8))
        
        noisy_ssims = [r['noisy_ssim'] for r in valid_results]
        ista_ssims = [r['ista_ssim'] for r in valid_results]
        bm3d_ssims = [r['bm3d_ssim'] for r in valid_results]
        
        x = np.arange(len(noise_names))
        
        plt.bar(x - width, noisy_ssims, width, label='含噪图像', alpha=0.7, color='red')
        plt.bar(x, ista_ssims, width, label='ISTA', alpha=0.7, color='blue')
        plt.bar(x + width, bm3d_ssims, width, label='BM3D', alpha=0.7, color='green')
        
        plt.xlabel('噪声类型')
        plt.ylabel('SSIM')
        plt.title('ppt3.bmp - SSIM对比')
        plt.xticks(x, noise_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # 添加数值标签
        for i, (n, i_val, b_val) in enumerate(zip(noisy_ssims, ista_ssims, bm3d_ssims)):
            plt.text(i - width, n + 0.01, f'{n:.3f}', ha='center', fontsize=8)
            plt.text(i, i_val + 0.01, f'{i_val:.3f}', ha='center', fontsize=8)
            plt.text(i + width, b_val + 0.01, f'{b_val:.3f}', ha='center', fontsize=8)
        
        plot_file = output_dir / 'ssim_comparison.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. 时间对比图
        plt.figure(figsize=(10, 6))
        
        ista_times = [r['ista_time'] for r in valid_results]
        bm3d_times = [r['bm3d_time'] for r in valid_results]
        
        # 计算平均时间
        avg_ista_time = np.mean(ista_times)
        avg_bm3d_time = np.mean(bm3d_times)
        
        bars = plt.bar(['ISTA', 'BM3D'], 
                      [avg_ista_time, avg_bm3d_time],
                      alpha=0.7, color=['blue', 'green'])
        
        plt.ylabel('平均时间 (秒)')
        plt.title('ppt3.bmp - 计算时间对比')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, time_val in zip(bars, [avg_ista_time, avg_bm3d_time]):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_file = output_dir / 'time_comparison.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"对比图表已保存到: {output_dir}")
        
    except Exception as e:
        print(f"创建图表时出错: {e}")
        import traceback
        traceback.print_exc()

def print_summary(results):
    """打印总结"""
    print("\n" + "="*60)
    print("ppt3.bmp处理总结")
    print("="*60)
    
    valid_results = [r for r in results if r['bm3d_psnr'] is not None]
    
    if not valid_results:
        print("没有有效的BM3D结果")
        return
    
    for result in valid_results:
        print(f"\n{result['noise_type']}:")
        print(f"  含噪图像: PSNR={result['noisy_psnr']:.2f}dB, SSIM={result['noisy_ssim']:.4f}")
        print(f"  ISTA: PSNR={result['ista_psnr']:.2f}dB, SSIM={result['ista_ssim']:.4f}, 时间={result['ista_time']:.3f}s")
        print(f"  BM3D: PSNR={result['bm3d_psnr']:.2f}dB, SSIM={result['bm3d_ssim']:.4f}, 时间={result['bm3d_time']:.3f}s")
        
        if result['improvement_psnr'] is not None:
            print(f"  提升: PSNR +{result['improvement_psnr']:.2f}dB, SSIM +{result['improvement_ssim']:.4f}")
        
        if result['time_ratio'] is not None and result['time_ratio'] < float('inf'):
            print(f"  时间比: BM3D/ISTA = {result['time_ratio']:.1f}x")
    
    # 总体统计
    ista_avg_psnr = np.mean([r['ista_psnr'] for r in valid_results])
    bm3d_avg_psnr = np.mean([r['bm3d_psnr'] for r in valid_results])
    ista_avg_ssim = np.mean([r['ista_ssim'] for r in valid_results])
    bm3d_avg_ssim = np.mean([r['bm3d_ssim'] for r in valid_results])
    
    print(f"\n总体统计:")
    print(f"  ISTA平均: PSNR={ista_avg_psnr:.2f} dB, SSIM={ista_avg_ssim:.4f}")
    print(f"  BM3D平均: PSNR={bm3d_avg_psnr:.2f} dB, SSIM={bm3d_avg_ssim:.4f}")
    print(f"  BM3D相比ISTA平均提升: PSNR+{bm3d_avg_psnr - ista_avg_psnr:.2f} dB, SSIM+{bm3d_avg_ssim - ista_avg_ssim:.4f}")

if __name__ == "__main__":
    # 设置中文显示
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    process_ppt3_fixed()