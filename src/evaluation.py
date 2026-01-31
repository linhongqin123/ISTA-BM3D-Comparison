"""
评估指标计算模块
计算PSNR和SSIM
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
import time

def calculate_psnr(original, denoised):
    """
    计算峰值信噪比(PSNR)
    
    参数:
        original: 原始图像
        denoised: 去噪图像
        
    返回:
        psnr_value: PSNR值 (dB)
    """
    # 确保图像为float类型
    if original.dtype != np.float32:
        original = original.astype(np.float32)
    if denoised.dtype != np.float32:
        denoised = denoised.astype(np.float32)
    
    # 如果图像在0-255范围，转换为0-1
    if original.max() > 1.0:
        original = original / 255.0
    if denoised.max() > 1.0:
        denoised = denoised / 255.0
    
    # 计算MSE
    mse = np.mean((original - denoised) ** 2)
    
    # 避免除以0
    if mse == 0:
        return float('inf')
    
    # 计算PSNR
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr

def calculate_ssim(original, denoised, multichannel=False):
    """
    计算结构相似性指数(SSIM)
    
    参数:
        original: 原始图像
        denoised: 去噪图像
        multichannel: 是否多通道
        
    返回:
        ssim_value: SSIM值
    """
    # 确保图像为float类型
    if original.dtype != np.float32:
        original = original.astype(np.float32)
    if denoised.dtype != np.float32:
        denoised = denoised.astype(np.float32)
    
    # 如果图像在0-255范围，转换为0-1
    if original.max() > 1.0:
        original = original / 255.0
    if denoised.max() > 1.0:
        denoised = denoised / 255.0
    
    # 计算SSIM
    if multichannel and len(original.shape) == 3:
        ssim_value = ssim(original, denoised, 
                         data_range=1.0, 
                         multichannel=True,
                         win_size=7)
    else:
        ssim_value = ssim(original, denoised, 
                         data_range=1.0,
                         win_size=7)
    
    return ssim_value

def calculate_metrics(original, denoised, algorithm_name="", verbose=True):
    """
    计算所有评估指标
    
    参数:
        original: 原始图像
        denoised: 去噪图像
        algorithm_name: 算法名称
        verbose: 是否打印结果
        
    返回:
        metrics_dict: 指标字典
    """
    # 确保图像维度一致
    if original.shape != denoised.shape:
        raise ValueError(f"图像形状不匹配: {original.shape} vs {denoised.shape}")
    
    # 计算指标
    psnr_value = calculate_psnr(original, denoised)
    ssim_value = calculate_ssim(original, denoised)
    
    # 计算处理时间（如果在metrics_dict中提供）
    metrics = {
        'algorithm': algorithm_name,
        'psnr': psnr_value,
        'ssim': ssim_value,
        'image_shape': original.shape
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"算法: {algorithm_name}")
        print(f"图像形状: {original.shape}")
        print(f"PSNR: {psnr_value:.2f} dB")
        print(f"SSIM: {ssim_value:.4f}")
        print(f"{'='*50}")
    
    return metrics

def compute_average_metrics(metrics_list):
    """
    计算平均指标
    
    参数:
        metrics_list: 指标列表
        
    返回:
        avg_metrics: 平均指标
    """
    if not metrics_list:
        return {}
    
    avg_metrics = {
        'avg_psnr': np.mean([m['psnr'] for m in metrics_list]),
        'avg_ssim': np.mean([m['ssim'] for m in metrics_list]),
        'std_psnr': np.std([m['psnr'] for m in metrics_list]),
        'std_ssim': np.std([m['ssim'] for m in metrics_list]),
        'num_images': len(metrics_list)
    }
    
    return avg_metrics

def save_metrics_to_csv(metrics_list, filename):
    """
    保存指标到CSV文件
    
    参数:
        metrics_list: 指标列表
        filename: 输出文件名
    """
    import pandas as pd
    
    # 转换为DataFrame
    df = pd.DataFrame(metrics_list)
    
    # 保存到CSV
    df.to_csv(filename, index=False)
    print(f"指标已保存到: {filename}")
    
    # 打印汇总统计
    print("\n汇总统计:")
    print(f"PSNR: {df['psnr'].mean():.2f} ± {df['psnr'].std():.2f} dB")
    print(f"SSIM: {df['ssim'].mean():.4f} ± {df['ssim'].std():.4f}")

def benchmark_performance(algorithm_func, test_images, algorithm_name=""):
    """
    性能基准测试
    
    参数:
        algorithm_func: 算法函数
        test_images: 测试图像列表
        algorithm_name: 算法名称
        
    返回:
        metrics_results: 指标结果列表
        times: 处理时间列表
    """
    metrics_results = []
    times = []
    
    for i, (noisy, original) in enumerate(test_images):
        print(f"处理图像 {i+1}/{len(test_images)}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 运行算法
        denoised = algorithm_func(noisy)
        
        # 记录结束时间
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        
        # 计算指标
        metrics = calculate_metrics(original, denoised, 
                                  algorithm_name=algorithm_name,
                                  verbose=False)
        metrics['time'] = elapsed
        metrics_results.append(metrics)
        
        print(f"  PSNR: {metrics['psnr']:.2f} dB, "
              f"SSIM: {metrics['ssim']:.4f}, "
              f"时间: {elapsed:.2f}秒")
    
    # 计算平均统计
    avg_time = np.mean(times)
    avg_psnr = np.mean([m['psnr'] for m in metrics_results])
    avg_ssim = np.mean([m['ssim'] for m in metrics_results])
    
    print(f"\n{algorithm_name} 算法统计:")
    print(f"平均PSNR: {avg_psnr:.2f} dB")
    print(f"平均SSIM: {avg_ssim:.4f}")
    print(f"平均处理时间: {avg_time:.2f} 秒")
    print(f"总处理时间: {sum(times):.2f} 秒")
    
    return metrics_results, times

if __name__ == "__main__":
    # 测试代码
    # 创建测试图像
    original = np.random.rand(256, 256).astype(np.float32)
    denoised = original + np.random.randn(256, 256) * 0.01
    
    # 计算指标
    metrics = calculate_metrics(original, denoised, "Test Algorithm")
    
    print(f"测试结果: {metrics}")
